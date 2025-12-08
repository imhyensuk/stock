# /Users/imhyeonseok/Documents/stock/web/backend/model.py

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
import torch
import torch.nn as nn
from datetime import datetime, timedelta

# [Mac/Linux 충돌 방지] OpenMP 설정
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
DEVICE = "cpu"

# ==========================================
# 1. PyTorch Model Architecture (Regime)
# ==========================================
class Config:
    SEQ_LEN = 20
    LATENT_DIM = 16
    D_MODEL = 64
    N_HEAD = 4
    N_LAYERS = 2

class TransformerVAE(nn.Module):
    def __init__(self, input_dim, config):
        super(TransformerVAE, self).__init__()
        self.config = config
        self.encoder_linear = nn.Linear(input_dim, config.D_MODEL)
        enc_layer = nn.TransformerEncoderLayer(d_model=config.D_MODEL, nhead=config.N_HEAD, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=config.N_LAYERS)
        self.mu_head = nn.Linear(config.D_MODEL, config.LATENT_DIM)
        self.logvar_head = nn.Linear(config.D_MODEL, config.LATENT_DIM)
        self.decoder_linear = nn.Linear(config.LATENT_DIM, config.D_MODEL * config.SEQ_LEN)
        dec_layer = nn.TransformerEncoderLayer(d_model=config.D_MODEL, nhead=config.N_HEAD, batch_first=True)
        self.transformer_decoder = nn.TransformerEncoder(dec_layer, num_layers=config.N_LAYERS)
        self.output_linear = nn.Linear(config.D_MODEL, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x_emb = self.encoder_linear(x)
        encoded = self.transformer_encoder(x_emb)
        last_hidden = encoded[:, -1, :] 
        mu = self.mu_head(last_hidden)
        logvar = self.logvar_head(last_hidden)
        z = self.reparameterize(mu, logvar)
        z_expanded = self.decoder_linear(z).view(-1, self.config.SEQ_LEN, self.config.D_MODEL)
        decoded = self.transformer_decoder(z_expanded)
        reconstructed = self.output_linear(decoded)
        return reconstructed, mu, logvar

# ==========================================
# 2. 데이터 전처리 파이프라인
# ==========================================
def fetch_and_preprocess(ticker):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    
    # [Fix] auto_adjust=True 경고 해결
    df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
    
    if df.empty: raise ValueError("No data found")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    if 'Close' not in df.columns: return None
    
    # 컬럼 표준화
    df = df.rename(columns={"Adj Close": "Close"}) 
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    for c in required:
        if c not in df.columns:
            if c == 'Volume': df[c] = 0
            else: df[c] = df['Close']
            
    df = df[required].dropna()

    # --- [Feature Engineering] ---
    
    # 1. 기본 지표
    df['daily_return'] = df['Close'].pct_change()
    df['daily_range_pct'] = (df['High'] - df['Low']) / (df['Open'] + 1e-8)
    
    # 2. GARCH 대체
    df['garch_vol'] = df['Close'].pct_change().rolling(20).std()
    df['vol_surprise'] = df['daily_range_pct'] / (df['garch_vol'] + 1e-8)
    df['garch_trend'] = df['garch_vol'] / (df['garch_vol'].shift(5) + 1e-8)
    df['garch_div'] = df['garch_vol'] / (df['garch_vol'].rolling(60).mean() + 1e-8)
    df['range_ma5'] = df['daily_range_pct'].rolling(5).mean()

    # 3. 기술적 지표
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain/(loss + 1e-8)))
    
    exp12 = df['Close'].ewm(span=12).mean()
    exp26 = df['Close'].ewm(span=26).mean()
    df['macd'] = exp12 - exp26
    
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift(1)).abs(),
        (df['Low'] - df['Close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    df['dist_ma20'] = df['Close'] / (df['Close'].rolling(20).mean() + 1e-8)
    df['vol_chg'] = df['Volume'].pct_change()
    df['upper_shadow'] = df['High'] - df[['Close', 'Open']].max(axis=1)

    # 4. Regime용 피처 (중요: regime.py와 이름/순서 일치해야 함)
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Illiquidity'] = df['Log_Ret'].abs() / (df['Volume'] + 1e-9)
    df['Hist_Vol'] = df['Log_Ret'].rolling(window=20).std()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Disparity'] = (df['Close'] - df['MA20']) / df['MA20']

    # 5. Z-Score 정규화 (LGBM용)
    target_cols = ['rsi', 'macd', 'atr', 'dist_ma20', 'vol_chg', 'upper_shadow', 
                   'range_ma5', 'garch_vol', 'vol_surprise', 'garch_trend', 'garch_div']
    
    z_cols_list = []
    for col in target_cols:
        if col in df.columns:
            roll_mean = df[col].rolling(60).mean()
            roll_std = df[col].rolling(60).std()
            z_col_name = f'{col}_z'
            df[z_col_name] = (df[col] - roll_mean) / (roll_std + 1e-8)
            z_cols_list.append(z_col_name)

    # 6. Lag Features
    for col in z_cols_list:
        df[f'{col}_lag1'] = df[col].shift(1)

    return df.dropna()

def align_features(df_last_row, model_booster):
    model_features = model_booster.feature_name()
    input_data = []
    for feat in model_features:
        if feat in df_last_row.columns:
            input_data.append(df_last_row[feat].item())
        else:
            input_data.append(0.0)
    return np.array([input_data])

# ==========================================
# 3. 메인 실행 로직
# ==========================================
def main(ticker):
    result = {
        "ticker": ticker,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "current_price": 0,
        "direction_score": 0.0,
        "predicted_volatility": 0.0,
        "regime_anomaly_score": 0.0,
        "error": None
    }

    try:
        # 1. 데이터 준비
        df = fetch_and_preprocess(ticker)
        if df is None: raise ValueError("데이터 전처리 실패")
        
        last_row = df.iloc[[-1]]
        result['current_price'] = float(last_row['Close'].item())
        result['date'] = last_row.index[-1].strftime("%Y-%m-%d")

        # 2. 모델 로드 및 추론
        
        # (A) Direction Model
        dir_path = os.path.join(MODEL_DIR, "lgbm_direction.txt")
        if os.path.exists(dir_path):
            model_dir = lgb.Booster(model_file=dir_path)
            X_input = align_features(last_row, model_dir)
            pred = model_dir.predict(X_input)
            result['direction_score'] = float(pred[0])

        # (B) Volatility Model
        vol_path = os.path.join(MODEL_DIR, "lgbm_volatility.txt")
        if os.path.exists(vol_path):
            model_vol = lgb.Booster(model_file=vol_path)
            X_input = align_features(last_row, model_vol)
            pred_log = model_vol.predict(X_input)
            result['predicted_volatility'] = float(np.expm1(pred_log[0]))

        # (C) Regime Model (PyTorch VAE)
        vae_path = os.path.join(MODEL_DIR, "regime_vae.pth")
        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

        if os.path.exists(vae_path) and os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            
            # [FIX] 에러 수정: Scaler가 피처 이름을 모를 경우를 대비해 수동 지정
            # regime.py에서 생성하는 피처 목록 (순서 중요)
            REGIME_FEATURES = ['Log_Ret', 'Illiquidity', 'Hist_Vol', 'MA20', 'Disparity']
            
            # scaler 객체에 feature_names_in_ 속성이 있으면 쓰고, 없으면 수동 리스트 사용
            req_feats = getattr(scaler, 'feature_names_in_', REGIME_FEATURES)
            
            seq_len = Config.SEQ_LEN
            
            if all(f in df.columns for f in req_feats) and len(df) >= seq_len:
                # DataFrame으로 전달하면 경고가 뜰 수 있으므로 values로 변환
                raw_data = df.iloc[-seq_len:][req_feats].values
                scaled_data = scaler.transform(raw_data)
                
                x_tensor = torch.FloatTensor(scaled_data).unsqueeze(0).to(DEVICE)
                
                # 모델 input_dim도 scaler에서 가져오거나 수동 지정
                input_dim = getattr(scaler, 'n_features_in_', len(req_feats))
                
                model_vae = TransformerVAE(input_dim=input_dim, config=Config())
                model_vae.load_state_dict(torch.load(vae_path, map_location=DEVICE))
                model_vae.to(DEVICE)
                model_vae.eval()

                with torch.no_grad():
                    recon, _, _ = model_vae(x_tensor)
                    loss = torch.mean((recon - x_tensor) ** 2).item()
                
                result['regime_anomaly_score'] = float(loss) * 100
            else:
                result['regime_anomaly_score'] = -1.0 

    except Exception as e:
        result['error'] = str(e)

    print(json.dumps(result))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ticker_symbol = sys.argv[1]
        if ticker_symbol.isdigit() and len(ticker_symbol) == 6:
            ticker_symbol = f"{ticker_symbol}.KS"
        main(ticker_symbol)
    else:
        print(json.dumps({"error": "Ticker argument missing"}))