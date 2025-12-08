# /Users/imhyeonseok/Documents/stock/web/backend/regime.py

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle 

# ---------------------------------------------------------
# 1. Configuration & Settings
# ---------------------------------------------------------
class Config:
    BASE_PATH = "/Users/imhyeonseok/Documents/stock/web/backend" # 경로 수정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULT_DIR = f"./results_{timestamp}"
    SEQ_LEN = 20
    BATCH_SIZE = 32
    EPOCHS = 50 
    LEARNING_RATE = 1e-4
    LATENT_DIM = 16
    D_MODEL = 64
    N_HEAD = 4
    N_LAYERS = 2
    DEMO_MODE = True 

# ---------------------------------------------------------
# 2. Data Processor
# ---------------------------------------------------------
class StockDataProcessor:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def load_and_process(self):
        if self.config.DEMO_MODE:
            print("[Info] Demo Mode: 가상 데이터를 생성합니다...")
            return self._generate_dummy_data()

        try:
            path = f"{self.config.BASE_PATH}/daily_data/A_daily_data.csv"
            if not os.path.exists(path): raise FileNotFoundError
            
            df = pd.read_csv(path, parse_dates=['Date'])
            df.set_index('Date', inplace=True)
            df = df.sort_index()
            df = self._engineer_features(df)
            return df.dropna()
            
        except FileNotFoundError:
            print("[Error] 데이터를 찾을 수 없어 데모 모드로 전환합니다.")
            return self._generate_dummy_data()

    def _engineer_features(self, df):
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Illiquidity'] = df['Log_Ret'].abs() / (df['Volume'] + 1e-9)
        df['Hist_Vol'] = df['Log_Ret'].rolling(window=20).std()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['Disparity'] = (df['Close'] - df['MA20']) / df['MA20']
        return df.dropna()

    def _generate_dummy_data(self):
        dates = pd.date_range(start="2020-01-01", end="2024-01-01")
        t = np.linspace(0, 100, len(dates))
        price = 100 + t + 10 * np.sin(t/5) + np.random.normal(0, 1, len(dates))
        data = {
            'Close': price,
            'Volume': np.abs(np.random.normal(10000, 2000, len(dates))),
            'VIX': 15 + 5 * np.sin(t/10) + np.random.normal(0, 2, len(dates)) # 이 VIX 컬럼이 문제였음
        }
        df = pd.DataFrame(data, index=dates)
        idx = int(len(df) * 0.8)
        df.iloc[idx:idx+10, 2] = 50 
        return self._engineer_features(df)

    def create_sequences(self, df):
        # [FIX] 학습에 사용할 피처를 명시적으로 지정 (5개)
        # VIX 등 불필요한 컬럼이 섞여 들어가는 것을 방지
        target_features = ['Log_Ret', 'Illiquidity', 'Hist_Vol', 'MA20', 'Disparity']
        
        # 해당 컬럼만 선택
        data = df[target_features].values
        self.feature_names = target_features
        
        print(f"[DEBUG] 학습에 사용되는 피처({len(target_features)}개): {target_features}")

        scaled_data = self.scaler.fit_transform(data)
        
        X, timestamps = [], []
        for i in range(len(scaled_data) - self.config.SEQ_LEN):
            X.append(scaled_data[i : i + self.config.SEQ_LEN])
            timestamps.append(df.index[i + self.config.SEQ_LEN])
            
        return np.array(X), np.array(timestamps), df

# ---------------------------------------------------------
# 3. Model Architecture
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 4. Trainer & Scorer
# ---------------------------------------------------------
class AnomalyDetector:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        self.loss_history = []
        self.err_mean = 0
        self.err_std = 1

    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = nn.MSELoss()(recon_x, x)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kld_loss /= x.size(0) * x.size(1)
        return recon_loss + 1e-5 * kld_loss 

    def train(self, dataloader):
        self.model.train()
        epoch_loss = 0
        for batch in dataloader:
            x = batch[0]
            self.optimizer.zero_grad()
            recon_x, mu, logvar = self.model(x)
            loss = self.loss_function(recon_x, x, mu, logvar)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        self.loss_history.append(avg_loss)
        return avg_loss

    def fit_anomaly_distribution(self, dataloader):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in dataloader:
                x = batch[0]
                recon_x, _, _ = self.model(x)
                loss = torch.mean((recon_x - x) ** 2, dim=[1, 2]) 
                losses.extend(loss.cpu().numpy())
        self.err_mean, self.err_std = norm.fit(losses)

    def detect(self, dataloader):
        self.model.eval()
        probs, raw_scores, recons = [], [], []
        with torch.no_grad():
            for batch in dataloader:
                x = batch[0]
                recon_x, _, _ = self.model(x)
                loss = torch.mean((recon_x - x) ** 2, dim=[1, 2])
                raw_scores.extend(loss.cpu().numpy())
                recons.extend(recon_x[:, -1, 0].cpu().numpy()) 
        for score in raw_scores:
            z = (score - self.err_mean) / self.err_std
            probs.append(norm.cdf(z) * 100)
        return np.array(probs), np.array(raw_scores), np.array(recons)

# ---------------------------------------------------------
# 5. Main Execution
# ---------------------------------------------------------
def main():
    print("=== Regime Detection Model Pipeline Started ===")
    config = Config()
    
    # 1. 데이터 준비
    processor = StockDataProcessor(config)
    raw_df = processor.load_and_process()
    X, timestamps, processed_df = processor.create_sequences(raw_df)
    
    dataset = TensorDataset(torch.FloatTensor(X))
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    # 2. 모델 학습
    model = TransformerVAE(input_dim=X.shape[2], config=config)
    detector = AnomalyDetector(model, config)
    
    print(f"[Training] {config.EPOCHS} Epochs 진행 중...")
    for epoch in range(config.EPOCHS):
        detector.train(dataloader)
            
    # 3. 모델 저장
    model_dir = "saved_models"
    os.makedirs(model_dir, exist_ok=True)
    
    # (1) 모델 가중치 저장
    torch.save(model.state_dict(), os.path.join(model_dir, "regime_vae.pth"))
    
    # (2) 스케일러 저장 (Pickle)
    with open(os.path.join(model_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(processor.scaler, f)
    
    print(f"✅ [SYSTEM] 모델 & 스케일러 파일 저장 완료 (Pickle 방식): {model_dir}")

if __name__ == "__main__":
    main()