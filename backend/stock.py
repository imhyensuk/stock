# /Users/imhyeonseok/Documents/stock/web/backend/stock.py

import yfinance as yf
import json
import sys
import pandas as pd
from datetime import datetime

def json_converter(o):
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

# [★★ 수정: period, start_date 인자 받도록 변경 ★★]
def get_all_stock_data(ticker, period=None, start_date=None):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # [★★ 수정: 더 많은 재무/기업가치 지표 추가 ★★]
        required_info = {
            'symbol': info.get('symbol'),
            'shortName': info.get('shortName'),
            'longName': info.get('longName'),
            'marketCap': info.get('marketCap'),
            'trailingPE': info.get('trailingPE'),
            'forwardPE': info.get('forwardPE'),
            'beta': info.get('beta'),
            'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh'),
            'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow'),
            'dividendYield': info.get('dividendYield'),
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'website': info.get('website'),
            'longBusinessSummary': info.get('longBusinessSummary'),
            'currency': info.get('currency', 'USD'),
            # [★★ 추가: EV/EBITDA, 이자보상배율 계산용 ★★]
            'enterpriseValue': info.get('enterpriseValue'),
            'ebitda': info.get('ebitda'),
            'interestExpense': info.get('interestExpense'),
            'ebit': info.get('ebit'),
        }

        # [★★ 수정: 재무제표에 'quarterly_income_stmt' 추가 ★★]
        financials = {
            'income_stmt': json.loads(stock.income_stmt.to_json(date_format='iso')),
            'quarterly_income_stmt': json.loads(stock.quarterly_income_stmt.to_json(date_format='iso')),
            'balance_sheet': json.loads(stock.balance_sheet.to_json(date_format='iso')),
            'cash_flow': json.loads(stock.cashflow.to_json(date_format='iso')),
        }

        # [★★ 수정: period와 start_date에 따라 history 호출 분기 ★★]
        hist_data = None
        if start_date:
            hist_data = stock.history(start=start_date, interval="1d", auto_adjust=False)
        else:
            # 기본값 (period가 None일 때 '1y' 같은)
            if not period:
                period = '1y'
            hist_data = stock.history(period=period, interval="1d", auto_adjust=False)
        
        if hist_data.empty:
            if start_date:
                return {"error": f"No historical data found for {ticker} since {start_date}"}
            else:
                return {"error": f"No historical data found for {ticker} with period {period}"}
        
        hist = []
        for index, row in hist_data.iterrows():
            # [★★ 수정: 캔들 차트 버그 픽 (NaN 값 필터링) ★★]
            # yfinance가 가끔 NaN을 반환할 때를 대비해 None으로 변환
            hist.append({
                'date': index.strftime('%Y-%m-%d'),
                'open': round(row['Open'], 2) if pd.notna(row['Open']) else None,
                'high': round(row['High'], 2) if pd.notna(row['High']) else None,
                'low': round(row['Low'], 2) if pd.notna(row['Low']) else None,
                'close': round(row['Close'], 2) if pd.notna(row['Close']) else None,
                'adjClose': round(row['Adj Close'], 2) if pd.notna(row['Adj Close']) else None,
                'volume': int(row['Volume']) if pd.notna(row['Volume']) else None,
            })

        quote = {
            'symbol': info.get('symbol'),
            'name': info.get('shortName', ticker),
            'price': info.get('currentPrice') or info.get('regularMarketPrice'),
            'change': info.get('regularMarketChange'),
            'changePercent': info.get('regularMarketChangePercent') * 100 if info.get('regularMarketChangePercent') else 0,
        }
        
        return {
            "info": required_info,
            "financials": financials,
            "hist": hist,
            "quote": quote
        }

    except Exception as e:
        return {"error": f"An error occurred while fetching data for {ticker}: {str(e)}"}


if __name__ == '__main__':
    # [★★ 수정: 인자 파싱 로직 변경 ★★]
    if len(sys.argv) > 2:
        ticker_symbol = sys.argv[1]
        
        if ticker_symbol.isdigit() and len(ticker_symbol) == 6:
            ticker_symbol = f"{ticker_symbol}.KS"

        period_val = None
        start_date_val = None

        # [★★ 신규: --period 또는 --start 플래그 파싱 ★★]
        if sys.argv[2] == '--period':
            if len(sys.argv) > 3:
                period_val = sys.argv[3]
                if period_val == 'all':
                    period_val = 'max'
            else:
                print(json.dumps({"error": "--period flag requires a value"}))
                sys.exit(1)
        
        elif sys.argv[2] == '--start':
            if len(sys.argv) > 3:
                start_date_val = sys.argv[3]
            else:
                print(json.dumps({"error": "--start flag requires a value"}))
                sys.exit(1)
        
        else: # [★★ 수정: 레거시 호출 방식 (플래그 없음) ★★]
            period_val = sys.argv[2]
            if period_val == 'all':
                period_val = 'max'
        
        # [★★ 수정: 인자를 get_all_stock_data로 전달 ★★]
        result = get_all_stock_data(ticker_symbol, period=period_val, start_date=start_date_val)
        
        try:
            print(json.dumps(result, default=json_converter))
        except TypeError as e:
            error_msg = {"error": f"JSON serialization error in Python: {e}"}
            print(json.dumps(error_msg))
    else:
        # [★★ 수정: 오류 메시지 ★★]
        print(json.dumps({"error": "No ticker symbol or period/start_date provided"}))