// /Users/imhyeonseok/Documents/stock/web/src/components/analysis/analysis.js

import React, { useState } from 'react';
import axios from 'axios';
import Nav from '../nav/nav';
import AnalysisChat from './AnalysisChat';
import './analysis.css';

const API_URL = (process.env.REACT_APP_API_URL || 'http://localhost:8000').replace(/\/$/, '');

const Analysis = () => {
  const [ticker, setTicker] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleAnalyze = async (e) => {
    e.preventDefault();
    if (!ticker) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.get(`${API_URL}/api/predict/${ticker}`);
      if (response.data.error) {
        setError(response.data.error);
      } else {
        setResult(response.data);
      }
    } catch (err) {
      setError("Server connection failed or execution error.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const getDirectionColor = (score) => {
    if (score > 0.55) return '#4caf50'; 
    if (score < 0.45) return '#f44336'; 
    return '#ff9800'; 
  };

  return (
    <>
      <Nav />
      <div className="ana-container">
        <div className="ana-layout">
          
          {/* ================= LEFT COLUMN ================= */}
          <div className="ana-col-left">
            <h1 className="ana-title">AI Analysis</h1>
            
            <div className="ana-input-box">
              <form onSubmit={handleAnalyze} className="ana-search-form">
                <input 
                  type="text" 
                  placeholder="Ticker (e.g. NVDA)" 
                  value={ticker}
                  onChange={(e) => setTicker(e.target.value.toUpperCase())}
                  className="ana-input"
                />
                <button type="submit" disabled={loading} className="ana-btn">
                  {loading ? 'Processing...' : 'Run'}
                </button>
              </form>
              {error && <div className="ana-error-msg">{error}</div>}
            </div>

            {loading && (
              <div className="ana-loading-container">
                <div className="ana-loader"></div>
                <p>Running Neural Networks...</p>
              </div>
            )}

            {!loading && result && (
              <div className="ana-summary-cards">
                <div className="ana-header-info">
                  <h2>{result.ticker}</h2>
                  <p className="ana-price">${result.current_price.toFixed(2)}</p>
                  <p className="ana-date">{result.date}</p>
                </div>

                <div className="ana-card" style={{ borderLeft: `5px solid ${getDirectionColor(result.direction_score)}` }}>
                  <h4>Direction</h4>
                  <div className="ana-score" style={{ color: getDirectionColor(result.direction_score) }}>
                    {(result.direction_score * 100).toFixed(1)}%
                  </div>
                  <p className="ana-desc">
                    {result.direction_score > 0.55 ? "Strong Buy" : result.direction_score < 0.45 ? "Sell Signal" : "Neutral"}
                  </p>
                </div>

                <div className="ana-card">
                  <h4>Volatility</h4>
                  <div className="ana-score text-blue">
                    {(result.predicted_volatility * 100).toFixed(2)}%
                  </div>
                  <p className="ana-desc">Daily Risk</p>
                </div>

                <div className="ana-card">
                  <h4>Regime</h4>
                  <div className="ana-score text-purple">
                    {result.regime_anomaly_score.toFixed(2)}
                  </div>
                  <p className="ana-desc">Stability Index</p>
                </div>
              </div>
            )}
          </div>

          {/* ================= RIGHT COLUMN (CHAT) ================= */}
          <div className="ana-col-right">
            <AnalysisChat 
              ticker={result?.ticker} 
              analysisResult={result} 
            />
          </div>

        </div>
      </div>
    </>
  );
};

export default Analysis;