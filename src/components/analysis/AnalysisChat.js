// /Users/imhyeonseok/Documents/stock/web/src/components/analysis/AnalysisChat.js

import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './analysis.css';

const AnalysisChat = ({ ticker, analysisResult }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const scrollRef = useRef(null);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  useEffect(() => {
    if (ticker && analysisResult) {
      triggerInitialAnalysis(ticker, analysisResult);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analysisResult]); 

  const triggerInitialAnalysis = async (ticker, result) => {
    // 한국어 시스템 메시지
    const systemMsg = `🔍 AI Agent가 ${ticker}의 데이터와 최신 뉴스를 분석 중입니다...`;
    setMessages([{ role: 'system', content: systemMsg }]);

    const prompt = `
    [시스템: 자동 분석 모드]
    사용자가 '${ticker}'를 분석했습니다.
    
    [모델 데이터]
    - 주가: $${result.current_price}
    - 상승 확률: ${(result.direction_score * 100).toFixed(1)}%
    - 예측 변동성: ${(result.predicted_volatility * 100).toFixed(2)}%
    - 시장 이상 징후: ${result.regime_anomaly_score.toFixed(2)}

    [임무]
    1. 검색 도구를 사용하여 '${ticker}'의 오늘/최신 주요 뉴스를 찾으세요.
    2. 위 모델 데이터와 뉴스를 연결하여 설명하세요.
    3. 투자 인사이트를 제공하세요.
    4. 반드시 한국어로 답변하세요.
    `;

    await sendMessageToAI(prompt, true);
  };

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    await sendMessageToAI(input, false);
    setInput('');
  };

  const sendMessageToAI = async (text, isSystemTrigger = false) => {
    if (!isSystemTrigger) {
      setMessages(prev => [...prev, { role: 'user', content: text }]);
    }
    setIsTyping(true);

    try {
      const contextMessages = messages.slice(-6).map(m => ({
        role: m.role === 'system' ? 'assistant' : m.role,
        content: m.content
      }));

      const payload = [...contextMessages, { role: 'user', content: text }];

      // language 파라미터는 더 이상 필요 없지만 호환성을 위해 'ko' 전달
      const response = await axios.post('http://localhost:8000/api/chat', {
        messages: payload,
        language: 'ko' 
      });

      const aiResponse = response.data.content || "분석 정보를 가져오는데 실패했습니다.";
      setMessages(prev => [...prev, { role: 'assistant', content: aiResponse }]);

    } catch (error) {
      console.error("AI Chat Error:", error);
      setMessages(prev => [...prev, { role: 'assistant', content: "⚠️ 서버 통신 중 오류가 발생했습니다." }]);
    } finally {
      setIsTyping(false);
    }
  };

  const renderContent = (text) => {
    return text.split('\n').map((line, i) => (
      <span key={i}>
        {line}
        <br />
      </span>
    ));
  };

  return (
    <div className="ana-chat-wrapper">
      <div className="ana-chat-header">
        <h3>AI 금융 애널리스트</h3>
        <span className="ana-status-dot"></span>
        <span className="ana-status-text">Online</span>
      </div>

      <div className="ana-chat-body">
        {messages.length === 0 && (
          <div className="ana-chat-empty">
            <p>분석을 실행하면 AI가 브리핑을 시작합니다.</p>
          </div>
        )}

        {messages.map((msg, idx) => (
          <div key={idx} className={`ana-message ${msg.role}`}>
            <div className="ana-bubble">
              {msg.role === 'system' ? (
                <div className="ana-system-msg">
                  <span className="ana-spinner-small"></span> {msg.content}
                </div>
              ) : (
                renderContent(msg.content)
              )}
            </div>
          </div>
        ))}

        {isTyping && (
          <div className="ana-message assistant">
            <div className="ana-bubble typing">
              <span>.</span><span>.</span><span>.</span>
            </div>
          </div>
        )}
        <div ref={scrollRef} />
      </div>

      <form onSubmit={handleSend} className="ana-chat-input-area">
        <input
          type="text"
          placeholder="추가 질문을 입력하세요..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={isTyping}
        />
        <button type="submit" disabled={isTyping || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  );
};

export default AnalysisChat;