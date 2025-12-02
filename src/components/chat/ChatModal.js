// /Users/imhyeonseok/Documents/stock/web/src/components/chat/chat-ChatModal.js

import React, { useState, useRef, useEffect } from 'react';
import './ChatModal.css'; 
// [★★ 수정: BsSoundwave 제거, IoSend 추가 ★★]
import { BsArrowDown } from "react-icons/bs"; 
import { IoSend } from "react-icons/io5"; // 전송 아이콘 임포트
import { FaRobot } from 'react-icons/fa';

const ChatModal = ({ isOpen, onClose }) => {
  const [prompt, setPrompt] = useState('');
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null); // Textarea 참조 추가

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };
  useEffect(scrollToBottom, [messages, isLoading]);

  // Textarea 높이 자동 조절
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'; // 높이 초기화
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`; // 스크롤 높이만큼 설정
    }
  }, [prompt]);

  const handleSend = async (messageContent) => {
    const currentPrompt = (typeof messageContent === 'string') ? messageContent : prompt;
    if (!currentPrompt.trim() || isLoading) return; 

    const newUserMessage = { role: 'user', content: currentPrompt };
    const newMessages = [...messages, newUserMessage];

    setMessages(newMessages);
    if (typeof messageContent !== 'string') {
        setPrompt(''); // 추천 질문이 아닐 때만 입력창 비우기
    }
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: newMessages }),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.error?.message || '서버에서 오류가 발생했습니다.');
      }

      const aiMessage = await response.json();
      setMessages([...newMessages, aiMessage]);

    } catch (error) {
      console.error("Chat Error:", error);
      setMessages([
        ...newMessages,
        { role: 'assistant', content: `죄송합니다. 답변 생성 중 오류가 발생했습니다: ${error.message}` },
      ]);
    } finally {
      setIsLoading(false);
      // 메시지 전송 후 textarea 높이 리셋
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  };
  
  // 추천 질문 클릭 핸들러
  const handleSuggestionClick = (suggestion) => {
    setPrompt(suggestion); // 입력창에 텍스트 설정 (즉시 전송은 안 함)
    // 만약 클릭 즉시 전송을 원한다면 handleSend(suggestion); 사용
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  if (!isOpen) {
    return null;
  }

  return (
    <div className="chat-chat-modal-container">
      <div className="chat-chat-modal-header">
        <div className="chat-chat-header-title-group">
          <h3>Stock-Guide-AI Chat</h3>
          <span className="chat-chat-header-status"></span> {/* Online 상태 표시등 */}
        </div>
        <button onClick={onClose} className="chat-chat-modal-close-btn">
          <BsArrowDown />
        </button>
      </div>

      <div className="chat-chat-modal-content">
        
        <div className="chat-chat-messages-area">
          {messages.length === 0 && !isLoading ? (
            <div className="chat-chat-initial-state">
              <div className="chat-chat-initial-icon">
                <FaRobot />
              </div>
              <h1 className="chat-chat-greeting">무엇을 도와드릴까요?</h1>
              {/* 추천 질문 영역 */}
              <div className="chat-chat-suggestions">
                <button 
                  className="chat-chat-suggestion-btn" 
                  onClick={() => handleSuggestionClick("MACD 지표가 뭐야")}
                >
                  "MACD 지표가 뭐야?"
                </button>
                <button 
                  className="chat-chat-suggestion-btn"
                  onClick={() => handleSuggestionClick("재무데이터에는 어떤 내용들이 있어?" )}
                >
                  "재무데이터에는 어떤 내용들이 있어?"
                </button>
                <button 
                  className="chat-chat-suggestion-btn"
                  onClick={() => handleSuggestionClick("캔들 차트의 패턴에는 어떤 것들이 있어?" )}
                >
                  "캔들 차트의 패턴에는 어떤 것들이 있어?"
                </button>
              </div>
            </div>
          ) : (
            messages.map((msg, index) => (
              <div key={index} className={`chat-message ${msg.role}`}>
                {msg.content}
              </div>
            ))
          )}

          {isLoading && (
            <div className="chat-message assistant">
              <div className="chat-loading-dots">
                <span>.</span><span>.</span><span>.</span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="chat-chat-input-wrapper">
          <div className="chat-chat-input-bar">
            {/* INPUT을 TEXTAREA로 변경 */}
            <textarea
              ref={textareaRef}
              rows="1" // 초기 행은 1
              className="chat-chat-input" 
              placeholder="무엇이든 물어보세요"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={isLoading}
            />
            {/* [★★ 수정: 아이콘 및 클래스 이름 변경 ★★] */}
            <button 
              className="chat-chat-icon-btn chat-send-btn" /* 'chat-wave-btn' -> 'chat-send-btn' */
              onClick={() => handleSend()} // 인자 없이 호출
              disabled={isLoading || !prompt.trim()}
            >
              <IoSend /> {/* <BsSoundwave /> -> <IoSend /> */}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatModal;