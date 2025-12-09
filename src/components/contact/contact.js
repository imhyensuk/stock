// /Users/imhyeonseok/Documents/stock/web/src/components/contact/contact.js
import React, { useState, useRef } from 'react';
import axios from 'axios'; // [추가] Axios import
import Nav from '../nav/nav';
import './contact.css';
// BiCopy 아이콘 import

import { BiUser, BiEnvelope, BiMap, BiCopy } from 'react-icons/bi';
import { FaGithub } from 'react-icons/fa';

const API_URL = (process.env.REACT_APP_API_URL || 'http://localhost:8000').replace(/\/$/, '');

const Contact = () => {
  // 모달 팝업 상태 관리
  const [isModalOpen, setIsModalOpen] = useState(false);
  // 중복 클릭 시 타이머 초기화를 위한 ref
  const modalTimerRef = useRef(null);

  // [추가] 폼 데이터 상태 관리
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    subject: '',
    message: ''
  });

  // [추가] 입력 핸들러
  const handleChange = (e) => {
    const { id, value } = e.target;
    setFormData(prevState => ({
      ...prevState,
      [id]: value
    }));
  };

  // [추가] 폼 제출 핸들러
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    try {
      // 서버로 POST 요청
      const response = await axios.post(`${API_URL}/api/contact`, formData);
      
      if (response.status === 201) {
        alert("Message sent successfully!");
        setFormData({ name: '', email: '', subject: '', message: '' }); // 폼 초기화
      }
    } catch (error) {
      console.error("Error sending message:", error);
      alert("Failed to send message. Please try again.");
    }
  };

  // 클립보드 복사 핸들러
  const handleCopy = (text) => {
    // 이미 타이머가 실행 중이면 초기화
    if (modalTimerRef.current) {
      clearTimeout(modalTimerRef.current);
    }

    navigator.clipboard.writeText(text).then(() => {
      // 복사 성공 시
      setIsModalOpen(true);
      
      // 3초 후에 모달을 닫는 타이머 설정
      modalTimerRef.current = setTimeout(() => {
        setIsModalOpen(false);
      }, 3000); // 3초
    }).catch(err => {
      // 복사 실패 시
      console.error("Failed to copy text: ", err);
    });
  };

  return (
    <>
      <Nav />
      <div className="contact-container">
        
        {/* --- 왼쪽 열 (폼) --- */}
        <div className="contact-left">
          <header className="contact-header">
            <h1>Get in Touch</h1>
            <p>Feel free to reach out for project inquiries, collaborations, or any questions.</p>
          </header>
          
          <form className="contact-form" onSubmit={handleSubmit}>
            <div className="contact-form-row"> {/* form-row 수정 */}
              <div className="contact-form-group"> {/* form-group 수정 */}
                <label htmlFor="name">Name</label>
                <input 
                  type="text" 
                  id="name" 
                  placeholder="Enter your name" 
                  value={formData.name} 
                  onChange={handleChange}
                  required 
                />
              </div>
              <div className="contact-form-group"> {/* form-group 수정 */}
                <label htmlFor="email">Email</label>
                <input 
                  type="email" 
                  id="email" 
                  placeholder="Enter your email" 
                  value={formData.email} 
                  onChange={handleChange}
                  required 
                />
              </div>
            </div>
            
            <div className="contact-form-group"> {/* form-group 수정 */}
              <label htmlFor="subject">Subject</label>
              <input 
                type="text" 
                id="subject" 
                placeholder="Enter the subject" 
                value={formData.subject} 
                onChange={handleChange}
                required 
              />
            </div>
            
            <div className="contact-form-group"> {/* form-group 수정 */}
              <label htmlFor="message">Message</label>
              <textarea 
                id="message" 
                rows="8" 
                placeholder="Enter your message" 
                value={formData.message} 
                onChange={handleChange}
                required
              ></textarea>
            </div>
            
            <button type="submit" className="contact-submit-btn"> {/* submit-btn 수정 */}
              Send Message
            </button>
          </form>
        </div>

        {/* --- 오른쪽 열 (정보) --- */}
        <div className="contact-right">
          <header className="contact-header">
            <h2>Developer Info</h2>
          </header>
          
          <ul className="contact-info-list">
            <li className="contact-info-item">
              <span className="contact-info-item-icon"><BiUser /></span> {/* info-item-icon 수정 */}
              <div className="contact-info-item-text"> {/* info-item-text 수정 */}
                <span>Name</span>
                <p>
                  IHS
                  <button onClick={() => handleCopy('Developer Name')} className="contact-copy-btn"> {/* copy-btn 수정 */}
                    <BiCopy />
                  </button>
                </p>
              </div>
            </li>
            <li className="contact-info-item">
              <span className="contact-info-item-icon"><BiEnvelope /></span> {/* info-item-icon 수정 */}
              <div className="contact-info-item-text"> {/* info-item-text 수정 */}
                <span>Email</span>
                <p>
                  <a href="mailto:email@example.com">him21176@gmail.com</a>
                  <button onClick={() => handleCopy('email@example.com')} className="contact-copy-btn"> {/* copy-btn 수정 */}
                    <BiCopy />
                  </button>
                </p>
              </div>
            </li>
            <li className="contact-info-item">
              <span className="contact-info-item-icon"><FaGithub /></span> {/* info-item-icon 수정 */}
              <div className="contact-info-item-text"> {/* info-item-text 수정 */}
                <span>Github</span>
                <p>
                  <a href="https://github.com" target="_blank" rel="noopener noreferrer">github.com/username</a>
                  <button onClick={() => handleCopy('github.com/username')} className="contact-copy-btn"> {/* copy-btn 수정 */}
                    <BiCopy />
                  </button>
                </p>
              </div>
            </li>
            <li className="contact-info-item">
              <span className="contact-info-item-icon"><BiMap /></span> {/* info-item-icon 수정 */}
              <div className="contact-info-item-text"> {/* info-item-text 수정 */}
                <span>Address</span>
                <p>
                  Gangdong, Seoul, Republic of Korea
                  <button onClick={() => handleCopy('Seoul, Republic of Korea')} className="contact-copy-btn"> {/* copy-btn 수정 */}
                    <BiCopy />
                  </button>
                </p>
              </div>
            </li>
          </ul>
        </div>

      </div>
      
      {/* 복사 완료 모달 팝업 */}
      <div className={`contact-copy-modal ${isModalOpen ? 'visible' : ''}`}> {/* copy-modal 수정 */}
        <p>Copied Successfully</p>
        {isModalOpen && ( // 모달이 보일 때만 프로그레스 바 렌더링 (애니메이션 재시작)
          <div className="contact-modal-progress-bar"> {/* modal-progress-bar 수정 */}
            <div className="contact-modal-progress-bar-inner"></div> {/* modal-progress-bar-inner 수정 */}
          </div>
        )}
      </div>
    </>
  );
};

export default Contact;