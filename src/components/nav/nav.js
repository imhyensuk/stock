// /Users/imhyeonseok/Documents/stock/web/src/components/nav/nav.js
import React from 'react';
import { NavLink } from 'react-router-dom'; 
import './nav.css'; 

// (수정) 2. FaRobot 아이콘 import
import { FaRobot } from 'react-icons/fa';
import { 
  BiUser, 
  // BiBriefcase, (삭제)
  BiBarChartAlt2, 
  BiEnvelope, 
  BiMessageSquareDetail  
} from 'react-icons/bi';

const Nav = () => {
  return (
    <nav className="nav-sidebar">
      
      {/* 중앙 메뉴 링크 */}
      <div className="nav-links">
        {/* (수정) 1. 툴팁 영어로 변경 */}
        <NavLink to="/" end>
          <BiUser />
          <span className="nav-tooltip-text">Introduction</span>
        </NavLink>
        
        {/* (수정) 1 & 2. 아이콘 및 툴팁 변경 */}
        <NavLink to="/analysis">
          <FaRobot />
          <span className="nav-tooltip-text">Model Analysis</span>
        </NavLink>
        
        {/* (수정) 1. 툴팁 영어로 변경 */}
        <NavLink to="/stock">
          <BiBarChartAlt2 />
          <span className="nav-tooltip-text">Analysis</span>
        </NavLink>

        {/* (수정) 1. 툴팁 영어로 변경 */}
        <NavLink to="/contact">
          <BiEnvelope />
          <span className="nav-tooltip-text">Contact</span>
        </NavLink>
      </div>

    </nav>
  );
};

export default Nav;