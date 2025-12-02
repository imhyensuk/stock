// /Users/imhyeonseok/Documents/stock/web/src/components/analysis/analysis.js

import React from 'react';
import Nav from '../nav/nav';
import './analysis.css';

const Analysis = () => {
  return (
    <>
      <Nav />
      <div className="analysis-container">
        <div className="analysis-message-box">
          <h1>Developing Stock Analysis & Prediction Models</h1>
          <div className="loading-dots">
            <span>.</span><span>.</span><span>.</span>
          </div>
        </div>
      </div>
    </>
  );
};

export default Analysis;