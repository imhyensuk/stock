// /Users/imhyeonseok/Documents/stock/web/src/App.js
import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Main from './components/main/main'; 
// import Chat from './components/chat/chat'; // [★★ 삭제 ★★]
import Stock from './components/stock/stock'; 
import Contact from './components/contact/contact'; 
import Analysis from './components/analysis/analysis';

function App() {
  return (
    <BrowserRouter>
      <div className="App">
        <Routes>
          <Route path="/" element={<Main />} />
          {/* <Route path="/chat" element={<Chat />} /> */}{/* [★★ 삭제 ★★] */}
          <Route path="/stock" element={<Stock />} />
          <Route path="/contact" element={<Contact />} />
          <Route path="/analysis" element={<Analysis />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;