// /Users/imhyeonseok/Documents/stock/web/backend/server.js

import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import { exec } from 'child_process';
import axios from 'axios';
import { createRequire } from 'module';
import Groq from 'groq-sdk';
import mongoose from 'mongoose';
import path from 'path';           // [추가] 경로 처리를 위한 모듈
import { fileURLToPath } from 'url'; // [추가] ES Modules에서 경로 처리를 위한 모듈

const require = createRequire(import.meta.url);
const technicalindicators = require('technicalindicators');

// [추가] __dirname, __filename 설정 (ES Modules 환경)
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(cors());
app.use(express.json({ limit: '10mb' }));

const NODE_PORT = process.env.NODE_PORT || 8000;

// --- MongoDB 연결 및 스키마 정의 ---
mongoose.connect(process.env.MONGO_URI)
  .then(() => console.log('✅ MongoDB Connected'))
  .catch(err => console.error('❌ MongoDB Connection Error:', err));

// Contact 스키마 정의
const contactSchema = new mongoose.Schema({
  name: { type: String, required: true },
  email: { type: String, required: true },
  subject: { type: String, required: true },
  message: { type: String, required: true },
  createdAt: { type: Date, default: Date.now }
});

const Contact = mongoose.model('Contact', contactSchema);
// ----------------------------------------

const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY
});


// --- [헬퍼 함수들: 수동 지표 계산] ---

/**
 * 표준 편차 (Standard Deviation)
 */
function calculateSD(data, period) {
  let results = [];
  if (data.length < period) return results;
  
  for (let i = period - 1; i < data.length; i++) {
    const slice = data.slice(i - period + 1, i + 1);
    const mean = slice.reduce((a, b) => a + b) / period;
    const variance = slice.reduce((a, b) => a + (b - mean) ** 2, 0) / period;
    results.push(Math.sqrt(variance));
  }
  return results;
}

/**
 * 변화율 (Rate of Change)
 */
function calculateROC(data, period) {
  let results = [];
  if (data.length < period) return results;

  for (let i = period; i < data.length; i++) {
    const prev = data[i - period];
    if (prev === 0) { // 0으로 나누기 방지
      results.push(null);
    } else {
      results.push(((data[i] - prev) / prev) * 100);
    }
  }
  return results;
}

/**
 * 스토캐스틱 %K (Stochastic %K)
 */
function calculateK(high, low, close, kPeriod) {
  let k = [];
  if (close.length < kPeriod) return k;

  for (let i = kPeriod - 1; i < close.length; i++) {
    const sliceH = high.slice(i - kPeriod + 1, i + 1);
    const sliceL = low.slice(i - kPeriod + 1, i + 1);
    const highestHigh = Math.max(...sliceH);
    const lowestLow = Math.min(...sliceL);
    const kVal = ((close[i] - lowestLow) / (highestHigh - lowestLow)) * 100;
    k.push(kVal);
  }
  return k;
}

/**
 * OBV (On-Balance Volume)
 */
function calculateOBV(close, volume) {
  let obv = [0]; // OBV는 0에서 시작
  for (let i = 1; i < close.length; i++) {
    if (close[i] > close[i-1]) {
      obv.push(obv[i-1] + volume[i]);
    } else if (close[i] < close[i-1]) {
      obv.push(obv[i-1] - volume[i]);
    } else {
      obv.push(obv[i-1]);
    }
  }
  return obv;
}

/**
 * VWMA (Volume Weighted Moving Average)
 */
function calculateVWMA(close, volume, period) {
  let results = [];
  if (close.length < period) return results;

  for (let i = period - 1; i < close.length; i++) {
    const sliceC = close.slice(i - period + 1, i + 1);
    const sliceV = volume.slice(i - period + 1, i + 1);
    
    let sumPriceVol = 0;
    let sumVol = 0;
    
    for (let j = 0; j < period; j++) {
      sumPriceVol += sliceC[j] * sliceV[j];
      sumVol += sliceV[j];
    }
    
    if (sumVol === 0) {
      results.push(null); // 0으로 나누기 방지
    } else {
      results.push(sumPriceVol / sumVol);
    }
  }
  return results;
}


// --- [헬퍼 함수들: 데이터 포맷팅] ---
const padArrayStart = (arr, targetLength) => {
  if (arr.length >= targetLength) return arr;
  return Array(targetLength - arr.length).fill(null).concat(arr);
};

const formatLine = (hist, columnName = 'close') => hist.map(row => ({ x: row.date, y: row[columnName] }));
const formatBar = (hist, columnName = 'volume') => hist.map(row => ({ x: row.date, y: row[columnName] }));

// --- [데이터 처리 함수: processStockData] ---
const processStockData = (rawData, period) => {
  const { info, financials, hist: fullHist, quote } = rawData;

  const hist = fullHist
    .filter(row => row.open != null && row.high != null && row.low != null && row.close != null && row.volume != null);
    
  if (!hist || hist.length === 0) throw new Error('Historical data for the selected period not found.');

  // 기술적 분석 (TA) 계산
  const closePrices = hist.map(row => row.close);
  const highPrices = hist.map(row => row.high);
  const lowPrices = hist.map(row => row.low);
  const openPrices = hist.map(row => row.open);
  const volume = hist.map(row => row.volume);
  const totalLength = hist.length;

  const sma20 = padArrayStart(technicalindicators.SMA.calculate({ period: 20, values: closePrices }), totalLength);
  const sma50 = padArrayStart(technicalindicators.SMA.calculate({ period: 50, values: closePrices }), totalLength);
  const macdData = technicalindicators.MACD.calculate({ values: closePrices, fastPeriod: 12, slowPeriod: 26, signalPeriod: 9, SimpleMAOscillator: false, SimpleMASignal: false });
  const macd = {
    MACD: padArrayStart(macdData.map(d => d.MACD), totalLength),
    signal: padArrayStart(macdData.map(d => d.signal), totalLength),
    histogram: padArrayStart(macdData.map(d => d.histogram), totalLength),
  };
  const bollingerData = technicalindicators.BollingerBands.calculate({ period: 20, values: closePrices, stdDev: 2 });
  const bollinger = {
    upper: padArrayStart(bollingerData.map(d => d.upper), totalLength),
    middle: padArrayStart(bollingerData.map(d => d.middle), totalLength),
    lower: padArrayStart(bollingerData.map(d => d.lower), totalLength),
  };
  const rsiData = padArrayStart(technicalindicators.RSI.calculate({ values: closePrices, period: 14 }), totalLength);

  // 수동 계산 지표들
  const dailyPriceRange = hist.map(row => ({ x: row.date, y: row.high - row.low }));
  const dailyChange = hist.map((row, i) => {
    if (i === 0) return { x: row.date, y: 0 };
    const prevClose = hist[i-1].close;
    if (prevClose === 0) return { x: row.date, y: 0 };
    return { x: row.date, y: ((row.close - prevClose) / prevClose) * 100 };
  });
  const basePrice = hist[0].close;
  const cumulativeReturn = hist.map(row => {
    if (basePrice === 0) return { x: row.date, y: 0 };
    return { x: row.date, y: ((row.close - basePrice) / basePrice) * 100 };
  });
  const roc = padArrayStart(calculateROC(closePrices, 10), totalLength);
  const volatility = padArrayStart(calculateSD(closePrices, 20), totalLength);
  const kData = calculateK(highPrices, lowPrices, closePrices, 14);
  const dData = technicalindicators.SMA.calculate({ period: 3, values: kData });
  const dDataPadded = Array(kData.length - dData.length).fill(null).concat(dData);
  const stochastic = {
    k: padArrayStart(kData, totalLength),
    d: padArrayStart(dDataPadded, totalLength) 
  };
  const vma = padArrayStart(calculateVWMA(closePrices, volume, 20), totalLength);
  const obv = calculateOBV(closePrices, volume);

  const haInput = {
    open: openPrices,
    high: highPrices,
    low: lowPrices,
    close: closePrices
  };
  const haDataCalculated = technicalindicators.HeikinAshi.calculate(haInput);

  const haHist = hist.map((row, i) => {
    return {
      date: row.date,
      open: haDataCalculated.open[i],
      high: haDataCalculated.high[i],
      low: haDataCalculated.low[i],
      close: haDataCalculated.close[i],
    };
  });


  // 차트 포매팅
  const labels = hist.map(row => row.date);
  const lineStyle = (color, width = 1.5) => ({ borderColor: color, borderWidth: width, fill: false, pointRadius: 0 });
  
  const createCandleChart = (data) => {
    const wicks = { 
      label: 'Wicks', 
      data: data.map(row => ({ x: row.date, y: [row.low, row.high] })), 
      type: 'bar', 
      barPercentage: 0.1, 
      backgroundColor: data.map(row => row.open > row.close ? 'rgba(217, 4, 41, 0.8)' : 'rgba(0, 128, 0, 0.8)'), 
      order: 1 
    };
    const body = { 
      label: 'Body', 
      data: data.map(row => ({ x: row.date, y: [row.open, row.close] })), 
      type: 'bar', 
      barPercentage: 0.8, 
      backgroundColor: data.map(row => row.open > row.close ? 'rgba(217, 4, 41, 0.8)' : 'rgba(0, 128, 0, 0.8)'), 
      order: 2 
    };
    return { labels, datasets: [wicks, body] };
  };

  const charts = {
    candlestick: createCandleChart(hist),
    heikinAshi: createCandleChart(haHist),
    line: { labels, datasets: [{ label: 'Price', data: formatLine(hist), ...lineStyle('#a0c4e0', 2), fill: true, backgroundColor: 'rgba(160, 196, 224, 0.1)' }] },
    volume: { labels, datasets: [{ label: 'Volume', data: formatBar(hist, 'volume'), backgroundColor: hist.map(row => row.close < row.open ? 'rgba(217, 4, 41, 0.6)' : 'rgba(0, 128, 0, 0.6)') }] },
    sma: { labels, datasets: [{ label: 'SMA20', data: sma20, ...lineStyle('rgba(255, 159, 64, 0.8)') }, { label: 'SMA50', data: sma50, ...lineStyle('rgba(153, 102, 255, 0.8)') }] },
    macd: { labels, datasets: [{ label: 'MACD', data: macd.MACD, type: 'line', ...lineStyle('rgba(75, 192, 192, 0.8)') }, { label: 'Signal', data: macd.signal, type: 'line', ...lineStyle('rgba(255, 99, 132, 0.8)') }, { label: 'Histogram', data: macd.histogram, type: 'bar', backgroundColor: macd.histogram.map(v => v < 0 ? 'rgba(217, 4, 41, 0.6)' : 'rgba(0, 128, 0, 0.6)') }] },
    bollinger: { labels, datasets: [{ label: 'Upper', data: bollinger.upper, ...lineStyle('rgba(54, 162, 235, 0.5)'), fill: '+1', backgroundColor: 'rgba(54, 162, 235, 0.1)' }, { label: 'Middle', data: bollinger.middle, ...lineStyle('rgba(255, 206, 86, 0.8)') }, { label: 'Lower', data: bollinger.lower, ...lineStyle('rgba(54, 162, 235, 0.5)') }] },
    rsi: { labels, datasets: [{ label: 'RSI', data: rsiData, ...lineStyle('rgba(186, 85, 211, 0.8)') }] },
    dailyRange: { labels, datasets: [{ label: 'Daily Range (H-L)', data: dailyPriceRange, ...lineStyle('#fff', 1), fill: true, backgroundColor: 'rgba(255, 255, 255, 0.1)' }] },
    dailyChange: { labels, datasets: [{ label: '% Change', data: dailyChange, type: 'bar', backgroundColor: dailyChange.map(v => v.y < 0 ? 'rgba(217, 4, 41, 0.6)' : 'rgba(0, 128, 0, 0.6)') }] },
    cumulativeReturn: { labels, datasets: [{ label: 'Cumulative Return', data: cumulativeReturn, ...lineStyle('#33a02c', 2), fill: true, backgroundColor: 'rgba(51, 160, 44, 0.1)' }] },
    momentum: { labels, datasets: [{ label: 'Momentum (10D)', data: roc, ...lineStyle('#ff7f00', 2) }] },
    volatility: { labels, datasets: [{ label: 'Volatility (20D SD)', data: volatility, ...lineStyle('#fb9a99', 2), fill: true, backgroundColor: 'rgba(251, 154, 153, 0.1)' }] },
    stochastic: { labels, datasets: [{ label: 'Stochastic %K', data: stochastic.k, ...lineStyle('rgba(75, 192, 192, 0.8)') }, { label: 'Stochastic %D', data: stochastic.d, ...lineStyle('rgba(255, 99, 132, 0.8)') }] },
    vma: { labels, datasets: [{ label: 'VMA (20D)', data: vma, ...lineStyle('#cab2d6', 2) }] },
    obv: { labels, datasets: [{ label: 'On-Balance Volume (OBV)', data: obv, ...lineStyle('#fdbf6f', 2), fill: true, backgroundColor: 'rgba(253, 191, 111, 0.1)' }] },
  };

  return { info, financials, charts, quote, hist };
};


// --- [Tool 실행 함수: Tavily] ---
const searchTavilyApi = async (query) => {
  console.log(`[Tool Call] Tavily 검색: ${query}`);
  try {
    const response = await axios.post('https://api.tavily.com/search', {
      api_key: process.env.TAVILY_API_KEY,
      query: query,
      search_depth: "basic",
      include_answer: true,
      max_results: 5
    });
    if (response.data.answer) {
      return response.data.answer;
    }
    return JSON.stringify(response.data.results.map(r => ({ title: r.title, content: r.content, url: r.url })));
  } catch (error) {
    console.error("Tavily API 오류:", error.message);
    return "Tavily API 검색 중 오류가 발생했습니다.";
  }
};

// --- [Tool 실행 함수: SerpAPI (일반 웹 검색)] ---
const searchSerpApi = async (query) => {
  console.log(`[Tool Call] SerpAPI (Web) 검색: ${query}`);
  try {
    const response = await axios.get('https://serpapi.com/search', {
      params: {
        api_key: process.env.SERP_API_KEY,
        q: query,
        gl: 'kr',
        hl: 'ko',
      }
    });
    if (response.data.answer_box) {
      return response.data.answer_box.answer || response.data.answer_box.snippet;
    }
    if (response.data.sports_results) {
      return `스포츠 결과: ${response.data.sports_results.game_spotlight || JSON.stringify(response.data.sports_results)}`;
    }
    if (response.data.organic_results && response.data.organic_results.length > 0) {
      return JSON.stringify(response.data.organic_results.slice(0, 3).map(r => ({ title: r.title, snippet: r.snippet })));
    }
    return "특별한 검색 결과를 찾지 못했습니다.";
  } catch (error) {
    console.error("SerpAPI (Web) 오류:", error.message);
    return "SerpAPI (Web) 검색 중 오류가 발생했습니다.";
  }
};

// --- [Tool 실행 함수: News API (NewsAPI.org)] ---
const searchNewsApi = async (query) => {
  console.log(`[Tool Call] NewsAPI (NewsAPI.org) 검색: ${query}`);
  try {
    const response = await axios.get('https://newsapi.org/v2/everything', {
      params: {
        q: query,
        apiKey: process.env.NEWS_API_KEY,
        language: 'ko',
        sortBy: 'relevancy',
        pageSize: 5
      }
    });
    // 뉴스 기사 요약
    return JSON.stringify(response.data.articles.map(a => ({ 
      title: a.title, 
      source: a.source.name, 
      description: a.description 
    })));
  } catch (error) {
    console.error("NewsAPI (NewsAPI.org) 오류:", error.message);
    return "NewsAPI (NewsAPI.org) 검색 중 오류가 발생했습니다.";
  }
};

// --- [Tool 실행 함수: SerpAPI (Google News) (채팅 툴용)] ---
const searchSerpApiGoogleNews = async (query) => {
  console.log(`[Tool Call] SerpAPI (Google News) 검색: ${query}`);
  try {
    const response = await axios.get('https://serpapi.com/search', {
      params: {
        api_key: process.env.SERP_API_KEY,
        q: query,
        gl: 'kr',
        hl: 'ko',
        tbm: 'nws', // Google News 검색
        tbs: 'qdr:y' // 최근 1년
      }
    });

    if (response.data && response.data.news_results) {
      return JSON.stringify(response.data.news_results.slice(0, 5).map(r => ({
        title: r.title,
        source: r.source,
        summary: r.snippet, 
        url: r.link
      })));
    }
    return "SerpAPI (Google News)에서 관련 뉴스를 찾지 못했습니다.";
  } catch (error) {
    console.error("SerpAPI (Google News) 오류:", error.response ? error.response.data : error.message);
    return "SerpAPI (Google News) 검색 중 오류가 발생했습니다.";
  }
};


// --- [헬퍼 함수: SerpAPI (Google News) (Events 탭용)] ---
const getNewsForDateViaSerpApi = async (query, date) => {
  console.log(`[Tool Call] SerpAPI (Google News) ${date} 기준 ${query} 검색`);
  
  // Google 'tbs' 파라미터 형식(MM/DD/YYYY)으로 변환
  const [y, m, d] = date.split('-');
  const googleDate = `${m}/${d}/${y}`;
  
  try {
    const response = await axios.get('https://serpapi.com/search', {
      params: {
        api_key: process.env.SERP_API_KEY,
        q: query,
        gl: 'kr',    // 국가: 대한민국
        hl: 'ko',    // 언어: 한국어
        tbm: 'nws',  // 검색 엔진: Google News
        tbs: `cdr:1,cd_min:${googleDate},cd_max:${googleDate}` // 날짜 범위 지정 (정확히 그 날짜)
      }
    });
    
    if (response.data && response.data.news_results) {
      return response.data.news_results.map(d => ({
        title: d.title,
        source: d.source,       // 언론사
        description: d.snippet, // SerpAPI는 'snippet'을 제공
        url: d.link           // SerpAPI는 'link'를 제공
      }));
    }
    return [];
  } catch (error) {
    console.error(`SerpAPI (Google News) (${date}) 오류:`, error.response ? error.response.data : error.message);
    return []; // 오류 시 빈 배열 반환
  }
};


// --- [Tool 정의] ---
const tools = [
  {
    type: 'function',
    function: {
      name: 'searchTavilyApi',
      description: "AI 기반 요약 검색. 일반적인 웹 질문, 최신 토픽, 복잡한 질문에 대한 요약된 답변이 필요할 때 **가장 먼저** 사용하세요.",
      parameters: {
        type: 'object',
        properties: {
          query: { type: 'string', description: "AI 요약 검색 엔진으로 검색할 쿼리" },
        },
        required: ['query'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'searchSerpApiGoogleNews',
      description: "Tavily로 찾기 어려운 **최신 뉴스 기사(Google News 기반)**를 검색할 때 사용하세요. (예: '삼성전자 1분기 실적 뉴스', 'SK하이닉스 관련 기사')",
      parameters: {
        type: 'object',
        properties: {
          query: { type: 'string', description: "Google News에서 검색할 쿼리" },
        },
        required: ['query'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'searchNewsApi',
      description: "Google News(searchSerpApiGoogleNews)로 찾지 못했거나, **특정 해외 언론사(NewsAPI.org 기반)**의 기사가 필요할 때 사용하세요.",
      parameters: {
        type: 'object',
        properties: {
          query: { type: 'string', description: "NewsAPI.org에서 검색할 일반 뉴스 쿼리" },
        },
        required: ['query'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'searchSerpApi',
      description: "**실시간 주가** 또는 매우 간단한 **단순 사실 확인**(예: '테슬라 CEO', '미국 수도')이 필요할 때만 사용하세요. (이것은 일반 웹 검색입니다)",
      parameters: {
        type: 'object',
        properties: {
          query: { type: 'string', description: "Google (Web)에서 검색할 쿼리." },
        },
        required: ['query'],
      },
    },
  }
];


// --- [API 엔드포인트: /api/chat] ---
app.post('/api/chat', async (req, res) => {
  const { messages } = req.body; 

  if (!process.env.GROQ_API_KEY || !process.env.SERP_API_KEY || !process.env.TAVILY_API_KEY || !process.env.NEWS_API_KEY) {
    return res.status(500).json({ error: "필수 API 키가 서버에 설정되지 않았습니다." });
  }
  if (!messages || messages.length === 0) {
    return res.status(400).json({ error: "메시지 내용이 없습니다." });
  }

  const systemPrompt = {
    role: 'system',
    content: `당신은 'Stock-AI'입니다. 당신의 정체성은 단순한 금융 분석가나 투자 전문가를 넘어섭니다.
당신의 핵심 철학은 **"세상의 모든 사건, 뉴스, 기술, 역사, 심지어 문화 현상까지도 결국 금융 시장과 연결된다"**는 것입니다.

**[I. 당신의 핵심 임무 (Mandate)]**
당신의 임무는 사용자의 모든 질문에 대해, 그것이 무엇이든, **'사실'**을 제공하고 그 **'금융적 연결고리(The Financial Link)'**를 찾아 해석해주는 것입니다.

1.  **모든 질문 수용 (Accept All Queries):**
    사용자의 질문이 '코로나가 언제 시작되었어?', '5G 기술이 뭐야?', '오늘 날씨 어때?'처럼 주식과 무관해 보여도 절대 회피하지 않습니다.
2.  **신속한 사실 확인 (Fact-Finding):**
    모든 사실 기반 질문(날짜, 사건, 정의, 데이터, 인물 등)에 대해 **반드시** 제공된 도구를 사용해 객관적이고 정확한 정보를 찾습니다. 당신의 지식은 도구에서 나옵니다.
3.  **시장과 연결 (The Financial Link - 가장 중요):**
    사실 정보를 답변한 후, **반드시** 그 사건/정보가 주식 시장, 특정 산업, 원자재, 거시 경제, 또는 투자 심리에 **어떤 영향을 미쳤는지(과거), 미치고 있는지(현재), 또는 미칠 수 있는지(미래)**를 심층적으로 분석하고 연결하여 설명해야 합니다.

**[II. 임무 수행 시나리오 (Examples)]**

* **시나리오 1: 역사/사건 (과거)**
    * **사용자:** "코로나19는 정확히 언제 시작됐어?"
    * **Stock-AI:** (도구 검색...) "네, 코로나19 팬데믹은 2019년 12월 중국 우한에서 처음 보고되었으며, 2020년 3월 11일 WHO에 의해 팬데믹으로 선언되었습니다. **[금융적 연결]** 이 선언은 전 세계 금융 시장에 즉각적인 충격을 주어 2020년 2월과 3월에 VIX 지수가 폭등하고 주식 시장이 역사적인 속도로 폭락했습니다. 하지만 동시에 이는 '언택트(Untact)' 트렌드를 가속화시켜 줌(ZM), 아마존(AMZN) 같은 기술주 및 바이오(모더나, 화이자) 관련주가 시장을 주도하는 계기가 되었습니다."

* **시나리오 2: 기술/정의 (현재)**
    * **사용자:** "5G 기술이 정확히 뭐야?"
    * **Stock-AI:** (도구 검색...) "5G는 5세대 이동통신 기술로, 4G LTE 대비 초고속, 초저지연, 초연결성을 특징으로 합니다. **[금융적 연결]** 이는 단순히 스마트폰 속도 향상을 넘어, 자율주행, 원격 의료, 스마트 팩토리, 사물 인터넷(IoT) 등 4차 산업혁명의 핵심 인프라입니다. 따라서 5G 기술의 확산은 관련 반도체(예: 퀄컴, 브로드컴), 통신 장비(예: 삼성전자, 에릭슨), 그리고 기지국 인프라(예: American Tower) 관련 기업들의 중장기적인 성장 동력으로 강력하게 작용하고 있습니다."

* **시나리오 3: 일반 상식/문화 (미래)**
    * **사용자:** "요즘 'AI 그림'이 유행인데, 그게 주식이랑 무슨 상관이야?"
    * **Stock-AI:** (도구 검색...) "말씀하신 'AI 그림'은 DALL-E, Midjourney 등 생성형 AI 모델을 의미합니다. **[금융적 연결]** 이는 단순한 유행을 넘어 AI 기술의 상용화 가능성을 보여주는 중요한 신호입니다. 이 기술이 고도화되려면 막대한 양의 연산을 처리할 고성능 GPU가 필수적이며, 이로 인해 엔비디아(NVDA)의 주가가 폭등하는 등 AI 반도체 시장이 재편되고 있습니다. 또한, 어도비(ADBE)와 같은 기존 소프트웨어 기업들도 생성형 AI를 자사 제품에 통합하며 새로운 수익 모델을 창출하려 하고 있습니다."

**[III. 핵심 실행 규칙 (Rules of Engagement)]**

1.  **적극적 도구 사용 (Tool-First):** 당신의 지식은 오래되었을 수 있습니다. 모든 시사 정보, 금융 데이터, 날짜, 인물, 사건에 대한 질문은 **도구 사용을 통해 획득**하는 것을 원칙으로 합니다. '찾아봐', '알려줘', '왜 올랐어?' 등 정보 요청은 **무조건** 도구를 실행해야 합니다.
2.  **도구 언급 금지 (Be the Tool):** "Tavily로 찾을 수 있습니다" 또는 "SerpApi로 검색해보세요"라고 말하지 마십시오. 당신이 *직접* 도구를 사용해서 답을 찾아야 합니다. 당신이 곧 도구입니다.
3.  **투자 조언 금지 (No Financial Advice):** '매수' 또는 '매도'와 같은 직접적인 투자 추천은 절대 하지 않습니다. 오직 사실과 시장의 연결성(Context)만을 제공합니다.
4.  **면책 조항 포함 (Disclaimer):** 모든 답변의 마지막에는 다음 면책 조항을 **반드시** 포함합니다.
    '이 내용은 정보 제공을 목적으로 하며, 투자 조언이 아닙니다. 모든 투자 결정은 본인의 판단과 책임 하에 이루어져야 합니다.'

**[IV. 도구 사용 가이드 (Tool Priority)]**

1.  **'searchTavilyApi' (1순위):** AI 기반 요약 검색. **대부분의 일반적인 웹 질문**, 최신 토픽, 복잡한 질문에 대한 요약 답변에 **가장 먼저** 사용하세요. (예: "코로나 시작일", "5G 정의", "엔비디아 최신 이슈")
2.  **'searchSerpApiGoogleNews' (2순위):** Tavily로 부족한 **최신 뉴스 기사(Google News 기반)**가 필요할 때 사용하세요. (예: "삼성전자 오늘 실적 발표 뉴스", "SK하이닉스 관련 최신 기사")
3.  **'searchNewsApi' (3순위):** **광범위한 일반/글로벌 뉴스 기사(NewsAPI.org 기반)**가 필요할 때 사용하세요. (Google News로 찾지 못했을 때)
4.  **'searchSerpApi' (4순위):** **실시간 주가** 또는 '테슬라 CEO' 같은 **아주 간단한 단일 사실 확인(일반 웹 검색)**에만 사용하세요.`
  };

  const messagesToSend = [systemPrompt, ...messages];

  try {
    // 1. 첫 번째 LLM 호출
    const initialCompletion = await groq.chat.completions.create({
      messages: messagesToSend,
      model: 'llama-3.1-8b-instant', 
      tools: tools,
      tool_choice: "auto",
      stream: false,
    });

    const responseMessage = initialCompletion.choices[0].message;

    // 2. Tool 사용 시
    if (responseMessage.tool_calls) {
      console.log("[Tool Call] LLM이 도구 호출을 요청했습니다.");
      
      messagesToSend.push(responseMessage);
      
      const toolPromises = responseMessage.tool_calls.map(async (toolCall) => {
        const functionName = toolCall.function.name;
        const functionArgs = JSON.parse(toolCall.function.arguments);
        
        let functionResult;
        if (functionName === 'searchTavilyApi') {
          functionResult = await searchTavilyApi(functionArgs.query);
        } else if (functionName === 'searchSerpApi') {
          functionResult = await searchSerpApi(functionArgs.query);
        } else if (functionName === 'searchNewsApi') {
          functionResult = await searchNewsApi(functionArgs.query);
        } else if (functionName === 'searchSerpApiGoogleNews') { 
          functionResult = await searchSerpApiGoogleNews(functionArgs.query); 
        }

        return {
          tool_call_id: toolCall.id,
          role: 'tool',
          name: functionName,
          content: String(functionResult),
        };
      });
      
      const toolResults = await Promise.all(toolPromises);
      messagesToSend.push(...toolResults);

      // 3. 두 번째 LLM 호출 (최종 답변)
      const finalCompletion = await groq.chat.completions.create({
        messages: messagesToSend,
        model: 'llama-3.1-8b-instant',
        tools: tools,
        tool_choice: "auto",
      });
      
      res.json(finalCompletion.choices[0].message);

    } else {
      // Tool 미사용 시, 첫 번째 답변 바로 반환
      res.json(responseMessage);
    }

  } catch (error) {
    console.error("Groq API Error:", error.response ? error.response.data : error.message);
    if (error.response && error.response.data && error.response.data.error) {
       res.status(error.response.status || 500).json(error.response.data);
    } else {
       res.status(500).json({ error: "Groq API와 통신 중 알 수 없는 오류가 발생했습니다." });
    }
  }
});

// --- [추가] Contact 폼 저장 엔드포인트 ---
app.post('/api/contact', async (req, res) => {
  try {
    const { name, email, subject, message } = req.body;
    
    if (!name || !email || !subject || !message) {
      return res.status(400).json({ error: 'All fields are required.' });
    }

    const newContact = new Contact({ name, email, subject, message });
    await newContact.save();

    res.status(201).json({ message: 'Contact information saved successfully.' });
  } catch (error) {
    console.error('Contact Save Error:', error);
    res.status(500).json({ error: 'Failed to save contact information.' });
  }
});


// --- [API 엔드포인트: /api/stock/:ticker] ---
app.get('/api/stock/:ticker', async (req, res) => {
  const { ticker } = req.params;
  const { period = '1y', startDate } = req.query; 

  // [수정] 운영체제에 따라 python 명령어 분기 처리 (Linux/Docker 환경 대비)
  const pythonCommand = process.platform === 'win32' ? 'python' : 'python3';
  let command;
  
  if (startDate) {
    command = `${pythonCommand} stock.py ${ticker} --start ${startDate}`;
  } else {
    command = `${pythonCommand} stock.py ${ticker} --period ${period}`;
  }
  
  exec(command, { maxBuffer: 1024 * 1024 * 50 }, (error, stdout, stderr) => {
    if (error) {
      console.error(`[PYTHON EXEC ERROR] for ${ticker}: ${error.message}`);
      return res.status(500).json({ error: `Python script execution failed: ${stderr}` });
    }
    if (stderr) {
      console.warn(`[PYTHON STDERR] for ${ticker}: ${stderr}`);
    }

    try {
      const rawData = JSON.parse(stdout);
      if (rawData.error) {
        return res.status(404).json({ error: rawData.error });
      }
      
      const taPeriod = startDate ? 'all' : period;
      const processedData = processStockData(rawData, taPeriod);
      res.json(processedData);

    } catch (parseError) {
      console.error("[JSON PARSE ERROR]", parseError.message, parseError.stack);
      console.error("[RAW STDOUT]", stdout.substring(0, 1000)); 
      res.status(500).json({ error: "Failed to parse data from Python script." });
    }
  });
});

// --- [API 엔드포인트: /api/fred/:seriesId] ---
app.get('/api/fred/:seriesId', async (req, res) => {
  const { seriesId } = req.params;
  const { startDate } = req.query; 
  const apiKey = process.env.FRED_API_KEY;

  if (!apiKey) {
    return res.status(500).json({ error: "FRED_API_KEY is not set in .env file." });
  }

  const url = `https://api.stlouisfed.org/fred/series/observations`;
  
  try {
    const response = await axios.get(url, {
      params: {
        series_id: seriesId,
        api_key: apiKey,
        file_type: 'json',
        observation_start: startDate,
        sort_order: 'asc',
      }
    });

    const formattedData = response.data.observations.map(obs => ({
      x: obs.date,
      y: obs.value === '.' ? null : parseFloat(obs.value) 
    }));
    
    res.json(formattedData);

  } catch (error) {
    console.error("FRED API Error:", error.response?.data || error.message);
    res.status(500).json({ error: "Failed to fetch FRED data." });
  }
});


// --- [날짜 기반 뉴스 API 엔드포인트 (Events 탭용)] ---
app.post('/api/news-for-date', async (req, res) => {
  const { date, ticker, companyName } = req.body;

  if (!date || !ticker || !companyName) {
    return res.status(400).json({ error: "date, ticker, companyName이 필요합니다." });
  }

  const domesticQuery = `${companyName} OR ${ticker}`;

  try {
    const domesticNews = await getNewsForDateViaSerpApi(domesticQuery, date);

    res.json({
      domesticNews: domesticNews // domesticNews는 오류 시 빈 배열 반환
    });

  } catch (error) {
    console.error("뉴스 취합 오류:", error.message);
    res.status(500).json({ error: "뉴스 검색 중 서버 오류가 발생했습니다." });
  }
});


// --- [★★ 수정된 API 엔드포인트: /api/search-ticker ★★] ---
app.get('/api/search-ticker', async (req, res) => {
  const { query } = req.query;
  if (!query) {
    return res.status(400).json({ error: "Search query is required." });
  }

  const url = `https://query2.finance.yahoo.com/v6/finance/autocomplete?query=${encodeURIComponent(query)}&region=US&lang=en-US`;

  try {
    const response = await axios.get(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
      }
    });

    if (response.data && response.data.ResultSet && response.data.ResultSet.Result) {
      const results = response.data.ResultSet.Result
        .filter(item => item.type === 'S') // 'S'는 주식(Stock)을 의미
        .map(item => ({
          symbol: item.symbol,
          shortname: item.name,      // 'name'을 shortname으로 사용
          longname: item.name,       // 'longname' 대신 'name'을 사용
          exchange: item.exchDisp || item.exch, // 'exchDisp' (예: NASDAQ) 또는 'exch' (예: NMS)
        }));
      res.json(results);
    } else {
      res.json([]); // 검색 결과가 없는 경우 빈 배열 반환
    }
  } catch (error) {
    const status = error.response ? error.response.status : 'UNKNOWN';
    console.error(`Yahoo Finance Search API Error: Request failed with status code ${status}`, error.message);
    res.status(500).json({ error: `Failed to fetch ticker search results (Status: ${status}).` });
  }
});


// --- [추가] React 정적 파일 서빙 및 Catch-all 라우트 (맨 마지막에 위치) ---
// 프로덕션 환경(Fly.io 등)에서 React 빌드 파일을 제공하기 위함
app.use(express.static(path.join(__dirname, '../build')));

app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '../build', 'index.html'));
});


// --- [서버 시작] ---
app.listen(NODE_PORT, () => {
  console.log(`🚀 Server running on http://localhost:${NODE_PORT}`);
});