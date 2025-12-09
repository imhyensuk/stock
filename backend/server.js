// /Users/imhyeonseok/Documents/stock/web/backend/server.js

import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import { exec } from 'child_process';
import axios from 'axios';
import { createRequire } from 'module';
import Groq from 'groq-sdk';
import mongoose from 'mongoose';
import path from 'path';           
import { fileURLToPath } from 'url'; 

const require = createRequire(import.meta.url);
const technicalindicators = require('technicalindicators');

// ES Modules setup
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(cors());
app.use(express.json({ limit: '10mb' }));

const NODE_PORT = process.env.PORT || process.env.NODE_PORT || 8000;

// --- MongoDB Connection ---
mongoose.connect(process.env.MONGO_URI)
  .then(() => console.log('✅ MongoDB Connected'))
  .catch(err => console.error('❌ MongoDB Connection Error:', err));

const contactSchema = new mongoose.Schema({
  name: { type: String, required: true },
  email: { type: String, required: true },
  subject: { type: String, required: true },
  message: { type: String, required: true },
  createdAt: { type: Date, default: Date.now }
});
const Contact = mongoose.model('Contact', contactSchema);

// --- Groq Client Initialization ---
const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY
});


// ==========================================
// [SECTION 1] Data Processing Helpers (Preserved)
// ==========================================

const padArrayStart = (arr, targetLength) => {
  if (arr.length >= targetLength) return arr;
  return Array(targetLength - arr.length).fill(null).concat(arr);
};
const formatLine = (hist, columnName = 'close') => hist.map(row => ({ x: row.date, y: row[columnName] }));
const formatBar = (hist, columnName = 'volume') => hist.map(row => ({ x: row.date, y: row[columnName] }));

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
function calculateROC(data, period) {
  let results = [];
  if (data.length < period) return results;
  for (let i = period; i < data.length; i++) {
    const prev = data[i - period];
    results.push(prev === 0 ? null : ((data[i] - prev) / prev) * 100);
  }
  return results;
}
function calculateK(high, low, close, kPeriod) {
  let k = [];
  if (close.length < kPeriod) return k;
  for (let i = kPeriod - 1; i < close.length; i++) {
    const sliceH = high.slice(i - kPeriod + 1, i + 1);
    const sliceL = low.slice(i - kPeriod + 1, i + 1);
    const highestHigh = Math.max(...sliceH);
    const lowestLow = Math.min(...sliceL);
    k.push(((close[i] - lowestLow) / (highestHigh - lowestLow)) * 100);
  }
  return k;
}
function calculateOBV(close, volume) {
  let obv = [0];
  for (let i = 1; i < close.length; i++) {
    if (close[i] > close[i-1]) obv.push(obv[i-1] + volume[i]);
    else if (close[i] < close[i-1]) obv.push(obv[i-1] - volume[i]);
    else obv.push(obv[i-1]);
  }
  return obv;
}
function calculateVWMA(close, volume, period) {
  let results = [];
  if (close.length < period) return results;
  for (let i = period - 1; i < close.length; i++) {
    const sliceC = close.slice(i - period + 1, i + 1);
    const sliceV = volume.slice(i - period + 1, i + 1);
    let sumPriceVol = 0, sumVol = 0;
    for (let j = 0; j < period; j++) {
      sumPriceVol += sliceC[j] * sliceV[j];
      sumVol += sliceV[j];
    }
    results.push(sumVol === 0 ? null : sumPriceVol / sumVol);
  }
  return results;
}

const processStockData = (rawData, period) => {
  const { info, financials, hist: fullHist, quote } = rawData;
  const hist = fullHist.filter(row => row.close != null);
  if (!hist || hist.length === 0) throw new Error('Historical data not found.');

  const closePrices = hist.map(r => r.close);
  const highPrices = hist.map(r => r.high);
  const lowPrices = hist.map(r => r.low);
  const openPrices = hist.map(r => r.open);
  const volume = hist.map(r => r.volume);
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
  
  const dailyPriceRange = hist.map(row => ({ x: row.date, y: row.high - row.low }));
  const dailyChange = hist.map((row, i) => i === 0 ? { x: row.date, y: 0 } : { x: row.date, y: ((row.close - hist[i-1].close) / hist[i-1].close) * 100 });
  const cumulativeReturn = hist.map(row => ({ x: row.date, y: ((row.close - hist[0].close) / hist[0].close) * 100 }));
  const roc = padArrayStart(calculateROC(closePrices, 10), totalLength);
  const volatility = padArrayStart(calculateSD(closePrices, 20), totalLength);
  
  const kData = calculateK(highPrices, lowPrices, closePrices, 14);
  const dData = technicalindicators.SMA.calculate({ period: 3, values: kData });
  const stochastic = { k: padArrayStart(kData, totalLength), d: padArrayStart(padArrayStart(dData, kData.length), totalLength) };
  
  const vma = padArrayStart(calculateVWMA(closePrices, volume, 20), totalLength);
  const obv = calculateOBV(closePrices, volume);

  const haData = technicalindicators.HeikinAshi.calculate({ open: openPrices, high: highPrices, low: lowPrices, close: closePrices });
  const haHist = hist.map((row, i) => ({
    date: row.date, open: haData.open[i], high: haData.high[i], low: haData.low[i], close: haData.close[i]
  }));

  const labels = hist.map(r => r.date);
  const lineStyle = (color, width = 1.5) => ({ borderColor: color, borderWidth: width, fill: false, pointRadius: 0 });
  
  const createCandleChart = (data) => ({
    labels,
    datasets: [
      { label: 'Wicks', data: data.map(r => ({ x: r.date, y: [r.low, r.high] })), type: 'bar', barPercentage: 0.1, backgroundColor: data.map(r => r.open > r.close ? 'rgba(217, 4, 41, 0.8)' : 'rgba(0, 128, 0, 0.8)'), order: 1 },
      { label: 'Body', data: data.map(r => ({ x: r.date, y: [r.open, r.close] })), type: 'bar', barPercentage: 0.8, backgroundColor: data.map(r => r.open > r.close ? 'rgba(217, 4, 41, 0.8)' : 'rgba(0, 128, 0, 0.8)'), order: 2 }
    ]
  });

  const charts = {
    candlestick: createCandleChart(hist),
    heikinAshi: createCandleChart(haHist),
    line: { labels, datasets: [{ label: 'Price', data: formatLine(hist), ...lineStyle('#a0c4e0', 2), fill: true, backgroundColor: 'rgba(160, 196, 224, 0.1)' }] },
    volume: { labels, datasets: [{ label: 'Volume', data: formatBar(hist, 'volume'), backgroundColor: hist.map(r => r.close < r.open ? 'rgba(217, 4, 41, 0.6)' : 'rgba(0, 128, 0, 0.6)') }] },
    sma: { labels, datasets: [{ label: 'SMA20', data: sma20, ...lineStyle('#ff9f40') }, { label: 'SMA50', data: sma50, ...lineStyle('#9966ff') }] },
    macd: { labels, datasets: [{ label: 'MACD', data: macd.MACD, ...lineStyle('#4bc0c0') }, { label: 'Signal', data: macd.signal, ...lineStyle('#ff6384') }, { label: 'Hist', data: macd.histogram, type: 'bar', backgroundColor: '#4bc0c0' }] },
    bollinger: { labels, datasets: [{ label: 'Upper', data: bollinger.upper, ...lineStyle('#36a2eb') }, { label: 'Middle', data: bollinger.middle, ...lineStyle('#ffcd56') }, { label: 'Lower', data: bollinger.lower, ...lineStyle('#36a2eb') }] },
    rsi: { labels, datasets: [{ label: 'RSI', data: rsiData, ...lineStyle('#ba55d3') }] },
    dailyRange: { labels, datasets: [{ label: 'Range', data: dailyPriceRange, ...lineStyle('#fff') }] },
    dailyChange: { labels, datasets: [{ label: '% Change', data: dailyChange, type: 'bar', backgroundColor: '#ff6384' }] },
    cumulativeReturn: { labels, datasets: [{ label: 'Return', data: cumulativeReturn, ...lineStyle('#33a02c') }] },
    momentum: { labels, datasets: [{ label: 'Momentum', data: roc, ...lineStyle('#ff7f00') }] },
    volatility: { labels, datasets: [{ label: 'Volatility', data: volatility, ...lineStyle('#fb9a99') }] },
    stochastic: { labels, datasets: [{ label: '%K', data: stochastic.k, ...lineStyle('#4bc0c0') }, { label: '%D', data: stochastic.d, ...lineStyle('#ff6384') }] },
    vma: { labels, datasets: [{ label: 'VMA', data: vma, ...lineStyle('#cab2d6') }] },
    obv: { labels, datasets: [{ label: 'OBV', data: obv, ...lineStyle('#fdbf6f') }] },
  };

  return { info, financials, charts, quote, hist };
};


// ==========================================
// [SECTION 2] AI Agent Tools (Tavily, Serp, NewsAPI, DeepSearch)
// ==========================================

const searchTavilyApi = async (query) => {
  console.log(`[Tool] Tavily Search: ${query}`);
  try {
    const response = await axios.post('https://api.tavily.com/search', {
      api_key: process.env.TAVILY_API_KEY,
      query: query,
      search_depth: "advanced",
      include_answer: true,
      max_results: 5
    });
    if (response.data.answer) return response.data.answer;
    return JSON.stringify(response.data.results.map(r => ({ title: r.title, content: r.content })));
  } catch (error) { return "Tavily Search Error."; }
};

const searchSerpApiGoogleNews = async (query) => {
  console.log(`[Tool] Google News Search: ${query}`);
  try {
    const response = await axios.get('https://serpapi.com/search', {
      params: {
        api_key: process.env.SERP_API_KEY,
        q: query,
        gl: 'kr',
        hl: 'ko',
        tbm: 'nws',
        tbs: 'qdr:w' 
      }
    });
    if (response.data.news_results) {
      return JSON.stringify(response.data.news_results.slice(0, 5).map(r => ({
        title: r.title,
        source: r.source,
        summary: r.snippet,
        url: r.link
      })));
    }
    return "No Google News results found.";
  } catch (error) { return "Google News Search Error."; }
};

const searchNewsApi = async (query) => {
  console.log(`[Tool] NewsAPI Search: ${query}`);
  try {
    const response = await axios.get('https://newsapi.org/v2/everything', {
      params: {
        q: query,
        apiKey: process.env.NEWS_API_KEY,
        sortBy: 'publishedAt',
        pageSize: 5
      }
    });
    return JSON.stringify(response.data.articles.map(a => ({
      title: a.title,
      source: a.source.name,
      description: a.description
    })));
  } catch (error) { return "NewsAPI Search Error."; }
};

const searchDeepSearchApi = async (query) => {
  console.log(`[Tool] DeepSearch: ${query}`);
  try {
    if (!process.env.DEEPSEARCH_API_KEY) return "DeepSearch API Key missing.";
    
    const response = await axios.get('https://api.deepsearch.com/v1/articles', {
      params: { keyword: query, limit: 5 },
      headers: { 'Authorization': `Bearer ${process.env.DEEPSEARCH_API_KEY}` }
    });
    
    if (response.data && response.data.data) {
      return JSON.stringify(response.data.data.map(d => ({
        title: d.title,
        summary: d.summary || d.content_snippet
      })));
    }
    return "No DeepSearch results found.";
  } catch (error) {
    console.warn("DeepSearch API call failed:", error.message);
    return "DeepSearch API currently unavailable.";
  }
};

const tools = [
  {
    type: 'function',
    function: {
      name: 'searchTavilyApi',
      description: "Use this for high-level summaries. Do NOT translate the parameter key 'query'.",
      parameters: {
        type: 'object',
        properties: { query: { type: 'string', description: "Search query string" } },
        required: ['query'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'searchSerpApiGoogleNews',
      description: "Find specific breaking news. Do NOT translate the parameter key 'query'.",
      parameters: {
        type: 'object',
        properties: { query: { type: 'string', description: "News search query string" } },
        required: ['query'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'searchNewsApi',
      description: "Search global news. Do NOT translate the parameter key 'query'.",
      parameters: {
        type: 'object',
        properties: { query: { type: 'string', description: "News search query string" } },
        required: ['query'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'searchDeepSearchApi',
      description: "Search deep financial insights. Do NOT translate the parameter key 'query'.",
      parameters: {
        type: 'object',
        properties: { query: { type: 'string', description: "Keyword string" } },
        required: ['query'],
      },
    },
  }
];


// ==========================================
// [SECTION 3] API Endpoints
// ==========================================

// 1. AI Chat Endpoint with Language Support
app.post('/api/chat', async (req, res) => {
  const { messages, language } = req.body; 

  if (!messages || messages.length === 0) return res.status(400).json({ error: "No messages provided." });

  // [FIX] 언어 지침을 명확히 하고, Tool 호출과 답변 생성을 엄격히 분리
  const langInstruction = language === 'ko' 
    ? "최종 답변은 반드시 **한국어(Korean)**로 작성하세요." 
    : "All final responses must be in **English**.";

  const systemPrompt = {
    role: 'system',
    content: `You are 'Quantum Insight', a Senior Quantitative Strategist & AI Analyst.
    
    [CRITICAL INSTRUCTION - TOOL USE]
    1. When you need information, you MUST call a tool (function).
    2. **DO NOT** generate any conversational text or "thoughts" when calling a tool. Output **ONLY** the JSON for the tool call.
    3. The tool parameter keys MUST remain in English (e.g., 'query'), even if the value is Korean.
    
    [MANDATORY LANGUAGE INSTRUCTION]
    ${langInstruction}
    
    [YOUR MISSION]
    You are provided with hard data from proprietary AI models (LightGBM, PyTorch VAE, GARCH).
    Your job is to explain the 'WHY' by connecting data to real-world events.
    
    [DATA INTERPRETATION RULES]
    1. **Direction Score (>0.5 Bullish, <0.5 Bearish):**
       - Bullish? Find earnings beats, product launches, sector upgrades.
       - Bearish? Find regulatory issues, poor guidance, macro headwinds.
    2. **Predicted Volatility (Risk):**
       - High? Search for "Fed meetings", "Earnings dates", "Geopolitical tension".
    3. **Regime Anomaly Score (Market Stability):**
       - High (>15)? This signals structural break. Search for "Crash risks", "Black swan", "Liquidity crisis".
    
    [RESPONSE STYLE]
    - Structure: "Model Verdict" -> "Real-world Evidence" -> "Strategic Implication".
    - **Always** cite the news source or event found via tools.
    - Disclaimer: "This is AI analysis, not financial advice."
    `
  };

  const messagesToSend = [systemPrompt, ...messages];

  try {
    const completion = await groq.chat.completions.create({
      messages: messagesToSend,
      model: 'llama-3.1-8b-instant', 
      tools: tools,
      tool_choice: "auto",
      temperature: 0.1, // [FIX] 온도를 낮춰서 도구 호출의 정확도 향상
    });

    const responseMsg = completion.choices[0].message;

    // 도구 호출이 발생했는지 확인
    if (responseMsg.tool_calls && responseMsg.tool_calls.length > 0) {
      messagesToSend.push(responseMsg); 

      const toolPromises = responseMsg.tool_calls.map(async (toolCall) => {
        const fnName = toolCall.function.name;
        // [FIX] JSON 파싱 안전장치
        let fnArgs = {};
        try {
            fnArgs = JSON.parse(toolCall.function.arguments);
        } catch (e) {
            return { tool_call_id: toolCall.id, role: 'tool', name: fnName, content: "Error: Invalid JSON parameters." };
        }

        let result = "Info not found.";
        
        // [FIX] 파라미터 키가 번역되었을 경우를 대비한 유연한 처리
        const query = fnArgs.query || fnArgs['뉴스 쿼리'] || fnArgs['검색어'] || Object.values(fnArgs)[0] || "";

        if (!query) return { tool_call_id: toolCall.id, role: 'tool', name: fnName, content: "Error: Missing query parameter." };

        if (fnName === 'searchTavilyApi') result = await searchTavilyApi(query);
        else if (fnName === 'searchSerpApiGoogleNews') result = await searchSerpApiGoogleNews(query);
        else if (fnName === 'searchNewsApi') result = await searchNewsApi(query);
        else if (fnName === 'searchDeepSearchApi') result = await searchDeepSearchApi(query);

        return {
          tool_call_id: toolCall.id,
          role: 'tool',
          name: fnName,
          content: String(result)
        };
      });

      const toolResults = await Promise.all(toolPromises);
      messagesToSend.push(...toolResults); 

      // 2차 호출 (도구 결과 바탕으로 최종 답변 생성)
      const finalCompletion = await groq.chat.completions.create({
        messages: messagesToSend,
        model: 'llama-3.1-8b-instant',
        tools: tools,
        tool_choice: "auto",
        temperature: 0.3, // 답변 생성 시에는 약간의 창의성 허용
      });

      return res.json(finalCompletion.choices[0].message);
    }

    res.json(responseMsg);

  } catch (error) {
    console.error("AI Chat Error:", error.response ? error.response.data : error.message);
    // [FIX] 에러 발생 시 사용자에게 친절한 메시지 반환
    const errorMessage = language === 'ko' 
      ? "죄송합니다. 분석 중 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
      : "Sorry, a temporary error occurred during analysis. Please try again later.";
      
    res.json({ role: "assistant", content: errorMessage });
  }
});

// 2. AI Model Prediction
app.get('/api/predict/:ticker', async (req, res) => {
  // Model prediction service disabled - torch dependency removed to reduce Docker image size
  res.status(503).json({ 
    error: "Model prediction service is temporarily unavailable",
    message: "이 기능은 현재 개선 중입니다."
  });
});

// 3. Stock Data Endpoint
app.get('/api/stock/:ticker', async (req, res) => {
  const { ticker } = req.params;
  const { period = '1y', startDate } = req.query;
  const pythonCommand = process.platform === 'win32' ? 'python' : 'python3';
  
  let command;
  if (startDate) command = `${pythonCommand} stock.py ${ticker} --start ${startDate}`;
  else command = `${pythonCommand} stock.py ${ticker} --period ${period}`;

  exec(command, { maxBuffer: 1024 * 1024 * 50 }, (error, stdout, stderr) => {
    if (error) return res.status(500).json({ error: stderr || error.message });
    try {
      const rawData = JSON.parse(stdout);
      if (rawData.error) return res.status(404).json(rawData);
      const processed = processStockData(rawData, startDate ? 'all' : period);
      res.json(processed);
    } catch (e) {
      res.status(500).json({ error: "Data parsing failed." });
    }
  });
});

// 4. Other Utils
app.get('/api/fred/:seriesId', async (req, res) => {
  try {
    const { seriesId } = req.params;
    const { startDate } = req.query;
    const response = await axios.get(`https://api.stlouisfed.org/fred/series/observations`, {
      params: { series_id: seriesId, api_key: process.env.FRED_API_KEY, file_type: 'json', observation_start: startDate, sort_order: 'asc' }
    });
    res.json(response.data.observations.map(obs => ({ x: obs.date, y: parseFloat(obs.value) })));
  } catch (e) { res.status(500).json({ error: "FRED API Error" }); }
});

app.post('/api/contact', async (req, res) => {
  try {
    const { name, email, subject, message } = req.body;
    await new Contact({ name, email, subject, message }).save();
    res.status(201).json({ message: 'Saved' });
  } catch (e) { res.status(500).json({ error: 'Save failed' }); }
});

app.get('/api/search-ticker', async (req, res) => {
  try {
    const { query } = req.query;
    const url = `https://query2.finance.yahoo.com/v6/finance/autocomplete?query=${encodeURIComponent(query)}&region=US&lang=en-US`;
    const response = await axios.get(url, { headers: { 'User-Agent': 'Mozilla/5.0' } });
    const results = response.data.ResultSet.Result.filter(i => i.type === 'S').map(i => ({ symbol: i.symbol, shortname: i.name, exchange: i.exchDisp }));
    res.json(results);
  } catch (e) { res.status(500).json({ error: "Search failed" }); }
});

// 5. News for Date (Helper for Events Tab)
const getNewsForDateViaSerpApi = async (query, date) => {
  try {
    const [y, m, d] = date.split('-');
    const googleDate = `${m}/${d}/${y}`;
    const response = await axios.get('https://serpapi.com/search', {
      params: { api_key: process.env.SERP_API_KEY, q: query, gl: 'kr', hl: 'ko', tbm: 'nws', tbs: `cdr:1,cd_min:${googleDate},cd_max:${googleDate}` }
    });
    if (response.data && response.data.news_results) {
      return response.data.news_results.map(d => ({ title: d.title, source: d.source, description: d.snippet, url: d.link }));
    }
    return [];
  } catch (e) { return []; }
};

app.post('/api/news-for-date', async (req, res) => {
  const { date, ticker, companyName } = req.body;
  try {
    const news = await getNewsForDateViaSerpApi(`${companyName} OR ${ticker}`, date);
    res.json({ domesticNews: news });
  } catch (e) { res.status(500).json({ error: "News fetch error" }); }
});

// React Static Files
const buildPath = path.join(__dirname, '../build');
try {
  app.use(express.static(buildPath));
  app.get('*', (req, res) => res.sendFile(path.join(buildPath, 'index.html')));
} catch (e) { console.warn('Running in API-only mode'); }

app.listen(NODE_PORT, () => {
  console.log(`🚀 Server running on http://localhost:${NODE_PORT}`);
});