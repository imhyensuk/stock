// /Users/imhyeonseok/Documents/stock/web/src/components/stock/stock.js

import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import Nav from '../nav/nav';
import './stock.css';
// [★★ 수정: BiSearchAlt 아이콘 추가 ★★]
import { BiSearch, BiX, BiShow, BiDownload, BiQuestionMark, BiNews, BiSearchAlt } from 'react-icons/bi'; 
// import { FaRobot } from 'react-icons/fa'; // [주석 처리] 채팅 아이콘
// import ChatModal from '../chat/ChatModal'; // [주석 처리] 채팅 모달 컴포넌트
import axios from 'axios';
import { Chart as ChartComponent, Bar, Line } from 'react-chartjs-2';
import { Chart as ChartJS, registerables } from 'chart.js';
import 'chartjs-adapter-date-fns';

ChartJS.register(...registerables);

// --- API 설정 ---
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// --- 상수 정의 ---
const periodOptions = [
  { label: '1M', value: '1mo' }, { label: '3M', value: '3mo' },
  { label: '6M', value: '6mo' }, { label: '1Y', value: '1y' },
  { label: '3Y', value: '3y' }, { label: '5Y', value: '5y' }, { label: '10Y', value: '10y' },
  { label: 'All', value: 'all' }
];
const TABS = ['Dashboard', 'Financials', 'Historical Data', 'Events', 'Federal Info'];

const FEDERAL_SERIES = {
  'GDPC1': 'Real GDP (GDPC1)', 
  'PCE': 'Personal Consumption Expenditures (PCE)', 
  'CPIAUCSL': 'Consumer Price Index (CPIAUCSL)', 
  'INDPRO': 'Industrial Production (INDPRO)', 
  'UNRATE': 'Unemployment Rate (UNRATE)', 
  'FEDFUNDS': 'Federal Funds Rate (FEDFUNDS)', 
  'DGS10': '10-Year Treasury Yield (DGS10)', 
  'T10Y2Y': '10-Yr vs 2-Yr Yield Curve (T10Y2Y)', 
  'M2SL': 'M2 Money Supply (M2SL)', 
  'HOUST': 'Housing Starts (HOUST)', 
  'UMCSENT': 'Consumer Sentiment (UMCSENT)', 
};


// [★★ 수정: 차트 툴팁 전체 보강 ★★]
const chartTooltips = {
  "Performance Comparison (%)": "선택한 기간의 시작일을 0% 기준으로, 두 기업의 주가 수익률을 비교합니다. A 주식이 10%, B 주식이 5%라면 A 주식의 상승률이 더 높았음을 의미합니다.",
  "Price vs 10-Yr Treasury": "주가(좌측 축)와 미국 10년 만기 국채 금리(우측 축)를 비교합니다. 일반적으로 **국채 금리가 오르면(안전 자산 선호 또는 인플레이션 우려) 기술주 등 성장주는 하락**하는 경향이 있습니다.",
  "Price (Candlestick)": "시가(Open), 고가(High), 저가(Low), 종가(Close)를 캔들(봉) 모양으로 나타냅니다. **녹색(양봉)은 종가가 시가보다 높게(상승 마감), 적색(음봉)은 낮게(하락 마감)** 끝났음을 의미합니다.",
  "Heikin Ashi": "OHLC(시가/고가/저가/종가)를 평균화하여 추세를 더 부드럽게 보여줍니다. **연속된 녹색 캔들은 강한 상승 추세**를, **연속된 적색 캔들은 강한 하락 추세**를 나타내어 노이즈를 줄이고 추세 파악에 용이합니다.",
  "Price (Line)": "선택된 기간의 종가(Close) 기준 주가 추이를 라인 차트로 비교합니다. 주가의 전반적인 방향성을 한눈에 파악할 수 있습니다.",
  "Volume": "선택한 기간 동안의 일별 거래량을 막대 차트로 나타냅니다. **주가가 상승하며 거래량이 터지면 강한 매수세**를, 하락하며 터지면 강한 매도세를 의미합니다.",
  "Volume Comparison": "두 기업의 일별 거래량을 라인 차트로 비교합니다. 시장의 관심도가 어느 기업에 더 집중되어 있는지 상대적으로 파악할 수 있습니다.",
  "Moving Averages (SMA)": "주가 이동평균선(Simple Moving Average)입니다. **단기선(20일)이 장기선(50일)을 상향 돌파(골든 크로스)하면 강세 신호**로, 하향 돌파(데드 크로스)하면 약세 신호로 해석합니다.",
  "MACD": "MACD(12일-26일) 선과 시그널(9일) 선의 교차를 통해 매매 시점을 파악합니다. **MACD 선이 시그널 선을 상향 돌파하면 매수 신호**, 하향 돌파하면 매도 신호로 봅니다. 막대(히스토그램)가 0선 위에 있으면 상승 에너지가 강함을 의미합니다.",
  "Bollinger Bands": "주가가 20일 이동평균선을 기준으로 어느 위치에 있는지(표준편차 2배)를 보여주는 밴드입니다. **주가가 상단 밴드에 닿으면 과매수(비쌈)**, **하단 밴드에 닿으면 과매도(쌈)** 상태로 봅니다. 밴드의 폭이 좁아지면(수축) 곧 큰 변동성이 올 수 있음을 예고합니다.",
  "RSI": "상대강도지수(RSI)입니다. 70 이상은 과매수, 30 이하는 과매도 구간으로 해석합니다. **과매수(70 이상) 구간에서는 매도**를, **과매도(30 이하) 구간에서는 매수**를 고려하는 역추세 전략에 사용됩니다.",
  "Daily Price Range (H-L)": "일일 고가(High)와 저가(Low)의 차이. **이 범위가 넓어지면(그래프 상승) 주가의 일일 변동성이 커졌음**을 의미합니다.",
  "% Change (Day-over-Day)": "전일 대비 종가 기준 가격 변화율(%)을 나타냅니다. **막대가 0선 위에 있으면 상승, 아래에 있으면 하락**한 것입니다. 변동폭이 얼마나 컸는지 시각적으로 보여줍니다.",
  "Cumulative Return": "선택한 기간의 시작일 대비 누적 수익률(%)을 나타냅니다. **그래프가 우상향하면 투자 성과가 좋음**을, 우하향하면 나쁨을 의미합니다.",
  "Momentum (10D)": "10일 전 주가 대비 현재 주가의 상승/하락 모멘텀(ROC)을 나타냅니다. **그래프가 0선 위에 있으면 10일 전보다 상승**했음을, 아래에 있으면 하락했음을 의미하며, 그 강도를 보여줍니다.",
  "Volatility (20D SD)": "주가의 변동성을 20일 표준편차로 측정한 지표입니다. **그래프가 높을수록 최근 20일간 주가 변동폭이 컸음(위험도 증가)**을 의미합니다.",
  "Stochastic Oscillator": "%K(빠른 선)와 %D(느린 선)를 통해 현재 주가가 일정 기간의 고가/저가 범위 중 어디에 있는지 보여줍니다. **80 이상이면 과매수, 20 이하면 과매도** 상태로 해석합니다. %K선이 %D선을 상향 돌파 시 매수 신호로 보기도 합니다.",
  "VMA (20D)": "가변 이동평균(Variable Moving Average)으로, 시장 변동성에 따라 가중치를 조절합니다. **변동성이 클 때는 둔감하게, 작을 때는 민감하게** 반응하여 추세를 더 잘 따르도록 설계되었습니다.",
  "On-Balance Volume (OBV)": "거래량을 기반으로 주가의 상승/하락 압력을 측정합니다. **주가는 횡보하는데 OBV가 상승하면 매집 신호**로, 주가는 상승하는데 OBV가 하락하면 매도 신호로 해석할 수 있습니다.",
  "Key Financials": "재무제표의 핵심 지표(매출, 순이익 등)를 연도별로 비교합니다. **매출과 순이익이 꾸준히 성장하는지**가 기업 분석의 핵심입니다.",
  "Quarterly Revenue & Net Income": "분기별 매출(Revenue)과 순이익(Net Income) 추이를 보여줍니다. **최근 분기 실적이 시장 예상치(컨센서스)와 비교하여 어떠했는지(어닝 서프라이즈/쇼크)**가 단기 주가에 큰 영향을 줍니다.",
  "Valuation Metrics": "기업가치(EV/EBITDA)와 이자보상배율(Interest Coverage)을 보여줍니다. **EV/EBITDA는 낮을수록 저평가**, **이자보상배율은 높을수록(예: 3배 이상) 재무 건전성이 좋음**을 의미합니다.",
  "Historical Price (Line)": "선택된 기간의 종가(Close) 기준 주가 추이를 라인 차트로 보여줍니다. 과거 주가 흐름을 한눈에 파악할 수 있습니다.",
  
  // Federal Info 탭 툴팁 보강
  "Consumer Price Index (CPIAUCSL)": "소비자 물가 지수(CPI, 좌측 축)와 주가(우측 축)를 비교합니다. **CPI가 급격히 상승하면(인플레이션), 연준(Fed)이 금리를 인상할 수 있어 주식 시장에 악재**로 작용하는 경우가 많습니다.",
  "Unemployment Rate (UNRATE)": "실업률(좌측 축)과 주가(우측 축)를 비교합니다. **실업률이 낮으면(경제 호황) 주가에 긍정적**이지만, 너무 낮으면 인플레이션 우려로 금리 인상 요인이 될 수 있습니다. **실업률이 급등하면 경기 침체 신호**로 주가에 큰 악재입니다.",
  "Federal Funds Rate (FEDFUNDS)": "미국 연방 준비 제도의 기준 금리(좌측 축)와 주가(우측 축)를 비교합니다. **금리 인상은 일반적으로 기업의 대출 비용을 증가시키고 시장 유동성을 축소시켜 주가에 부정적**인 영향을 줍니다. (반대는 긍정적)",
  "M2 Money Supply (M2SL)": "시중에 유통되는 통화량(M2, 좌측 축)과 주가(우측 축)를 비교합니다. **통화량이 증가(유동성 공급)하면 돈의 가치가 하락하고 자산(주식) 가격이 상승**하는 경향이 있습니다.",
  "Real GDP (GDPC1)": "실질 국내총생산(GDP, 좌측 축)과 주가(우측 축)를 비교합니다. **GDP가 꾸준히 성장한다는 것은 국가 경제가 건강하다는 신호**이며, 이는 기업 이익 증가로 이어져 주식 시장에 장기적으로 긍정적입니다.",
  "Personal Consumption Expenditures (PCE)": "개인 소비 지출(PCE) 물가 지수(좌측 축)와 주가(우측 축)를 비교합니다. 연준(Fed)이 CPI보다 선호하는 인플레이션 지표입니다. **PCE 상승률이 목표치(보통 2%)를 크게 상회하면 금리 인상 압력**이 커져 주가에 부담이 됩니다.",
  "Industrial Production (INDPRO)": "산업 생산 지수(좌측 축)와 주가(우측 축)를 비교합니다. **산업 생산이 활발(지수 상승)하다는 것은 기업들의 제조업 활동이 왕성함**을 의미하며, 이는 경제와 주가에 긍정적인 신호입니다.",
  "10-Year Treasury Yield (DGS10)": "미국 10년 만기 국채 수익률(좌측 축)과 주가(우측 축)를 비교합니다. **시장의 장기 금리 기대를 반영하며, 이 금리가 상승하면 특히 기술주의 밸류에이션(가치)에 부담**을 주어 주가를 하락시키는 경향이 있습니다.",
  "10-Yr vs 2-Yr Yield Curve (T10Y2Y)": "10년물-2년물 국채 금리 차이(장단기 금리차, 좌측 축)와 주가(우측 축)를 비교합니다. **이 값이 0 미만(역전)이 되면 단기 금리가 장기 금리보다 높아지는 현상으로, 강력한 경기 침체 신호**로 해석되며 보통 1~2년 뒤 주가 하락을 예고합니다.",
  "Housing Starts (HOUST)": "신규 주택 착공 건수(좌측 축)와 주가(우측 축)를 비교합니다. **주택 착공이 증가하면 건설 경기가 활발하고 소비가 촉진된다는 의미**로, 전반적인 경제 건전성과 주가에 긍정적인 영향을 줍니다.",
  "Consumer Sentiment (UMCSENT)": "미시간 대학 소비자 심리 지수(좌측 축)와 주가(우측 축)를 비교합니다. **소비자 심리가 낙관적(지수 상승)이면 향후 소비 지출이 늘어날 가능성**이 크며, 이는 기업 실적과 주가에 긍정적인 선행 지표가 됩니다.",
};

const COLORS = ['#a0c4e0', '#ff6384', '#ff9f40', '#4bc0c0', '#9966ff', '#ffcd56'];
const COLOR_1 = COLORS[0]; 
const COLOR_2 = COLORS[1]; 


const LoadingSpinner = () => <div className="stock-loading-spinner"></div>;
const ErrorDisplay = ({ message }) => <div className="stock-error-display">{message}</div>;

// --- 차트 옵션 ---
const getChartOptions = (title, isComparison = false) => ({
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { 
      display: !title.includes('Candlestick') && !title.includes('Heikin Ashi'), 
      labels: { color: '#f5f5f5' },
    },
    title: { display: false },
    tooltip: { mode: 'index', intersect: false },
  },
  scales: {
    x: { type: 'time', time: { unit: 'month', tooltipFormat: 'yyyy-MM-dd' }, ticks: { color: '#888' }, grid: { color: '#333' } },
    y: {
      ticks: { color: '#888', callback: isComparison ? (value) => `${value}%` : (value) => value.toLocaleString() },
      grid: { color: '#333' },
      stacked: false, 
    },
  },
  interaction: { mode: 'nearest', axis: 'x', intersect: false }
});

const lineStyle = (color, width = 1.5) => ({ borderColor: color, borderWidth: width, fill: false, pointRadius: 0 });

// --- 모달 컴포넌트 ---
const StockModal = ({ chartConfig, onClose }) => {
  const chartRef = useRef(null);
  if (!chartConfig) return null;

  const { title, data, type, options } = chartConfig;
  
  if (!data || !data.labels || !data.datasets) {
    return (
      <div className="stock-modal-overlay" onClick={onClose}>
        <div className="stock-modal-content" onClick={(e) => e.stopPropagation()}>
           <div className="stock-modal-header">
             <h3>{title}</h3>
             <button className="stock-modal-close-btn" onClick={onClose}>&times;</button>
           </div>
           <div className="stock-modal-chart-container">
             <ErrorDisplay message="Chart data is unavailable." />
           </div>
        </div>
      </div>
    );
  }

  const modalOptions = { ...options, maintainAspectRatio: false };

  const handleDownload = () => {
    const chartInstance = chartRef.current;
    if (chartInstance) {
      const image = chartInstance.toBase64Image('image/png', 1);
      const link = document.createElement('a');
      link.href = image;
      link.download = `${title.replace(/\s/g, '_')}_chart.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  return (
    <div className="stock-modal-overlay" onClick={onClose}>
      <div className="stock-modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="stock-modal-header">
          <h3>{title}</h3>
          <div>
            <button className="stock-modal-icon-btn" onClick={handleDownload}>
              <BiDownload />
            </button>
            <button className="stock-modal-close-btn" onClick={onClose}>&times;</button>
          </div>
        </div>
        <div className="stock-modal-chart-container">
          <ChartComponent 
            ref={chartRef} 
            type={type} 
            data={data} 
            options={modalOptions} 
          />
        </div>
      </div>
    </div>
  );
};


// --- [★★ 신규: 티커 검색 모달 ★★] ---
const TickerSearchModal = ({ isOpen, onClose, onSelectTicker }) => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  // 모달이 열릴 때 input에 포커스 및 상태 초기화
  useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 100);
      setQuery('');
      setResults([]);
      setError(null);
    }
  }, [isOpen]);

  // 검색 실행 함수
  const handleSearch = async () => {
    if (!query.trim()) return;
    setIsLoading(true);
    setError(null);
    setResults([]);
    try {
      const response = await axios.get(`${API_URL}/api/search-ticker?query=${query}`);
      setResults(response.data);
      if (response.data.length === 0) {
        setError("검색 결과가 없습니다. (주식(EQUITY)만 필터링됩니다)");
      }
    } catch (err) {
      setError("티커 검색 중 오류가 발생했습니다.");
    } finally {
      setIsLoading(false);
    }
  };

  // Enter 키로 검색
  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  // 티커 선택
  const handleSelect = (ticker) => {
    onSelectTicker(ticker);
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="stock-modal-overlay" onClick={onClose}>
      <div className="stock-modal-content ticker-search-modal" onClick={(e) => e.stopPropagation()}>
        <div className="stock-modal-header">
          <h3><BiSearchAlt />Finding Company Ticker</h3>
          <button className="stock-modal-close-btn" onClick={onClose}>&times;</button>
        </div>
        <div className="ticker-search-body">
          <div className="ticker-search-input-group">
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Enter Company Name"
            />
            <button onClick={handleSearch} disabled={isLoading}>
              {isLoading ? <LoadingSpinner /> : 'Search'}
            </button>
          </div>
          <div className="ticker-search-results-container">
            {isLoading && <div className="ticker-search-loading"><LoadingSpinner /></div>}
            {error && <div className="ticker-search-error"><ErrorDisplay message={error} /></div>}
            <ul className="ticker-search-results-list">
              {results.map((item) => (
                <li key={item.symbol} onClick={() => handleSelect(item.symbol)}>
                  <strong>{item.symbol}</strong>
                  <span>{item.shortname} ({item.exchange})</span>
                  <small>{item.longname}</small>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};


// --- [Events 탭 컴포넌트] ---
const EventsTab = ({ data1 }) => {
  const [selectedDate, setSelectedDate] = useState(null);
  const [news, setNews] = useState([]);
  const [isNewsLoading, setIsNewsLoading] = useState(false);
  const chartRef = useRef(null);

  const info = data1?.info || {};
  const charts = data1?.charts || {};
  const chartData = charts.line || { labels: [], datasets: [] }; 
  const companyName = info.shortName || info.symbol;

  // 날짜가 변경될 때마다 뉴스를 패치하는 useEffect
  useEffect(() => {
    if (!selectedDate || !data1) return;

    setIsNewsLoading(true);
    setNews([]); 
    const fetchNews = async () => {
      try {
        const response = await axios.post(`${API_URL}/api/news-for-date`, {
          date: selectedDate,
          ticker: info.symbol,
          companyName: companyName
        });
        setNews(response.data.domesticNews || []);
      } catch (err) {
        console.error("뉴스 패치 오류:", err);
        setNews([]);
      } finally {
        setIsNewsLoading(false);
      }
    };

    fetchNews();
  }, [selectedDate, info.symbol, companyName, data1]); 

  const handleChartClick = useCallback((event, elements, chart) => {
    if (elements.length > 0 && chartData.labels && chartData.labels.length > 0) {
      const index = elements[0].index;
      const date = chartData.labels[index];
      setSelectedDate(date); 
    }
  }, [chartData.labels]); 

  const chartOptions = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      title: { display: false },
      tooltip: { 
        mode: 'index', 
        intersect: false, 
        displayColors: false,
        callbacks: {
          title: function(tooltipItems) {
            return tooltipItems[0]?.label.split(',')[0];
          },
          label: function(tooltipItem) {
            return `Price: ${tooltipItem.formattedValue}`;
          }
        }
      },
    },
    scales: {
      x: { type: 'time', time: { unit: 'month', tooltipFormat: 'yyyy-MM-dd' }, ticks: { color: '#888' }, grid: { color: '#333' } },
      y: { ticks: { color: '#888', callback: (value) => value.toLocaleString() }, grid: { color: '#333' } },
    },
    interaction: { mode: 'index', axis: 'x', intersect: false },
    onClick: handleChartClick,
  }), [chartData.labels, handleChartClick]); 

  if (!data1) {
    return (
      <div className="events-tab-container">
        <div className="events-news-placeholder">
          <p>첫 번째 기업의 데이터를 먼저 로드하세요.</p>
        </div>
      </div>
    );
  }

  // 뉴스 카드 컴포넌트
  const NewsCard = ({ article }) => (
    <div className="news-card">
      <a href={article.url} target="_blank" rel="noopener noreferrer">
        <h6 className="news-card-title">{article.title}</h6>
        <span className="news-card-source">{article.source}</span>
        <p className="news-card-desc">{article.description}</p>
      </a>
    </div>
  );

  return (
    <div className="events-tab-container">
      <div className="events-chart-container">
        <Line ref={chartRef} data={chartData} options={chartOptions} />
      </div>
      <section className="events-news-section">
        <div className="events-news-header">
          <h4>
            <BiNews /> Related News
            {selectedDate && <span>{selectedDate}</span>}
          </h4>
          {isNewsLoading && <LoadingSpinner />}
        </div>
        
        {!selectedDate ? (
          <div className="events-news-placeholder">
            <p>그래프에서 특정 날짜를 클릭하여<br/>해당일의 국내 뉴스를 확인하세요.</p>
          </div>
        ) : (
          <div className="events-news-grid">
            <div className="events-news-column">
              <h5>국내 기사 (Google News)</h5>
              {!isNewsLoading && news.length === 0 && <p>관련 국내 기사가 없습니다.</p>}
              {news.map((article, index) => <NewsCard key={`dom-${index}`} article={article} />)}
            </div>
          </div>
        )}
      </section>
    </div>
  );
};


// --- [Federal Info 탭 컴포넌트 (수정됨)] ---
const FederalInfoTab = ({ onChartClick, data1 }) => {
  const [federalData, setFederalData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  
  const [localStockData, setLocalStockData] = useState(null);

  useEffect(() => {
    const ticker = data1?.info?.symbol;

    if (!ticker) {
      setIsLoading(false);
      setError("Please load a primary stock to see federal data comparison.");
      setFederalData(null); 
      setLocalStockData(null); 
      return;
    }

    const fetchAllData = async () => {
      setIsLoading(true);
      setError(null);
      setFederalData(null); 
      setLocalStockData(null); 

      const startDate = new Date();
      startDate.setFullYear(startDate.getFullYear() - 20);
      const startDateString = startDate.toISOString().split('T')[0];

      try {
        // 1. FRED 데이터 요청
        const fredRequests = Object.keys(FEDERAL_SERIES).map(seriesId => 
          axios.get(`${API_URL}/api/fred/${seriesId}?startDate=${startDateString}`)
            .then(res => ({ seriesId, data: res.data }))
            .catch(err => ({ seriesId, error: err.message }))
        );
        
        // 2. 주가 데이터 20년 기간으로 요청
        const stockRequest = axios.get(`${API_URL}/api/stock/${ticker}?startDate=${startDateString}`)
                                  .then(res => ({ type: 'stock', data: res.data }))
                                  .catch(err => ({ type: 'stock', error: err.message }));

        // 3. 모든 요청 병렬 실행
        const allResults = await Promise.all([...fredRequests, stockRequest]);

        // 4. FRED 결과 처리
        const dataMap = {};
        let hasFredError = false;
        allResults.filter(res => res.seriesId).forEach(res => {
          if (res.data) {
            dataMap[res.seriesId] = res.data;
          } else {
            console.error(`Failed to fetch FRED data for ${res.seriesId}: ${res.error}`);
            hasFredError = true;
          }
        });
        setFederalData(dataMap);

        // 5. 주가 결과 처리
        const stockResult = allResults.find(res => res.type === 'stock');
        if (stockResult && stockResult.data) {
          setLocalStockData(stockResult.data);
        } else {
          console.error(`Failed to fetch 20-year stock data: ${stockResult?.error}`);
          setError("Failed to load long-term stock data for comparison.");
          setLocalStockData(null);
        }

        if (hasFredError) {
           setError("Some FRED data couldn't be loaded."); 
        }

      } catch (err) {
        console.error("Failed to fetch data for FederalInfoTab", err);
        setError("Failed to load data for this tab.");
        setFederalData(null);
        setLocalStockData(null);
      } finally {
        setIsLoading(false);
      }
    };

    fetchAllData();
  }, [data1?.info?.symbol]); 

  const charts = useMemo(() => {
    if (!federalData || !localStockData) return {};
    
    return Object.keys(federalData).reduce((acc, seriesId) => {
      const data = federalData[seriesId];
      if (!data || data.length === 0) return acc;
      
      const labels = data.map(d => d.x); 
      const chartData = data.map(d => d.y); 
      const title = FEDERAL_SERIES[seriesId];
      
      const fredDataset = {
        label: seriesId,
        data: chartData,
        ...lineStyle(COLOR_1, 2),
        fill: true,
        backgroundColor: `${COLOR_1}1A`,
        yAxisID: 'y' // 왼쪽 Y축
      };

      let datasets = [fredDataset];

      if (localStockData && localStockData.charts && localStockData.charts.line) {
        const stockDataset = localStockData.charts.line.datasets[0];
        const stockInfo = localStockData.info;
        datasets.push({
          label: stockInfo.symbol || 'Stock Price',
          data: stockDataset.data, 
          ...lineStyle(COLOR_2, 1.5),
          yAxisID: 'y1' // 오른쪽 Y축
        });
      }
      
      acc[seriesId] = {
        title: title,
        type: 'line',
        data: {
          labels: labels, 
          datasets: datasets 
        }
      };
      return acc;
    }, {});

  }, [federalData, localStockData]); 

  if (isLoading) {
    return (
      <section className="stock-section">
        <LoadingSpinner />
      </section>
    );
  }

  if (error) {
    return (
      <section className="stock-section">
        <ErrorDisplay message={error} />
      </section>
    );
  }
  
  if ((!federalData || !localStockData) && !isLoading) {
    return (
        <section className="stock-section">
          <ErrorDisplay message="No data available." />
        </section>
    );
  }

  return (
    <section className="stock-section">
      <h4>Federal Reserve Economic Data (FRED) vs Stock Price (20-Year)</h4>
      {error && <ErrorDisplay message={error} />}
      
      <div className="stock-chart-grid">
        {Object.values(charts).map(chart => (
          <ChartCard 
            key={chart.title}
            title={chart.title} 
            data={chart.data} 
            type={chart.type} 
            onViewClick={onChartClick} 
          />
        ))}
        {Object.keys(charts).length === 0 && !isLoading && (
           <ErrorDisplay message="There is no chart to display the data." />
        )}
      </div>
    </section>
  );
};



// --- 메인 컴포넌트 ---
const Stock = () => {
  const [stockData, setStockData] = useState([]);
  const [loading, setLoading] = useState([true]); 
  const [errors, setErrors] = useState([null]);
  
  const [tickers, setTickers] = useState(['AAPL']);
  const [inputs, setInputs] = useState(['AAPL', '']); 
  
  const [selectedPeriod, setSelectedPeriod] = useState('1y');
  const [activeTab, setActiveTab] = useState(TABS[0]);

  const [modalChartConfig, setModalChartConfig] = useState(null);
  const openModal = (config) => setModalChartConfig(config);
  const closeModal = () => setModalChartConfig(null);
  
  const [fredData, setFredData] = useState(null);
  
  // const [isChatOpen, setIsChatOpen] = useState(false); // [주석 처리] 채팅 상태
  const [isTickerModalOpen, setIsTickerModalOpen] = useState(false); // [★★ 신규 ★★]

  const fetchStockData = useCallback(async (ticker, index) => {
    if (!ticker) {
      setStockData(prev => {
        const newData = [...prev];
        newData[index] = null;
        return newData;
      });
      setLoading(prev => {
        const newLoading = [...prev];
        newLoading[index] = false;
        return newLoading;
      });
      return;
    }
    
    setLoading(prev => {
      const newLoading = [...prev];
      newLoading[index] = true;
      return newLoading;
    });
    setErrors(prev => {
      const newErrors = [...prev];
      newErrors[index] = null;
      return newErrors;
    });

    try {
      const response = await axios.get(`${API_URL}/api/stock/${ticker}?period=${selectedPeriod}`);
      setStockData(prev => {
        const newData = [...prev];
        newData[index] = response.data;
        return newData;
      });
    } catch (err) {
      const errorMsg = err.response?.data?.error || `Failed to fetch data for ${ticker}.`;
      setErrors(prev => {
        const newErrors = [...prev];
        newErrors[index] = errorMsg;
        return newErrors;
      });
      setStockData(prev => {
        const newData = [...prev];
        newData[index] = null;
        return newData;
      });
    } finally {
      setLoading(prev => {
        const newLoading = [...prev];
        newLoading[index] = false;
        return newLoading;
      });
    }
  }, [selectedPeriod]); 

  useEffect(() => {
    const tickerCount = tickers.length;
    setStockData(prev => [...prev.slice(0, tickerCount), ...Array(Math.max(0, tickerCount - prev.length)).fill(null)]);
    setLoading(prev => [...prev.slice(0, tickerCount), ...Array(Math.max(0, tickerCount - prev.length)).fill(false)]);
    setErrors(prev => [...prev.slice(0, tickerCount), ...Array(Math.max(0, tickerCount - prev.length)).fill(null)]);

    tickers.forEach((ticker, index) => {
      fetchStockData(ticker, index);
    });
  }, [tickers, selectedPeriod, fetchStockData]);
  
  useEffect(() => {
    const fetchFredData = async () => {
      const data1 = stockData[0]; 
      if (!data1 || data1.hist.length === 0) return;
      const startDate = data1.hist[0].date;
      
      try {
        const res = await axios.get(`${API_URL}/api/fred/DGS10?startDate=${startDate}`);
        setFredData(res.data);
      } catch (err) {
        console.error("Failed to fetch FRED data", err);
        setFredData(null); 
      }
    };
    
    fetchFredData();
  }, [stockData]); 

  const handleInputChange = (e, index) => {
    const newInputs = [...inputs];
    newInputs[index] = e.target.value;
    setInputs(newInputs);
  };

  const handleSearch = (e, index) => {
    if (e.key === 'Enter') {
      const newTicker = inputs[index].toUpperCase().trim();
      if (!newTicker) {
        if (index > 0) handleRemove(index);
        return;
      }
      
      const newTickers = [...tickers];
      if (index < newTickers.length) {
        newTickers[index] = newTicker;
      } else {
        newTickers.push(newTicker);
      }
      setTickers(newTickers);
      
      if (index === inputs.length - 1) {
        setInputs([...inputs, '']);
      }
    }
  };

  const handleRemove = (indexToRemove) => {
    if (indexToRemove === 0) return; 

    setTickers(prev => prev.filter((_, i) => i !== indexToRemove));
    setInputs(prev => prev.filter((_, i) => i !== indexToRemove));
  };

  // [★★ 신규: 티커 선택 핸들러 ★★]
  const handleSelectTicker = (ticker) => {
    const newInputs = [...inputs];
    newInputs[0] = ticker; // 첫 번째 검색창에 선택한 티커를 설정
    setInputs(newInputs);
    
    const newTickers = [...tickers];
    newTickers[0] = ticker; // 메인 티커를 변경
    setTickers(newTickers);
    
    setIsTickerModalOpen(false); // 모달 닫기
  };

  const mainLoading = loading[0] && !stockData[0];
  const mainError = errors[0] && !stockData[0];
  const data1 = stockData[0]; 

  return (
    <>
      <Nav />
      <StockModal chartConfig={modalChartConfig} onClose={closeModal} />
      {/* [★★ 신규: 티커 검색 모달 추가 ★★] */}
      <TickerSearchModal 
        isOpen={isTickerModalOpen}
        onClose={() => setIsTickerModalOpen(false)}
        onSelectTicker={handleSelectTicker}
      />
      
      <div className="stock-dashboard-container">
        <header className="stock-top-header">
          <div className="stock-search-controls">
            {/* [★★ 신규: 티커 검색 버튼 ★★] */}
            <button 
              className="stock-ticker-search-btn" 
              onClick={() => setIsTickerModalOpen(true)}
              title="기업 티커 검색"
            >
              <BiSearchAlt />
              <span>티커 찾기</span>
            </button>
            
            {inputs.map((input, index) => (
              <div 
                key={index} 
                className={`stock-search-bar ${index > 0 ? 'stock-compare-bar' : ''}`}
              >
                <BiSearch />
                <input 
                  type="text" 
                  value={input} 
                  onChange={(e) => handleInputChange(e, index)} 
                  onKeyDown={(e) => handleSearch(e, index)} 
                  placeholder={index === 0 ? "e.g., AAPL" : "Add comparison..."} 
                />
                {index > 0 && index < inputs.length - 1 && (
                  <button onClick={() => handleRemove(index)} className="stock-remove-btn"><BiX /></button>
                )}
              </div>
            ))}
          </div>
          <div className="stock-period-selector">
            {periodOptions.map(opt => (
              <button key={opt.value} className={`stock-period-btn ${selectedPeriod === opt.value ? 'stock-active' : ''}`} onClick={() => setSelectedPeriod(opt.value)}>
                {opt.label}
              </button>
            ))}
          </div>
        </header>

        <nav className="stock-tab-nav">
          {TABS.map(tab => (
            <button key={tab} className={`stock-tab-btn ${activeTab === tab ? 'stock-active' : ''}`} onClick={() => setActiveTab(tab)}>
              {tab}
            </button>
          ))}
        </nav>

        <main className="stock-main-content">
          {mainLoading ? <LoadingSpinner /> : mainError ? <ErrorDisplay message={mainError} /> : (
            <>
              {activeTab === 'Dashboard' && <DashboardTab dataCollection={stockData} loadingCollection={loading} errorCollection={errors} fredData={fredData} onChartClick={openModal} />}
              {activeTab === 'Financials' && <FinancialsTab dataCollection={stockData} onChartClick={openModal} />}
              {activeTab === 'Historical Data' && <HistoricalDataTab dataCollection={stockData} onChartClick={openModal} />}
              {activeTab === 'Events' && <EventsTab data1={data1} />}
              {activeTab === 'Federal Info' && <FederalInfoTab onChartClick={openModal} data1={data1} />}
            </>
          )}
        </main>
      </div>
      
      {/* [주석 처리] 채팅 모달 관련 컴포넌트 */}
      {/*
      <ChatModal 
        isOpen={isChatOpen} 
        onClose={() => setIsChatOpen(false)} 
      />
      <button 
        className={`chat-fab ${isChatOpen ? 'chat-fab-hidden' : ''}`}
        onClick={() => setIsChatOpen(true)}
        title="Stock-AI 채팅 열기"
      >
        <FaRobot />
      </button>
      */}
    </>
  );
};


// --- 탭 컴포넌트들 ---
const DashboardTab = ({ dataCollection, loadingCollection, errorCollection, fredData, onChartClick }) => {
  
  const performanceChart = useMemo(() => {
    const validData = dataCollection.filter(d => d && d.hist.length > 0);
    if (validData.length < 1) return null;

    const allDatesSet = new Set();
    const dataMaps = validData.map(data => {
      data.hist.forEach(d => allDatesSet.add(d.date));
      return new Map(data.hist.map(d => [d.date, d]));
    });
    const allDates = [...allDatesSet].sort();

    const datasets = validData.map((data, index) => {
      const hist = data.hist;
      const baseValue = hist[0].close;
      const dataMap = dataMaps[index];
      const color = COLORS[dataCollection.indexOf(data) % COLORS.length]; 

      return { 
        label: data.info.symbol, 
        data: allDates.map(date => {
          if (!dataMap.has(date)) return null;
          const price = dataMap.get(date).close;
          if (baseValue === 0) return null; 
          return ((price - baseValue) / baseValue * 100).toFixed(2);
        }), 
        ...lineStyle(color, 2),
        spanGaps: true 
      };
    });
    return { labels: allDates, datasets };
  }, [dataCollection]);

  const fredChart = useMemo(() => {
    const data1 = dataCollection[0]; 
    if (!fredData || !data1) return null;
    const s1 = data1.info.symbol;
    return {
      labels: data1.charts.line.labels, 
      datasets: [
        { label: s1, data: data1.charts.line.datasets[0].data, ...lineStyle(COLOR_1, 2), yAxisID: 'y' },
        { label: '10-Yr Treasury (DGS10)', data: fredData, ...lineStyle(COLOR_2, 2), yAxisID: 'y1', spanGaps: true }
      ]
    }
  }, [fredData, dataCollection]);

  const charts = useMemo(() => {
    const validData = dataCollection.filter(Boolean);
    if (validData.length === 0) return {};

    // --- BRANCH 1: 단일 기업 뷰 (원본 색상 사용) ---
    if (validData.length === 1) {
        const data1 = validData[0];
        // 서버에서 받은, 이미 여러 색상으로 스타일링된 차트 객체를 그대로 반환
        return {
            price: { title: "Price (Candlestick)", data: data1.charts.candlestick, type: 'bar' },
            heikinAshi: { title: "Heikin Ashi", data: data1.charts.heikinAshi, type: 'bar' }, 
            cumulativeReturn: { title: "Cumulative Return", data: data1.charts.cumulativeReturn, type: 'line' },
            sma: { title: "Moving Averages (SMA)", data: data1.charts.sma, type: 'line' },
            vma: { title: "VMA (20D)", data: data1.charts.vma, type: 'line' },
            dailyChange: { title: "% Change (Day-over-Day)", data: data1.charts.dailyChange, type: 'bar' },
            dailyRange: { title: "Daily Price Range (H-L)", data: data1.charts.dailyRange, type: 'line' },
            volume: { title: "Volume", data: data1.charts.volume, type: 'bar' },
            obv: { title: "On-Balance Volume (OBV)", data: data1.charts.obv, type: 'line' },
            momentum: { title: "Momentum (10D)", data: data1.charts.momentum, type: 'line' },
            volatility: { title: "Volatility (20D SD)", data: data1.charts.volatility, type: 'line' },
            macd: { title: "MACD", data: data1.charts.macd, type: 'line' },
            rsi: { title: "RSI", data: data1.charts.rsi, type: 'line' },
            stochastic: { title: "Stochastic Oscillator", data: data1.charts.stochastic, type: 'line' },
            bollinger: { title: "Bollinger Bands", data: data1.charts.bollinger, type: 'line' },
        };
    }

    // --- BRANCH 2: 다중 기업 뷰 (기업별 색상 할당) ---
    const labels = validData[0].charts.line.labels; // 첫 번째 데이터의 레이블 기준

    // Helper: createChart (for charts with multiple datasets like SMA, MACD)
    const createChart = (title, type, chartName, fields) => {
      const datasets = [];
      validData.forEach((data, index) => {
        const s = data.info.symbol;
        const chart = data.charts[chartName];
        if (!chart) return;

        fields.forEach((field, i) => {
          if (!chart.datasets[field.index]) return; 
          const ds = chart.datasets[field.index];
          const color = COLORS[index % COLORS.length];
          const altColor = COLORS[(index + 1) % COLORS.length]; 
          
          let finalColor = color;
          // 보조 지표(SMA50, Signal, %D)는 다른 색상 사용
          if (field.label.includes('SMA50') || field.label.includes('Signal') || field.label.includes('%D')) {
             finalColor = altColor;
          }

          datasets.push({
            label: `${s} ${field.label}`,
            data: ds.data,
            ...lineStyle(finalColor),
            type: field.type || type,
            backgroundColor: finalColor,
            fill: false, // 비교 차트에서는 fill을 끕니다.
          });
        });
      });
      return { title, data: { labels, datasets }, type };
    };
    
    // Helper: createSingleDsChart (for charts with one dataset like RSI, VMA)
    const createSingleDsChart = (title, type, chartName) => {
       const datasets = [];
        validData.forEach((data, index) => {
          if (!data.charts[chartName] || !data.charts[chartName].datasets[0]) return;
          
          const s = data.info.symbol;
          const ds = data.charts[chartName].datasets[0];
          const color = COLORS[index % COLORS.length]; // e.g., '#a0c4e0'
          
          datasets.push({
            ...ds, 
            label: `${s} ${ds.label}`,
            data: ds.data,
            borderColor: color, // Solid line color
            
            backgroundColor: ds.fill ? `${color}1A` : undefined, 
            
            type: type,
          });
        });
        return { title, data: { labels, datasets }, type };
    }

    // 가격 차트는 항상 'line'
    const priceChart = createChart("Price (Line)", 'line', 'line', [{ label: 'Price', index: 0 }]);

    return {
      price: priceChart,
      cumulativeReturn: createSingleDsChart("Cumulative Return", 'line', 'cumulativeReturn'),
      sma: createChart("Moving Averages (SMA)", 'line', 'sma', [{ label: 'SMA20', index: 0 }, { label: 'SMA50', index: 1 }]),
      vma: createSingleDsChart("VMA (20D)", 'line', 'vma'),
      dailyChange: createSingleDsChart("% Change (Day-over-Day)", 'bar', 'dailyChange'),
      dailyRange: createSingleDsChart("Daily Price Range (H-L)", 'line', 'dailyRange'),
      volume: createChart("Volume Comparison", 'line', 'volume', [{ label: 'Volume', index: 0 }]),
      obv: createSingleDsChart("On-Balance Volume (OBV)", 'line', 'obv'),
      momentum: createSingleDsChart("Momentum (10D)", 'line', 'momentum'),
      volatility: createSingleDsChart("Volatility (20D SD)", 'line', 'volatility'),
      macd: createChart("MACD", 'line', 'macd', [{ label: 'MACD', index: 0 }, { label: 'Signal', index: 1 }]),
      rsi: createSingleDsChart("RSI", 'line', 'rsi'),
      stochastic: createChart("Stochastic Oscillator", 'line', 'stochastic', [{ label: '%K', index: 0 }, { label: '%D', index: 1 }]),
      bollinger: createChart("Bollinger Bands (Middle)", 'line', 'bollinger', [{ label: 'Middle', index: 1 }]),
    };
  }, [dataCollection]);
  

  return (
    <>
      <section className="stock-section stock-header-section">
        {dataCollection.map((data, index) => {
          if (loadingCollection[index]) {
            return <div key={index} className="stock-loading-section-small"><LoadingSpinner /></div>;
          }
          if (errorCollection[index]) {
            return <div key={index} className="stock-error-section-small"><ErrorDisplay message={errorCollection[index]} /></div>;
          }
          if (data) {
            return <HeaderInfo key={index} data={data} />;
          }
          return null;
        })}
      </section>
      
      {performanceChart && dataCollection.filter(Boolean).length > 1 && (
        <section className="stock-section">
          <ChartHeader title="Performance Comparison (%)" onChartClick={() => onChartClick({ title: "Performance Comparison (%)", data: performanceChart, type: 'line', options: getChartOptions("Performance Comparison (%)", true) })} />
          <div className="stock-chart-container-large">
             <ChartComponent type='line' data={performanceChart} options={getChartOptions("Performance Comparison (%)", true)} />
          </div>
        </section>
      )}

      {fredChart && dataCollection[0] && (
        <section className="stock-section">
          <ChartHeader title="Price vs 10-Yr Treasury" onChartClick={() => {
            const chartOptions = getChartOptions("Price vs 10-Yr Treasury");
            chartOptions.scales.y1 = { type: 'linear', display: true, position: 'right', ticks: { color: COLOR_2 }, grid: { drawOnChartArea: false } };
            chartOptions.scales.y.ticks.color = COLOR_1;
            onChartClick({ title: "Price vs 10-Yr Treasury", data: fredChart, type: 'line', options: chartOptions });
          }} />
          <div className="stock-chart-container-large">
             <ChartComponent type='line' data={fredChart} options={{
                ...getChartOptions("Price vs 10-Yr Treasury"),
                scales: {
                  ...getChartOptions("Price vs 10-Yr Treasury").scales,
                  y: { ...getChartOptions("Price vs 10-Yr Treasury").scales.y, ticks: { color: COLOR_1 } },
                  y1: { type: 'linear', display: true, position: 'right', ticks: { color: COLOR_2 }, grid: { drawOnChartArea: false } }
                }
             }} />
          </div>
        </section>
      )}

      <section className="stock-section">
        <h4>Technical Analysis Charts</h4>
        <div className="stock-chart-grid">
          {Object.values(charts).map(chart => (
            <ChartCard 
              key={chart.title}
              title={chart.title} 
              data={chart.data} 
              type={chart.type} 
              onViewClick={onChartClick} 
            />
          ))}
        </div>
      </section>
    </>
  );
};

const FinancialsChart = ({ data, info, type, onChartClick }) => {
  const chartData = useMemo(() => {
    if (!data && !info) return null; 
    let datasets = [];
    let labels = [];
    
    if (type === 'quarterly') {
      if (!data) return null; 
      const metrics = ['Total Revenue', 'Net Income'];
      const firstMetricData = data[metrics[0]];
      if (!firstMetricData) return null;
      labels = Object.keys(firstMetricData).map(date => date.split('T')[0]); 
      datasets = metrics.map((metric, index) => {
        if (!data[metric]) return null;
        return {
          label: metric,
          data: Object.values(data[metric]),
          backgroundColor: index === 0 ? COLOR_1 : COLOR_2,
          type: 'bar',
        };
      }).filter(Boolean);
    } 
    else if (type === 'valuation') {
      if (!info) return null; 
      labels = [info.symbol]; 
      const evEbitda = (info.enterpriseValue && info.ebitda) ? (info.enterpriseValue / info.ebitda) : null;
      const interestCoverage = (info.ebit && info.interestExpense && info.interestExpense !== 0) ? (info.ebit / Math.abs(info.interestExpense)) : null;
      datasets = [
        { label: 'EV/EBITDA', data: [evEbitda], backgroundColor: COLOR_1, yAxisID: 'y' },
        { label: 'Interest Coverage', data: [interestCoverage], backgroundColor: COLOR_2, yAxisID: 'y1' }
      ];
    }

    return { labels, datasets };
  }, [data, info, type]);

  if (!chartData || chartData.datasets.length === 0) {
     return (
        <div className="stock-chart-card-large">
          <ChartHeader title={type === 'quarterly' ? "Quarterly Revenue & Net Income" : "Valuation Metrics"} onChartClick={null} />
          <div className="stock-chart-container-large">
            <ErrorDisplay message="No data available for this chart." />
          </div>
        </div>
     );
  }

  const chartTitle = type === 'quarterly' ? "Quarterly Revenue & Net Income" : "Valuation Metrics";
  const chartOptions = getChartOptions(chartTitle);

  if (type === 'valuation') {
    chartOptions.scales.x.type = 'category'; 
    chartOptions.scales.y1 = { type: 'linear', display: true, position: 'right', ticks: { color: COLOR_2 }, grid: { drawOnChartArea: false } };
  }

  return (
    <div className="stock-chart-card-large">
      <ChartHeader 
        title={chartTitle} 
        onChartClick={() => onChartClick({ title: chartTitle, data: chartData, type: 'bar', options: chartOptions })} 
      />
      <div className="stock-chart-container-large">
        <Bar data={chartData} options={chartOptions} />
      </div>
    </div>
  );
};


const FinancialsTab = ({ dataCollection, onChartClick }) => {
  const [statement, setStatement] = useState('income_stmt');
  return (
    <section className="stock-section">
      <div className="stock-charts-row">
        {dataCollection.map((data, index) => (
          data && 
          <FinancialsChart 
            key={index}
            data={data.financials.quarterly_income_stmt} 
            type="quarterly"
            onChartClick={onChartClick}
          />
        ))}
      </div>
      <div className="stock-charts-row">
         {dataCollection.map((data, index) => (
          data &&
          <FinancialsChart
            key={index} 
            data={null} 
            info={data.info}
            type="valuation"
            onChartClick={onChartClick}
          />
         ))}
      </div>

      <div className="stock-table-row">
        <div className="stock-financials-controls">
          <h4>Financial Statements (Annual)</h4>
          <div className="stock-pills">
            <button onClick={() => setStatement('income_stmt')} className={statement === 'income_stmt' ? 'stock-active' : ''}>Income Statement</button>
            <button onClick={() => setStatement('balance_sheet')} className={statement === 'balance_sheet' ? 'stock-active' : ''}>Balance Sheet</button>
            <button onClick={() => setStatement('cash_flow')} className={statement === 'cash_flow' ? 'stock-active' : ''}>Cash Flow</button>
          </div>
        </div>
        <div className="stock-financials-tables-container">
          {dataCollection.map((data, index) => (
            data &&
            <FinancialsTable 
              key={index} 
              title={data.info.symbol} 
              data={data.financials[statement]} 
            />
          ))}
        </div>
      </div>
    </section>
  );
};

const HistoricalDataTab = ({ dataCollection, onChartClick }) => {
    
    if (dataCollection.length === 0 || !dataCollection[0]) {
      return (
        <section className="stock-section">
          {dataCollection.length > 0 && !dataCollection[0] ? 
            <ErrorDisplay message="Failed to load data for the primary ticker." /> :
            <ErrorDisplay message="Historical data not available." />
          }
        </section>
      );
    }

    // 그리드 레이아웃을 사용
    return (
        <section className="stock-section stock-financials-tables-container">
            {dataCollection.map((data, index) => {
              if (!data) {
                return <ErrorDisplay key={index} message={`Data not loaded for ticker ${index+1}`} />;
              }
              
              const chartData = data.charts.line; 
              const chartOptions = getChartOptions('Historical Price (Line)');
              const chartTitle = `Historical Price (Line) - ${data.info.symbol}`;

              return (
                <div key={index} className="stock-historical-data-column">
                  {/* 1. 차트 */}
                  <div className="stock-chart-card-large">
                      <ChartHeader 
                          title={chartTitle}
                          onChartClick={() => onChartClick({ title: chartTitle, data: chartData, type: 'line', options: chartOptions })}
                      />
                      <div className="stock-chart-container-large">
                          <Line data={chartData} options={chartOptions} />
                      </div>
                  </div>
                  {/* 2. 테이블 */}
                  <div className="stock-table-row">
                      <h4>Historical Price Data for {data.info.symbol}</h4>
                      <div className="stock-table-container">
                          <table className="stock-data-table">
                              <thead>
                                  <tr>
                                      <th>Date</th>
                                      <th>Open</th>
                                      <th>High</th>
                                      <th>Low</th>
                                      <th>Close</th>
                                      <th>Volume</th>
                                  </tr>
                              </thead>
                              <tbody>
                                  {[...data.hist].reverse().map(row => (
                                      <tr key={row.date}>
                                          <td>{row.date}</td>
                                          <td>{row.open?.toLocaleString()}</td>
                                          <td>{row.high?.toLocaleString()}</td>
                                          <td>{row.low?.toLocaleString()}</td>
                                          <td>{row.close?.toLocaleString()}</td>
                                          <td>{row.volume?.toLocaleString()}</td>
                                      </tr>
                                  ))}
                              </tbody>
                          </table>
                      </div>
                  </div>
                </div>
              );
            })}
        </section>
    );
};

// --- 보조 컴포넌트들 ---
const HeaderInfo = ({ data }) => {
  const { quote, info } = data;
  const changeClass = quote.change >= 0 ? 'stock-up' : 'stock-down';
  return (
    <div className="stock-header-info">
      <h1>{info.shortName} ({info.symbol})</h1>
      <div className="stock-price-large">
        <span>{quote.price?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} {info.currency}</span>
        <span className={`stock-price-change ${changeClass}`}>
          {quote.change?.toFixed(2)} ({quote.changePercent?.toFixed(2)}%)
        </span>
      </div>
      <div className="stock-key-stats">
        <div><span>Market Cap</span><span>{(info.marketCap / 1_000_000_000).toFixed(2)}B</span></div>
        <div><span>P/E Ratio</span><span>{info.trailingPE?.toFixed(2) || 'N/A'}</span></div>
        <div><span>Dividend Yield</span><span>{info.dividendYield ? (info.dividendYield * 100).toFixed(2) + '%' : 'N/A'}</span></div>
      </div>
    </div>
  );
};

const ChartHeader = ({ title, onChartClick }) => {
  const tooltipText = chartTooltips[title] || "No information available.";
  
  return (
    <div className="stock-chart-header">
      <div className="stock-chart-title-group">
        <h5>{title}</h5>
        <div className="stock-tooltip-container">
          <BiQuestionMark className="stock-tooltip-icon" />
          <div className="stock-tooltip-content">{tooltipText}</div>
        </div>
      </div>
      {onChartClick && (
        <button className="stock-view-btn" onClick={onChartClick}>
          <BiShow /> View
        </button>
      )}
    </div>
  );
};

const ChartCard = ({ title, data, type, onViewClick }) => {
  const chartOptions = getChartOptions(title);
  
  if (title === "Volume Comparison") {
    //
  }
  
  if (title.includes("Candlestick") || title.includes("Heikin Ashi")) { 
    chartOptions.plugins.tooltip.mode = 'index';
    chartOptions.plugins.tooltip.intersect = false;
    chartOptions.scales.x.stacked = true; 
  }
  
  if (Object.values(FEDERAL_SERIES).includes(title)) {
      chartOptions.scales.x.time.unit = 'year'; 
      
      chartOptions.scales.y1 = {
        type: 'linear',
        display: true,
        position: 'right',
        ticks: { color: COLOR_2 }, 
        grid: { drawOnChartArea: false } 
      };
      
      chartOptions.scales.y.ticks.color = COLOR_1; 
  }

  if (!data || !data.labels || !data.datasets || data.datasets.length === 0) {
    return (
      <div className="stock-chart-card">
        <ChartHeader title={title} onChartClick={null} /> 
        <div className="stock-chart-wrapper">
          <ErrorDisplay message={`No data for ${title}`} />
        </div>
      </div>
    );
  }

  return (
    <div className="stock-chart-card">
      <ChartHeader title={title} onChartClick={() => onViewClick({ title, data, type, options: chartOptions })} />
      <div className="stock-chart-wrapper">
        <ChartComponent 
          type={type || 'bar'}
          data={data} 
          options={chartOptions} 
        />
      </div>
    </div>
  );
};


const FinancialsTable = ({ title, data }) => {
    const firstMetric = data && Object.keys(data).length > 0 ? Object.values(data)[0] : null;
    const years = firstMetric ? Object.keys(firstMetric) : [];

    if (years.length === 0) {
        return <div className="stock-financials-table-wrapper"><p>No data available for {title}.</p></div>;
    }
    
    const metrics = Object.keys(data);

    return (
        <div className="stock-financials-table-wrapper">
            <h5>{title}</h5>
            <div className="stock-table-container">
                <table className="stock-financials-table">
                    <thead>
                        <tr>
                            <th>Metric</th>
                            {years.map(year => (
                              <th key={year}>{new Date(year).getFullYear() || year}</th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {metrics.map(metric => (
                            <tr key={metric}>
                                <td>{metric.replace(/([A-Z])/g, ' $1').trim()}</td>
                                {years.map(year => (
                                    <td key={year}>
                                        {data[metric][year] ? (data[metric][year] / 1_000_000).toFixed(0).toLocaleString() + 'M' : '–'}
                                    </td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default Stock;