// /Users/imhyeonseok/Documents/stock/web/src/components/main/main.js
import React, { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import Nav from '../nav/nav'; 

// Swiper import
import { Swiper, SwiperSlide } from 'swiper/react';
import { Navigation, Autoplay, EffectFade } from 'swiper/modules';
import 'swiper/css';
import 'swiper/css/navigation';
import 'swiper/css/autoplay';
import 'swiper/css/effect-fade'; 

// Icons
import { FiChevronLeft, FiChevronRight } from "react-icons/fi";

// CSS & Assets
import './main.css';
import chartimg from '../../assets/chart.png';
import mountainImg from '../../assets/mountain.jpg'; 
import yahooimg from '../../assets/yahoo.png';
import priceimg from '../../assets/price.png';
import compareimg from '../../assets/compare.png';
import futureimg from '../../assets/future.png';

// Section 2 Data
const modelData = [
  { 
    title: 'Trusted Data Sources', 
    text: 'Powered by official APIs including Yahoo Finance and FRED. We ensure high-reliability data for every market decision.',
    img: yahooimg 
  },
  { 
    title: 'Comprehensive Data', 
    text: 'Access a full spectrum of data: from real-time stock prices to detailed financial statements and deep analysis reports.',
    img: priceimg 
  },
  { 
    title: 'Visual Analytics', 
    text: 'Interactive charts and technical graphs. We visualize complex trends to provide clear, actionable market guidance.',
    img: chartimg 
  },
  { 
    title: 'Peer Comparison', 
    text: 'Compare multiple companies side-by-side. Analyze relative strength, valuation gaps, and performance metrics instantly.',
    img: compareimg 
  },
  { 
    title: 'Continuous Evolution', 
    text: 'Our platform never stops learning. We continuously deploy new analytical features and algorithm upgrades for deeper insights.',
    img: futureimg 
  },
];

// Section 3 Data - 요청하신 4가지 핵심 가치 적용
const dataData = [
  { title: 'Lowering Entry Barriers', className: 'main-img01' },   // 1. 주식 이용의 문턱을 낮춤
  { title: 'Data Democratization', className: 'main-img02' },      // 2. 정보 및 데이터 민주화
  { title: 'Rational Insights', className: 'main-img03' },         // 3. 효율적이고 합리적인 인사이트 제공
  { title: 'Sustainable Growth', className: 'main-img04' },        // 4. 지속 가능한 성장과 발전
];

const Main = () => {
  const scrollContainerRef = useRef(null);
  const progressBarRef = useRef(null);
  const isScrolling = useRef(false);
  const dataSwiperRef = useRef(null);
  const [dataSlideIndex, setDataSlideIndex] = useState(0);
  
  const [dataHeading, setDataHeading] = useState("Your Gateway to Financial Freedom");
  const [isFooterVisible, setIsFooterVisible] = useState(false);

  // 세로 -> 가로 스크롤 변환
  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container) return;
    const onWheel = (e) => {
      e.preventDefault();
      if (isScrolling.current) return;
      isScrolling.current = true;
      const scrollAmount = container.clientWidth;
      if (e.deltaY > 0) {
        container.scrollBy({ left: scrollAmount, behavior: 'smooth' });
      } else {
        container.scrollBy({ left: -scrollAmount, behavior: 'smooth' });
      }
      setTimeout(() => {
        isScrolling.current = false;
      }, 800); 
    };
    container.addEventListener('wheel', onWheel, { passive: false });
    return () => {
      container.removeEventListener('wheel', onWheel);
    };
  }, []); 

  // 프로그레스 바 & 푸터 감지
  useEffect(() => {
    const container = scrollContainerRef.current;
    const bar = progressBarRef.current;
    if (!container || !bar) return;
    const onScroll = () => {
      const scrollLeft = container.scrollLeft;
      const totalScrollableWidth = container.scrollWidth - container.clientWidth;
      if (totalScrollableWidth === 0) {
        bar.style.width = '0%';
        return;
      }
      
      const totalSections = container.children.length;
      const progressPerSection = 100 / (totalSections - 1);
      const currentSectionIndex = Math.round(scrollLeft / container.clientWidth);
      const correctedProgress = currentSectionIndex * progressPerSection;

      bar.style.width = `${correctedProgress}%`;
      setIsFooterVisible(currentSectionIndex === totalSections - 1);
    };
    container.addEventListener('scroll', onScroll);
    onScroll(); 
    return () => {
      container.removeEventListener('scroll', onScroll);
    };
  }, []); 

  // Data 헤딩 변경 - 4가지 가치에 대한 상세 설명(슬로건)
  useEffect(() => {
    const titles = [
      "Making Investment Accessible to Everyone",  // 1. 주식 이용의 문턱을 낮춤
      "Institutional Grade Data, Open to All",     // 2. 정보 및 데이터 민주화
      "Efficiency Meets Strategic Clarity",        // 3. 효율적이고 합리적인 인사이트 제공
      "Evolving Platform, Growing Wealth"          // 4. 지속 가능한 성장과 발전
    ];
    
    const currentTitle = titles[dataSlideIndex % titles.length] || titles[0];
    setDataHeading(currentTitle);
  }, [dataSlideIndex]);

  const handleProgressClick = (index) => {
    if (dataSwiperRef.current && index !== dataSlideIndex) {
      dataSwiperRef.current.slideToLoop(index);
    }
  };

  return (
    <>
      <Nav /> 
      <main 
        className="main-scroll-container" 
        ref={scrollContainerRef}
        style={{ backgroundImage: `linear-gradient(rgba(5, 5, 5, 0.9), rgba(5, 5, 5, 0.9)), url(${mountainImg})` }}
      >

        {/* --- Section 1: Vision --- */}
        <section className="main-section" id="section-one">
          <div className="main-abstract-blob"></div>
          <div className="main-content-left">
            <header className="main-header-name">SMART INVESTMENT PLATFORM</header>
            <div className="main-title">
              <p className="main-line1">Navigate the</p>
              <p className="main-line2">Market with</p>
              <p className="main-line3">Confidence</p>
            </div>
            <div className="main-mission-area">
              <span className="main-mission-title">◎ OUR MISSION</span>
              <p className="main-mission-text">
                To empower individual investors with institutional-grade data, 
                intuitive analysis tools, and real-time insights. 
                Make smarter decisions and build a stronger portfolio today.
              </p>
            </div>
          </div>
        </section>

        {/* --- Section 2: Features --- */}
        <section className="main-section main-model-slider-section" id="section-models">
          <div className="main-inner">
            <div className="main-tit_bar">
              <p>Key Features</p>
              <div className="main-slide_arrow">
                <span className="main-arrow main-prev"><FiChevronLeft /></span>
                <span className="main-arrow main-next"><FiChevronRight /></span>
              </div>
            </div>
            <div className="main-latest_wrap">
              <Swiper
                modules={[Navigation, Autoplay]}
                loop={true}
                speed={1000}
                slidesPerView={1.5}
                spaceBetween={12}
                centeredSlides={true}
                navigation={{
                  prevEl: '.main-slide_arrow .main-prev',
                  nextEl: '.main-slide_arrow .main-next',
                }}
                autoplay={{ delay: 3000, disableOnInteraction: false }}
                breakpoints={{
                  1025: { slidesPerView: 4, spaceBetween: 30, centeredSlides: false },
                  769: { slidesPerView: 3, spaceBetween: 20, centeredSlides: true },
                  481: { slidesPerView: 3, spaceBetween: 15, centeredSlides: true },
                  381: { slidesPerView: 2, spaceBetween: 15, centeredSlides: false },
                }}
                className="main-latest_slide"
              >
                {modelData.map((item, index) => (
                  <SwiperSlide key={index}>
                    <a href="#">
                      <div className="main-img" style={{ backgroundImage: `url(${item.img})` }}></div>
                      <div className="main-info">
                        <p className="main-tit">{item.title}</p>
                        <p className="main-txt">{item.text}</p>
                      </div>
                    </a>
                  </SwiperSlide>
                ))}
              </Swiper>
            </div>
          </div>
        </section>

        {/* --- Section 3: Core Values --- */}
        <section className="main-section" id="section-data">
          <div className="main-slider_wrap">
            <Swiper
              modules={[Autoplay, EffectFade]}
              effect="fade"
              fadeEffect={{ crossFade: true }}
              loop={true}
              autoplay={{ delay: 4000, disableOnInteraction: false }}
              speed={500}
              allowTouchMove={false}
              className="main-data-slider"
              onInit={(swiper) => { dataSwiperRef.current = swiper; }}
              onSlideChange={(swiper) => { setDataSlideIndex(swiper.realIndex); }}
            >
              {dataData.map((item, index) => (
                <SwiperSlide 
                  key={index} 
                  className={item.className}
                  style={{ backgroundImage: `url(${mountainImg})` }} 
                ></SwiperSlide>
              ))}
            </Swiper>
            
            <div className="main-cont">
              <div className="main-txt">
                <b>Our Core Philosophy</b>
                <h2 key={dataSlideIndex} className="main-data-heading-animate">{dataHeading}</h2>
              </div>
              <ul className="main-progress">
                {dataData.map((item, index) => (
                  <li 
                    key={index} 
                    className={index === dataSlideIndex ? 'main-active' : ''} 
                    onClick={() => handleProgressClick(index)}
                  >
                    <p>{item.title}</p>
                    <span className="main-bar"></span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </section>

        {/* --- Section 4: Why Choose Us (Updated) --- */}
        <section className="main-section main-feature-section" id="section-stack">
           <h2>Why Choose Us?</h2>
          <div className="main-feature-grid">
            
            {/* 1. 데이터 신뢰성과 민주화 */}
            <div className="main-feature-card">
              <h4>Institutional-Grade Data Access</h4>
              <p>
                We democratize financial information by providing official data from Yahoo Finance and FRED. 
                Experience the same high-reliability data used by institutional investors, open to everyone.
              </p>
            </div>

            {/* 2. 시각화와 합리적 통찰 */}
            <div className="main-feature-card">
              <h4>Rational & Visual Clarity</h4>
              <p>
                Stop guessing and start analyzing. Our interactive charts and technical graphs transform complex 
                market trends into clear, rational insights for smarter decision-making.
              </p>
            </div>

            {/* 3. 진입장벽 완화와 비교 분석 */}
            <div className="main-feature-card">
              <h4>Effortless Comparative Analysis</h4>
              <p>
                We lower the barrier to entry with intuitive peer comparison tools. 
                Instantly analyze relative strength and valuation gaps to discover hidden opportunities without the complexity.
              </p>
            </div>

            {/* 4. 지속적 진화와 성장 */}
            <div className="main-feature-card">
              <h4>Evolving Growth Engine</h4>
              <p>
                Your portfolio deserves a platform that grows with it. We continuously upgrade our algorithms 
                and features to support your sustainable wealth generation in a changing market.
              </p>
            </div>

          </div>
        </section>

        {/* --- Section 5: CTA --- */}
        <section className="main-section main-cta-section" id="section-cta">
          <h2>Start Your Journey</h2>
          <p>
            Join thousands of investors who are taking control of their financial future. 
            Sign up today and get access to premium features.
          </p>
          <Link to="/analysis" className="main-cta-button">
            Go to Analysis
          </Link>
        </section>
        
        {/* --- Footer Section --- */}
        <section className="main-section main-footer-section" id="section-footer">
          <div 
            className="main-footer-bg" 
            style={{ 
              backgroundImage: `url(${mountainImg})`,
              opacity: isFooterVisible ? 1 : 0 
            }}
          ></div>
          
          <div className="main-footer-content">
            <p>© 2025 Stock Analysis Platform. All rights reserved.</p>
            <p>Investment involves risk. Past performance is not indicative of future results.</p>
            <div className="main-footer-links">
              <Link to="/contact">Contact Support</Link>
              <Link to="/analysis">Market Analysis</Link>
              <Link to="/stock">Ticker Search</Link>
            </div>
          </div>
        </section>

      </main>
      
      <div className="main-progress-bar-container">
        <div className="main-progress-bar" ref={progressBarRef}></div>
      </div>
    </>
  );
};

export default Main;