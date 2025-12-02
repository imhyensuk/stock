백엔드 실행: /Users/imhyeonseok/Documents/stock/web/backend/ 터미널에서 다음을 실행합니다.

Bash

node server.js
(터미널에 Backend server running at http://localhost:8000가 표시되어야 합니다.)

프론트엔드 실행: /Users/imhyeonseok/Documents/stock/web/ 터미널에서 다음을 실행합니다.

Bash

npm start
브라우저에서 /stock 페이지로 이동하면, 기본값인 'Dow Jones' 데이터가 로드되며, 왼쪽 사이드바에서 다른 주식을 클릭하면 해당 주식의 차트, 뉴스, 트렌드가 실시간으로 로드됩니다.



/usr/local/bin/python3 -m venv venv

source venv/bin/activate

pip install yfinance pandas