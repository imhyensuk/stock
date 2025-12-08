FROM node:20-bookworm-slim

# Python 설치 (stock.py 실행용)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 가상환경 설정
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app/backend

# Node 의존성 설치
COPY package*.json ./
RUN npm install --omit=dev

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 백엔드 소스 복사
COPY . .

# 환경 변수 설정
ENV PORT=8080
ENV NODE_PORT=8080

EXPOSE 8080

CMD ["node", "server.js"]
