# 1. Multi-stage build: React 빌드
FROM node:20-bookworm AS builder

WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install

COPY . .
RUN npm run build

# 2. Production 런타임 (Node.js + Python)
FROM node:20-bookworm-slim

# Python 설치 (stock.py 실행용)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python-is-python3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 가상환경 생성
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app

# Node.js 의존성 설치 (production만)
COPY package.json package-lock.json* ./
RUN npm ci --only=production --no-optional

# Python 백엔드 의존성 설치
COPY backend/requirements.txt ./backend/
WORKDIR /app/backend
RUN pip install --no-cache-dir -r requirements.txt

# React 빌드파일 + 전체 소스 복사
WORKDIR /app
COPY --from=builder /app/build ./build
COPY backend ./backend

ENV PORT=8080
EXPOSE $PORT

WORKDIR /app/backend
# server.js가 메인 백엔드 (stock.py 호출)
CMD ["node", "server.js"]
