# 1. Multi-stage build: React 프론트 빌드 (루트 package.json 기준)
FROM node:20-bookworm AS builder

WORKDIR /app

# 프론트엔드 의존성 설치
COPY package.json package-lock.json* ./
RUN npm install

# 전체 소스 복사 후 빌드
COPY . .
RUN npm run build

# 2. Production 런타임 (Node.js + Python, 최소화)
FROM node:20-bookworm-slim

# Python 설치 (stock.py용) - 최소 패키지만 설치
RUN apt-get update && apt-get install -y \
    python3-full \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 캐시 최적화를 위해 한 번에 설치
WORKDIR /app/backend

# backend 의존성 먼저 복사 및 설치 (--break-system-packages 추가)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Node 의존성 설치
COPY backend/package.json backend/package-lock.json* ./
RUN npm install --omit=dev

# --- 빌드된 프론트와 백엔드 소스 복사 ---
WORKDIR /app
COPY --from=builder /app/build ./build
COPY backend ./backend

# 포트 설정
ENV PORT=8000
ENV NODEPORT=8000
EXPOSE 8000

WORKDIR /app/backend
CMD ["node", "server.js"]
