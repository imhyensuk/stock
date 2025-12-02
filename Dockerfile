# 1. Multi-stage build: React 프론트 빌드 (루트 package.json 기준)
FROM node:20-bookworm AS builder

WORKDIR /app

# 프론트엔드 의존성 설치
COPY package.json package-lock.json* ./
RUN npm install

# 전체 소스 복사 후 빌드
COPY . .
RUN npm run build

# 2. Production 런타임 (Node.js + Python)
FROM node:20-bookworm-slim

# Python 설치 (stock.py용)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python-is-python3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python venv
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app

# --- 백엔드 Node 의존성 설치 ---
# backend 디렉터리의 package.json을 기준으로 설치
COPY backend/package.json backend/package-lock.json* ./backend/
WORKDIR /app/backend
RUN npm install --omit=dev

# --- Python 백엔드 의존성 설치 ---
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- 빌드된 프론트와 백엔드 소스 복사 ---
WORKDIR /app
COPY --from=builder /app/build ./build
COPY backend ./backend

# 포트 설정 (server.js는 NODEPORT 사용)
ENV PORT=8000
ENV NODEPORT=8000
EXPOSE 8000

WORKDIR /app/backend
CMD ["node", "server.js"]
