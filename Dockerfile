ARG REPO=https://github.com/harryfrzz/neo.git
ARG BRANCH=main

# ── Stage 1: clone ────────────────────────────────────────────────────────────
FROM node:20-alpine AS cloner
ARG REPO
ARG BRANCH
RUN apk add --no-cache git
WORKDIR /app
RUN git clone --depth=1 --branch "$BRANCH" "$REPO" .

# ── Stage 2: install deps ─────────────────────────────────────────────────────
FROM node:20-alpine AS deps
WORKDIR /app
COPY --from=cloner /app/package.json /app/package-lock.json ./
RUN npm ci

# ── Stage 3: build ────────────────────────────────────────────────────────────
FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=cloner /app ./
COPY --from=deps /app/node_modules ./node_modules
# NEXT_PUBLIC_API_BASE is baked in at build time — passed via docker-compose build args
ARG NEXT_PUBLIC_API_BASE=http://localhost:8000
ENV NEXT_PUBLIC_API_BASE=$NEXT_PUBLIC_API_BASE
RUN npm run build

# ── Stage 4: minimal runtime ──────────────────────────────────────────────────
FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production

COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

EXPOSE 3000
CMD ["node", "server.js"]
