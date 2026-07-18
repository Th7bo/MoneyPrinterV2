# syntax=docker/dockerfile:1
# MoneyPrinterV2 Dashboard — single-image build.
# Add this file, dashboard_server.py, and docker-compose.yml to a fork of
# FujiwaraChoki/MoneyPrinterV2, then deploy the fork with Dokploy.
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    DISPLAY=:99 PORT=8080 ENABLE_VNC=false OLLAMA_BASE_URL=http://127.0.0.1:11434

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg imagemagick libsndfile1 fonts-liberation \
      firefox-esr xvfb x11