# syntax=docker/dockerfile:1
# MoneyPrinterV2 Dashboard — single-image build.
# Add this file, dashboard_server.py, and docker-compose.yml to a fork of
# FujiwaraChoki/MoneyPrinterV2, then deploy the fork with Dokploy.
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    DISPLAY=:99 PORT=8080 ENABLE_VNC=false OLLAMA_BASE_URL=http://127.0.0.1:11434

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg imagemagick libsndfile1 fonts-liberation \
      firefox-esr xvfb x11vnc novnc websockify fluxbox xterm \
      golang-go git wget curl unzip ca-certificates supervisor procps \
    && rm -rf /var/lib/apt/lists/*

# Let MoviePy/ImageMagick draw text captions (Debian blocks this by default).
RUN sed -i 's/rights="none" pattern="@\*"/rights="read|write" pattern="@*"/' /etc/ImageMagick-6/policy.xml || true \
    && sed -i '/pattern="TEXT"/d;/pattern="LABEL"/d' /etc/ImageMagick-6/policy.xml || true

WORKDIR /app

# Upstream Python deps (cached on requirements.txt), then the dashboard deps.
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip wheel \
    && pip install -r requirements.txt \
    && pip install "fastapi>=0.110" "uvicorn[standard]>=0.29" "apscheduler>=3.10,<4" "SQLAlchemy>=2.0"

# Everything from the fork (upstream src/ + dashboard_server.py + config.example.json).
COPY . .

# Helper to open a logged-in Firefox profile from the VNC desktop:
#   mp-firefox youtube   -> profile at /profiles/youtube
RUN printf '#!/usr/bin/env bash\nname="${1:-default}"\nmkdir -p "/profiles/$name"\nexec firefox -no-remote -profile "/profiles/$name"\n' > /usr/local/bin/mp-firefox \
    && chmod +x /usr/local/bin/mp-firefox

# Process manager: virtual display, optional VNC stack, and the web app.
RUN cat > /app/supervisord.conf <<'EOF'
[supervisord]
nodaemon=true
user=root
logfile=/dev/stdout
logfile_maxbytes=0
pidfile=/tmp/supervisord.pid

[program:xvfb]
command=Xvfb :99 -screen 0 1360x1020x24 -ac
autorestart=true
priority=10
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
redirect_stderr=true

[program:fluxbox]
command=fluxbox
environment=DISPLAY=":99"
autostart=%(ENV_ENABLE_VNC)s
autorestart=true
priority=20
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
redirect_stderr=true

[program:x11vnc]
command=x11vnc -display :99 -forever -shared -nopw -rfbport 5900 -quiet
autostart=%(ENV_ENABLE_VNC)s
autorestart=true
priority=30
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
redirect_stderr=true

[program:novnc]
command=websockify --web=/usr/share/novnc 6080 localhost:5900
autostart=%(ENV_ENABLE_VNC)s
autorestart=true
priority=40
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
redirect_stderr=true

[program:app]
command=uvicorn dashboard_server:app --host 0.0.0.0 --port %(ENV_PORT)s
directory=/app
environment=DISPLAY=":99"
autorestart=true
priority=50
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
redirect_stderr=true
EOF

EXPOSE 8080 6080
CMD ["supervisord", "-c", "/app/supervisord.conf"]
