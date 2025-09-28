# gunicorn.conf.py
import os

wsgi_app = "testbackend:app"   # or "testbackend:app" if that's your file
workers = 1                 # keep 1 to avoid TF/librosa OOM on small instances
threads = 1
timeout = 120
graceful_timeout = 120
keepalive = 5
loglevel = "info"

# Bind to the Render-assigned port
bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"

