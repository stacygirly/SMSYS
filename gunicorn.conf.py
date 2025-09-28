# gunicorn.conf.py
import os

wsgi_app = "testbackend:app"   # or "backend:app"
workers = 1                    # keep 1 for TF/librosa memory
threads = 1
timeout = 120
graceful_timeout = 120
keepalive = 5
loglevel = "info"

# Bind to Render’s assigned port
bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"
