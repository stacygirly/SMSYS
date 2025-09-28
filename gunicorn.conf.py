wsgi_app = "app:app"
workers = 1           # keep memory stable on Render free tier
threads = 1
timeout = 120
graceful_timeout = 120
keepalive = 5
loglevel = "info"
