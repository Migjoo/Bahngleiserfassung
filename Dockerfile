# ---- Base image mit Python 3.11
FROM python:3.11-slim

# Environment-Variablen
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/user

# Non-root-User anlegen (vermeidet Rechtefehler)
RUN useradd -m -u 1000 user

WORKDIR /app

# Dependencies zuerst installieren (Docker-Layer-Cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App-Code kopieren
COPY . .

# Streamlit-Config anlegen und Telemetrie ausschalten
RUN mkdir -p $HOME/.streamlit && \
    printf "[server]\nheadless = true\nport = 7860\naddress = \"0.0.0.0\"\n\n[browser]\ngatherUsageStats = false\n" > $HOME/.streamlit/config.toml && \
    chown -R user:user $HOME

USER user

# Start-Befehl
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
