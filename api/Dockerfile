FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    TOMATOGUARD_MODEL_PATH=/app/model/tomatoes.h5 \
    TOMATOGUARD_METADATA_PATH=/app/model/metadata.json

WORKDIR /app

RUN groupadd --system app && useradd --system --gid app --create-home app
COPY api/requirements.txt /app/api/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /app/api/requirements.txt

COPY api /app/api
COPY model/tomatoes.h5 /app/model/tomatoes.h5
COPY model/metadata.json /app/model/metadata.json

RUN chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.getenv('PORT','8000')+'/health/live', timeout=3)"

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
