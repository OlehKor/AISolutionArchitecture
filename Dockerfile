FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/opt/fastembed

RUN useradd --create-home --uid 1000 app
WORKDIR /app

# Editable install keeps data/ and var/ next to the code, which is where the settings look for them.
COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
RUN pip install -e . \
    && mkdir -p /opt/fastembed /app/var /data \
    && python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')" \
    && chown -R app:app /opt/fastembed /app/var /data

USER app
ENTRYPOINT ["parking-assistant"]
CMD ["--help"]
