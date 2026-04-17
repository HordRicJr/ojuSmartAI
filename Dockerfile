# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
FROM python:3.11-slim

# Libs systeme pour OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Variables d'environnement (valeurs de .env injectees ici)
ENV APP_NAME="OjuSmart Stateless AI Engine"
ENV APP_VERSION="1.0.0"
ENV DEBUG=false
ENV EMBEDDING_DIMENSION=512
ENV IMAGE_TARGET_SIZE=224

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
