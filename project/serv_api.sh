#!/bin/bash

IMAGE_NAME="fraud_detection_xgboost"
PORT=9000

# Se a porta 9000 estiver ocupada, encontra e remove o container
if lsof -i :$PORT | grep LISTEN; then
    CONTAINER_ID=$(docker ps --filter "publish=$PORT" --format "{{.ID}}")
    if [ -n "$CONTAINER_ID" ]; then
        echo "🛑 Parando e removendo container na porta $PORT ($CONTAINER_ID)..."
        docker stop "$CONTAINER_ID"
        docker rm "$CONTAINER_ID"
    else
        echo "⚠️ Porta $PORT ocupada, mas não por um container Docker conhecido."
        exit 1
    fi
fi

# Testa se a imagem existe
if ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    echo "🔄 Imagem $IMAGE_NAME não encontrada. Executando ml.py para gerar a imagem..."
    python3 ml.py
fi

# Sobe o container
docker run -p 9000:8080 "$IMAGE_NAME"