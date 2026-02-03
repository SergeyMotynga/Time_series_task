FROM python:3.12-slim

WORKDIR /code

# Устанавливаем системные зависимости для LightGBM и XGBoost
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements и устанавливаем зависимости
COPY API/requirements.txt ./API/
RUN pip install --no-cache-dir -r API/requirements.txt

# Копируем приложение
COPY API/app ./API/app/

EXPOSE $PORT

# Запускаем из корня проекта
CMD uvicorn API.app.main:app --host 0.0.0.0 --port $PORT
