FROM python:3.12-slim

WORKDIR /code

# Копируем requirements и устанавливаем зависимости
COPY API/requirements.txt ./API/
RUN pip install --no-cache-dir -r API/requirements.txt

# Копируем приложение
COPY API/app ./API/app/

EXPOSE $PORT

# Запускаем из корня проекта
CMD uvicorn API.app.main:app --host 0.0.0.0 --port $PORT
