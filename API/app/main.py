from fastapi import FastAPI, Path
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
import json
import traceback
import math
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pydantic import Field
from .models_functions.routing_func import routing_func, regression_routing_func
from .metrics_functions.metrics_func import calculate_metrics
from .schemas import ModelRequest, MetricsRequest, RegressionRequest

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


app = FastAPI()

# ThreadPoolExecutor для выполнения долгих операций
executor = ThreadPoolExecutor(max_workers=4)

# CORS middleware для доступа из Streamlit Cloud
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#Функция обработки запроса получения предсказания
@app.post("/api/models/{model_type}")
async def process_model(
    request: ModelRequest,
    model_type: str = Path(..., description="Тип модели для прогнозирования")
    ):

    predict = routing_func(model_type, request)
    predict_params = predict["model_params"]
    df_predict = predict["predictions"]

    response = {
        "hyper_params": predict_params,
        "df_predict": df_predict.to_json(orient='table', date_format='iso'),
    }

    return response

#Функция обработки запроса получения метрик
@app.post("/api/metrics")
async def process_metrics(request: MetricsRequest):

    def clean_metrics(metrics):
        if metrics is None:
            return None
        return {k: (None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else v) for k, v in metrics.items()}

    try:
        df_predict = pd.read_json(request.df_predict, orient='table')
        df_test = pd.read_json(request.df_test, orient='table')

        if len(df_test) > 0:
            metrics = calculate_metrics(real_data=df_test, predicted_data=df_predict)
        else:
            metrics = None

    except Exception as e:
        logger.error(f'Ошибка при расчёте метрик: {e}')
        return {
            'error': str(e)
        }

    response = {
        "metrics": clean_metrics(metrics)
    }

    return response

#Функция обработки запроса для регрессионных моделей
@app.post("/api/regression/{model_type}")
async def process_regression(
    request: RegressionRequest,
    model_type: str = Path(..., description="Тип регрессионной модели")
    ):

    try:
        # Выполняем долгую операцию в отдельном потоке
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            regression_routing_func,
            model_type,
            request
        )

        predict_params = result["model_params"]
        df_predictions = result["predictions"]

        response = {
            "model_params": predict_params,
            "df_predictions": df_predictions.to_json(orient='table', date_format='iso'),
        }

        return response

    except Exception as e:
        error_msg = str(e)[:200]  # Обрезаем ошибку до 200 символов
        logger.error(f'Error in {model_type}: {type(e).__name__} - {error_msg}')
        logger.error(f'Traceback: {traceback.format_exc()[:500]}')  # Только первые 500 символов
        return {
            'error': f'{type(e).__name__}: {error_msg}'
        }

#Функция обработки запроса получения списка предсказаний
@app.get("/api/models")
async def get_models():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'config.json')

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            models = config['models']

    except Exception as e:
        return {
            'error': e
        }
    response = {
        "models": models
    }

    return response
