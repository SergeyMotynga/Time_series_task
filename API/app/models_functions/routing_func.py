from API.app.models_functions.sarima_processing_manual_func import sarima_processing_manual
from API.app.models_functions.sarima_processing_auto_func import sarima_processing_auto
from API.app.models_functions.ets_processing_manual_func import ets_processing_manual
from API.app.models_functions.ets_processing_auto_func import ets_processing_auto
from API.app.models_functions.prophet_processing_manual_func import prophet_processing_manual
from API.app.models_functions.prophet_processing_auto_func import prophet_processing_auto
from API.app.models_functions.linear_regression_func import linear_regression_processing
from API.app.models_functions.random_forest_func import random_forest_processing
from API.app.models_functions.gradient_boosting_func import gradient_boosting_processing
from API.app.models_functions.elastic_net_func import elastic_net_processing
from API.app.models_functions.lightgbm_func import lightgbm_processing
from API.app.schemas import ModelRequest, RegressionRequest

routing_map={
    "sarima":{
        True: sarima_processing_auto,
        False: sarima_processing_manual
        },
    "ets":{
        True: ets_processing_auto,
        False: ets_processing_manual
        },
    "prophet":{
        True: prophet_processing_auto,
        False: prophet_processing_manual
        }
}

# Маппинг для регрессионных моделей (не требуют auto_params)
regression_routing_map = {
    "linear_regression": linear_regression_processing,
    "random_forest": random_forest_processing,
    "gradient_boosting": gradient_boosting_processing,
    "elastic_net": elastic_net_processing,
    "lightgbm": lightgbm_processing
}

def routing_func(model_type, request: ModelRequest) -> dict:
    request_json = request.dict()
    result = routing_map[model_type][request.auto_params](request_json)
    return result  # Возвращаем результат


def regression_routing_func(model_type, request: RegressionRequest) -> dict:
    """Роутинг для регрессионных моделей"""
    request_json = request.dict()
    result = regression_routing_map[model_type](request_json)
    return result