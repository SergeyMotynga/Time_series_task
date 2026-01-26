from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import json
import numpy as np
from API.app.models_functions.make_prediction_dataframe_func import make_prediction_dataframe

def sarima_processing_manual(params):
    """
    params:
        - S - сезонность
        - p - порядок авторегрессии (число используемых предыдущих значений ряда)
        - d - порядок дифферненцирования ряда
        - q - порядок скользящего среднего (число используемых предыдущих ошибок)
        - P - порядок сезонной авторегрессии
        - D - порядок сезонного дифференциорования
        - Q - порядок сезонного скользящего среднего
    """
    df_train = pd.read_json(params["df_train"], orient='table')

    # Извлекаем одномерный временной ряд из колонки 'sensor'
    y = df_train["sensor"].values

    # Извлекаем экзогенные переменные, если они есть
    exog_train = None
    exog_forecast = None
    if params.get("exog_vars"):
        df_exog = pd.read_json(params["exog_vars"], orient='table')
        exog_train = df_exog.values

        # Для прогноза используем простую экстраполяцию (последнее значение)
        forecast_steps = params["horizon"]
        last_values = df_exog.iloc[-1].values
        exog_forecast = np.tile(last_values, (forecast_steps, 1))

    hyper_params = json.loads(params["hyper_params"])
    if  hyper_params.get("S", False):
        model = SARIMAX(
            y,
            exog=exog_train,
            order=(hyper_params["p"], hyper_params["d"], hyper_params["q"]),
            seasonal_order=(hyper_params["P"], hyper_params["D"], hyper_params["Q"], hyper_params["S"])
        ).fit(disp=-1)
    else:
        model = SARIMAX(
            y,
            exog=exog_train,
            order=(hyper_params["p"], hyper_params["d"], hyper_params["q"]),
        ).fit(disp=-1)

    forecast_steps = params["horizon"]
    if exog_forecast is not None:
        predictions = model.get_forecast(steps=forecast_steps, exog=exog_forecast).predicted_mean
    else:
        predictions = model.get_forecast(steps=forecast_steps).predicted_mean

    # Конвертируем параметры в сериализуемый формат
    model_params = {
        'hyper_params': hyper_params,
        'params': model.params.tolist() if hasattr(model.params, 'tolist') else list(model.params)
    }

    return {
        "predictions": make_prediction_dataframe(df_train, predictions, forecast_steps),
        "model_params": model_params,
    }