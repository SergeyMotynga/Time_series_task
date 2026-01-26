from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import json
import numpy as np
from API.app.models_functions.make_prediction_dataframe_func import make_prediction_dataframe

def sarima_processing_auto(params):
    """
    - params:
        S - сезонность
    """

    df_train = pd.read_json(params["df_train"], orient='table')

    hyper_params = json.loads(params["hyper_params"])
    season=hyper_params.get("S",0)

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

    # ШАГ 1: auto_arima подбирает оптимальную структуру БЕЗ экзогенных переменных
    # (известное ограничение pmdarima - не учитывает exogenous при автоподборе)
    auto_model = auto_arima(
        y,
        exogenous=None,  # Намеренно не передаем exog на этапе подбора
        m=season,
        trace=False,
        stepwise=True,
        suppress_warnings=True,
        seasonal=True
    )

    best_order = auto_model.order
    best_seasonal_order = auto_model.seasonal_order

    # ШАГ 2: Переобучаем SARIMAX с найденной структурой, но С экзогенными переменными
    if exog_train is not None:
        model = SARIMAX(
            y,
            exog=exog_train,
            order=best_order,
            seasonal_order=best_seasonal_order
        ).fit(disp=-1)
    else:
        model = auto_model

    forecast_steps = params["horizon"]

    # Прогнозирование зависит от типа модели
    if exog_forecast is not None:
        # SARIMAX использует get_forecast
        predictions = model.get_forecast(steps=forecast_steps, exog=exog_forecast).predicted_mean
    else:
        # auto_arima использует predict
        predictions = model.predict(n_periods=forecast_steps)

    # Конвертируем параметры в сериализуемый формат
    if exog_train is not None:
        # Для SARIMAX
        model_params = {
            'order': best_order,
            'seasonal_order': best_seasonal_order,
            'params': model.params.tolist() if hasattr(model.params, 'tolist') else list(model.params),
            'n_exog_vars': exog_train.shape[1],
            'aic': model.aic
        }
    else:
        # Для auto_arima
        model_params = {
            'order': model.order,
            'seasonal_order': model.seasonal_order,
            'params': model.params().tolist() if hasattr(model.params(), 'tolist') else list(model.params()),
            'n_exog_vars': 0
        }


    return {
        "predictions": make_prediction_dataframe(df_train,predictions,forecast_steps),

        "model_params": model_params,
    }