from prophet import Prophet
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from itertools import product
from API.app.models_functions.make_prediction_dataframe_func import make_prediction_dataframe
import json
def prophet_processing_auto(params):

    df_train = pd.read_json(params["df_train"], orient='table')

    train_df = pd.DataFrame({
        'ds': df_train.index,
        'y': df_train["sensor"].values
    })

    # Обрабатываем экзогенные переменные
    exog_columns = []
    if params.get("exog_vars"):
        df_exog = pd.read_json(params["exog_vars"], orient='table')
        exog_columns = df_exog.columns.tolist()

        # Добавляем экзогенные переменные в train_df
        for col in exog_columns:
            train_df[col] = df_exog[col].values
    
    param_grid = {
        'growth': ['linear'],
        'seasonality_mode': ['additive', 'multiplicative'],
        'changepoint_prior_scale': [ 0.1, 0.5], #0.01,
        'seasonality_prior_scale': [0.1, 1.0], #, 10.0
        'yearly_seasonality': [True, False],
        'weekly_seasonality': [True, False]
    }
 
    validation_size = 0.2
    n_folds =  3
    
    best_score = np.inf
    best_params = {}
    best_model = None
    
    for fold in range(n_folds):

        split_idx = int(len(train_df) * (1 - validation_size))
        train = train_df.iloc[:split_idx]
        valid = train_df.iloc[split_idx:]
        
        train_df = train_df.iloc[split_idx//2:].reset_index(drop=True)
        
        for params_comb in product(*param_grid.values()):
            current_params = dict(zip(param_grid.keys(), params_comb))
            
            try:
                # Создание и обучение модели
                model = Prophet(**current_params)

                # Добавляем регрессоры
                for col in exog_columns:
                    model.add_regressor(col)

                model.fit(train)
                
                # Прогноз на валидационном наборе
                future = model.make_future_dataframe(
                    periods=len(valid), 
                    freq=pd.infer_freq(train['ds']))
                forecast = model.predict(future)
                
                # Оценка качества
                val_predictions = forecast.tail(len(valid))['yhat'].values
                score = mean_squared_error(valid['y'].values, val_predictions)
                
                # Сохранение лучшей модели
                if score < best_score:
                    best_score = score
                    best_params = current_params
                    best_model = model

            except Exception:
                pass
    
    # Обучение лучшей модели на всех данных
    final_model = Prophet(**best_params)

    # Добавляем регрессоры
    for col in exog_columns:
        final_model.add_regressor(col)

    final_model.fit(train_df)

    # Прогноз на тестовом наборе
    future = final_model.make_future_dataframe(
        periods=params["horizon"],
        freq=pd.infer_freq(df_train.index))

    # Добавляем экзогенные переменные в future (используем последнее значение)
    if exog_columns:
        for col in exog_columns:
            last_value = df_exog[col].iloc[-1]
            future[col] = last_value

    forecast = final_model.predict(future)
    predictions = forecast.tail(params["horizon"])['yhat']
    
    model_params = {
    "growth": model.growth,
    "changepoints": model.changepoints.tolist(),  # datetime → список строк
    "n_changepoints": model.n_changepoints,
    "seasonality_mode": model.seasonality_mode,
    "seasonalities": model.seasonalities,  # сезонности (годовая, недельная)
    "params": {  # внутренние параметры (тренд, сезонности, шумы)
        "k": model.params["k"][0].tolist(),  # коэффициент тренда
        "m": model.params["m"][0].tolist(),  # смещение тренда
        "sigma_obs": model.params["sigma_obs"][0].tolist(),  # шум данных
        "beta": model.params["beta"][0].tolist(),  # коэффициенты сезонности
    },
    "holidays": model.holidays.to_dict(orient="records") if model.holidays is not None else None,
    }
    # Конвертируем в JSON (с обработкой datetime)
    model_params = json.dumps(model_params, indent=4, default=str)

    return {
        "predictions": make_prediction_dataframe(df_train,predictions.values,params["horizon"]),
        "model_params": model_params
    }