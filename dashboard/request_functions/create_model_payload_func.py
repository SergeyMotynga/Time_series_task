import pandas as pd
import json
from typing import Optional


#Функция для создания payload для обращения к API моделей предсказания
def create_model_payload(auto_params: bool,
                         horizon: int,
                         df_train: pd.DataFrame,
                         hyper_params: dict = {},
                         exog_vars: Optional[pd.DataFrame] = None):
    json_df_train = df_train.to_json(orient='table', date_format='iso')

    payload = {
        'auto_params': auto_params,
        "horizon": horizon,
        "hyper_params": json.dumps(hyper_params),
        "df_train": json_df_train
    }

    # Добавляем экзогенные переменные, если они указаны
    if exog_vars is not None and not exog_vars.empty:
        payload["exog_vars"] = exog_vars.to_json(orient='table', date_format='iso')

    return payload

