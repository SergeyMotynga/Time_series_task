from pydantic import BaseModel
from typing import Optional

class ModelRequest(BaseModel):
    auto_params: bool
    horizon: int # Дальность предсказания (количество строк)
    hyper_params: str  # JSON строка c параметрами модели в виде словаря
    df_train: str # JSON строка с фоеймом на основе которого будет сделано предсказание
    exog_vars: Optional[str] = None  # JSON строка с дополнительными признаками (экзогенные переменные)



class MetricsRequest(BaseModel):
    df_predict: str  # JSON строкас фреймом предсказания
    df_test: str  # JSON строка с фоеймом на котором будет поверяться предсказание