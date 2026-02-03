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


class RegressionRequest(BaseModel):
    df_train: str  # JSON строка с DataFrame (признаки + целевая переменная)
    target_col: str  # Название колонки с целевой переменной
    feature_cols: Optional[str] = None  # JSON список признаков (если None, то все кроме target)
    hyper_params: str  # JSON строка с параметрами модели (или диапазонами для GridSearch)
    test_size: float = 0.2  # Размер тестовой выборки (0.0-1.0)
    use_auto_tune: bool = False  # Использовать GridSearchCV для автоподбора параметров