"""
Функция обработки для модели Linear Regression
"""
import pandas as pd
import json
import numpy as np
import logging
from io import StringIO
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from .auto_tune_helper import auto_tune_model

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def linear_regression_processing(params):
    """
    Обработка запроса для линейной регрессии

    Args:
        params: словарь параметров:
            - df_train: JSON строка с DataFrame
            - target_col: название целевой переменной
            - feature_cols: JSON список признаков (опционально)
            - hyper_params: JSON строка с параметрами модели
            - test_size: размер тестовой выборки

    Returns:
        dict с predictions и model_params
    """
    # Загрузка данных
    df = pd.read_json(StringIO(params["df_train"]), orient='table')
    target_col = params["target_col"]

    # Определение признаков
    if params.get("feature_cols"):
        feature_cols = json.loads(params["feature_cols"])
    else:
        feature_cols = [col for col in df.columns if col != target_col]

    # Подготовка данных
    X = df[feature_cols]
    y = df[target_col]

    # Разделение на train/test
    test_size = params.get("test_size", 0.2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    # Парсинг гиперпараметров
    hyper_params = json.loads(params["hyper_params"])
    use_auto_tune = params.get("use_auto_tune", False)

    # Обучение модели
    if use_auto_tune:
        # Автоподбор параметров через GridSearchCV
        base_model = LinearRegression()
        model, best_params, best_score = auto_tune_model(
            base_model, hyper_params, X_train, y_train, cv=3
        )
        used_params = best_params
        used_params['cv_score'] = best_score
    else:
        # Ручная настройка параметров
        model = LinearRegression(
            fit_intercept=hyper_params.get("fit_intercept", True)
        )
        model.fit(X_train, y_train)
        used_params = {'fit_intercept': hyper_params.get("fit_intercept", True)}

    # Предсказания
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Метрики
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_rmse = np.sqrt(test_mse)

    # Формула модели
    intercept = float(model.intercept_)
    coefficients = {feature: float(coef) for feature, coef in zip(feature_cols, model.coef_)}

    formula = f"{target_col} = {intercept:.4f}"
    for feature, coef in coefficients.items():
        sign = "+" if coef >= 0 else "-"
        formula += f" {sign} {abs(coef):.4f} * {feature}"

    # Формирование результата
    df_predictions = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred_test
    }, index=y_test.index)

    model_params = {
        'model_type': 'Linear Regression',
        'auto_tuned': use_auto_tune,
        'parameters': used_params,
        'formula': formula,
        'coefficients': coefficients,
        'intercept': intercept,
        'metrics': {
            'train_r2': float(train_r2),
            'test_r2': float(test_r2),
            'test_mae': float(test_mae),
            'test_mse': float(test_mse),
            'test_rmse': float(test_rmse)
        },
        'feature_importance': {k: abs(v) for k, v in sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)}
    }

    # Добавляем CV score если был автоподбор
    if use_auto_tune and 'cv_score' in used_params:
        model_params['metrics']['cv_r2'] = float(used_params['cv_score'])

    return {
        "predictions": df_predictions,
        "model_params": model_params
    }
