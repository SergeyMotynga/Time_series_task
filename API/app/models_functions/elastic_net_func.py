"""
Функция обработки для модели Elastic Net
"""
import pandas as pd
from io import StringIO
import json
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from .auto_tune_helper import auto_tune_model


def elastic_net_processing(params):
    """
    Обработка запроса для Elastic Net

    Args:
        params: словарь параметров

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

    # Нормализация данных (важно для Elastic Net)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Парсинг гиперпараметров
    hyper_params = json.loads(params["hyper_params"])
    use_auto_tune = params.get("use_auto_tune", False)

    # Обучение модели
    if use_auto_tune:
        # Автоподбор параметров через GridSearchCV
        base_model = ElasticNet(random_state=42)
        model, best_params, best_score = auto_tune_model(
            base_model, hyper_params, X_train_scaled, y_train, cv=3
        )
        used_params = best_params
        used_params['cv_score'] = best_score
    else:
        # Ручная настройка параметров
        model = ElasticNet(
            alpha=hyper_params.get("alpha", 1.0),
            l1_ratio=hyper_params.get("l1_ratio", 0.5),
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        used_params = {
            'alpha': hyper_params.get("alpha", 1.0),
            'l1_ratio': hyper_params.get("l1_ratio", 0.5)
        }

    # Предсказания
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    # Метрики
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_rmse = np.sqrt(test_mse)

    # Коэффициенты
    coefficients = {feature: float(coef) for feature, coef in zip(feature_cols, model.coef_)}
    intercept = float(model.intercept_)

    # Формула модели
    formula = f"{target_col} = {intercept:.4f}"
    for feature, coef in coefficients.items():
        sign = "+" if coef >= 0 else "-"
        formula += f" {sign} {abs(coef):.4f} * {feature}"

    # Feature importance (абсолютные значения коэффициентов)
    feature_importance = {k: abs(v) for k, v in coefficients.items()}
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    # Формирование результата
    df_predictions = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred_test
    }, index=y_test.index)

    model_params = {
        'model_type': 'Elastic Net',
        'auto_tuned': use_auto_tune,
        'parameters': used_params,
        'formula': formula,
        'coefficients': coefficients,
        'intercept': intercept,
        'alpha': used_params.get("alpha", 1.0),
        'l1_ratio': used_params.get("l1_ratio", 0.5),
        'metrics': {
            'train_r2': float(train_r2),
            'test_r2': float(test_r2),
            'test_mae': float(test_mae),
            'test_mse': float(test_mse),
            'test_rmse': float(test_rmse)
        },
        'feature_importance': feature_importance
    }

    # Добавляем CV score если был автоподбор
    if use_auto_tune and 'cv_score' in used_params:
        model_params['metrics']['cv_r2'] = float(used_params['cv_score'])

    return {
        "predictions": df_predictions,
        "model_params": model_params
    }
