"""
Функция обработки для модели LightGBM
"""
import pandas as pd
from io import StringIO
import json
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from .auto_tune_helper import auto_tune_model


def lightgbm_processing(params):
    """
    Обработка запроса для LightGBM

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
        X, y, test_size=test_size, random_state=42
    )

    # Парсинг гиперпараметров
    hyper_params = json.loads(params["hyper_params"])
    use_auto_tune = params.get("use_auto_tune", False)

    # Обучение модели
    if use_auto_tune:
        # Автоподбор параметров через GridSearchCV
        base_model = LGBMRegressor(random_state=42, verbose=-1)
        model, best_params, best_score = auto_tune_model(
            base_model, hyper_params, X_train, y_train, cv=3
        )
        used_params = best_params
        used_params['cv_score'] = best_score
    else:
        # Ручная настройка параметров
        model = LGBMRegressor(
            n_estimators=hyper_params.get("n_estimators", 100),
            learning_rate=hyper_params.get("learning_rate", 0.1),
            max_depth=hyper_params.get("max_depth", -1),
            num_leaves=hyper_params.get("num_leaves", 31),
            random_state=42,
            verbose=-1  # Отключаем вывод
        )
        model.fit(X_train, y_train)
        used_params = {
            'n_estimators': hyper_params.get("n_estimators", 100),
            'learning_rate': hyper_params.get("learning_rate", 0.1),
            'max_depth': hyper_params.get("max_depth", -1),
            'num_leaves': hyper_params.get("num_leaves", 31)
        }

    # Предсказания
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Метрики
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_rmse = np.sqrt(test_mse)

    # Feature importance
    feature_importance = {
        feature: float(importance)
        for feature, importance in zip(feature_cols, model.feature_importances_)
    }
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    # Формирование результата
    df_predictions = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred_test
    }, index=y_test.index)

    model_params = {
        'model_type': 'LightGBM',
        'auto_tuned': use_auto_tune,
        'parameters': used_params,
        'n_estimators': used_params.get("n_estimators", 100),
        'learning_rate': used_params.get("learning_rate", 0.1),
        'max_depth': used_params.get("max_depth", -1),
        'num_leaves': used_params.get("num_leaves", 31),
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
