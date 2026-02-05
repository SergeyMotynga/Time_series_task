"""
Вспомогательная функция для автоподбора гиперпараметров через GridSearchCV
"""
from sklearn.model_selection import GridSearchCV
import json
import logging
import os

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def auto_tune_model(model, param_grid, X_train, y_train, cv=3):
    """
    Автоподбор гиперпараметров через GridSearchCV

    Args:
        model: Экземпляр модели (не обученный)
        param_grid: Словарь или список значений параметров для поиска
        X_train: Признаки для обучения
        y_train: Целевая переменная
        cv: Количество фолдов для кросс-валидации

    Returns:
        Лучшая модель (уже обученная) и словарь с лучшими параметрами
    """
    # Преобразуем param_grid если он пришел как JSON строка
    if isinstance(param_grid, str):
        param_grid = json.loads(param_grid)

    # Определяем количество параллельных процессов
    # На Railway ограничиваем до 2, локально используем все ядра
    is_railway = os.environ.get('RAILWAY_ENVIRONMENT') is not None
    n_jobs = 2 if is_railway else -1

    logger.info(f"Starting GridSearchCV (cv={cv}, n_jobs={n_jobs}, data_shape={X_train.shape})")

    # GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring='r2',  # Используем R² для регрессии
        n_jobs=n_jobs,
        verbose=0  # Отключаем verbose чтобы не засорять логи
    )

    # Обучение
    grid_search.fit(X_train, y_train)
    logger.info(f"GridSearchCV done: score={grid_search.best_score_:.4f}")

    # Возвращаем лучшую модель и параметры
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    return best_model, best_params, best_score
