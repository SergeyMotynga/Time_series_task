"""
Вспомогательная функция для автоподбора гиперпараметров через GridSearchCV
"""
from sklearn.model_selection import GridSearchCV
import json


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

    # GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring='r2',  # Используем R² для регрессии
        n_jobs=-1,  # Используем все ядра
        verbose=0
    )

    # Обучение
    grid_search.fit(X_train, y_train)

    # Возвращаем лучшую модель и параметры
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    return best_model, best_params, best_score
