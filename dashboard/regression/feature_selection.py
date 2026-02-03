"""
Модуль для отбора признаков (feature selection) для регрессионных моделей
Реализует методы: Correlation, SVS, RFR, VAR, AIC, BIC
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def select_by_correlation(df: pd.DataFrame,
                           target_col: str,
                           threshold: float = 0.3,
                           top_n: Optional[int] = None) -> Tuple[List[str], pd.Series]:
    """
    Отбор признаков по корреляции с целевой переменной

    Args:
        df: DataFrame с данными
        target_col: Целевая переменная
        threshold: Минимальный порог корреляции (по модулю)
        top_n: Количество лучших признаков (если None, используется threshold)

    Returns:
        Tuple[List[str], pd.Series]: Список выбранных признаков и их корреляции
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    # Вычисляем корреляцию всех признаков с целевой переменной
    correlations = df.corr()[target_col].drop(target_col).abs().sort_values(ascending=False)

    if top_n is not None:
        # Выбираем top_n лучших признаков
        selected_features = correlations.head(top_n).index.tolist()
        selected_corr = correlations.head(top_n)
    else:
        # Выбираем признаки с корреляцией выше порога
        selected_features = correlations[correlations >= threshold].index.tolist()
        selected_corr = correlations[correlations >= threshold]

    return selected_features, selected_corr


def select_by_svs(df: pd.DataFrame,
                   target_col: str,
                   n_features: int = 10,
                   kernel: str = 'rbf') -> Tuple[List[str], np.ndarray]:
    """
    Отбор признаков методом опорных векторов (Support Vector Selection)

    Args:
        df: DataFrame с данными
        target_col: Целевая переменная
        n_features: Количество признаков для выбора
        kernel: Тип ядра ('linear', 'rbf', 'poly')

    Returns:
        Tuple[List[str], np.ndarray]: Список выбранных признаков и их важность
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    # Подготовка данных
    X = df.drop(columns=[target_col]).dropna()
    y = df.loc[X.index, target_col]

    # Нормализация данных
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Обучаем SVR для получения важности признаков
    svr = SVR(kernel=kernel)
    svr.fit(X_scaled, y)

    # Для линейного ядра можно напрямую использовать коэффициенты
    if kernel == 'linear':
        feature_importance = np.abs(svr.coef_[0])
    else:
        # Для нелинейных ядер используем permutation importance
        from sklearn.inspection import permutation_importance
        perm_importance = permutation_importance(svr, X_scaled, y, n_repeats=10, random_state=42)
        feature_importance = perm_importance.importances_mean

    # Сортируем признаки по важности
    feature_indices = np.argsort(feature_importance)[::-1][:n_features]
    selected_features = X.columns[feature_indices].tolist()
    selected_importance = feature_importance[feature_indices]

    return selected_features, selected_importance


def select_by_rfr(df: pd.DataFrame,
                   target_col: str,
                   n_features: int = 10,
                   n_estimators: int = 100) -> Tuple[List[str], np.ndarray]:
    """
    Отбор признаков методом случайного леса (Random Forest Regressor)

    Args:
        df: DataFrame с данными
        target_col: Целевая переменная
        n_features: Количество признаков для выбора
        n_estimators: Количество деревьев в лесу

    Returns:
        Tuple[List[str], np.ndarray]: Список выбранных признаков и их важность
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    # Подготовка данных
    X = df.drop(columns=[target_col]).dropna()
    y = df.loc[X.index, target_col]

    # Обучаем Random Forest
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    # Получаем важность признаков
    feature_importance = rf.feature_importances_

    # Сортируем признаки по важности
    feature_indices = np.argsort(feature_importance)[::-1][:n_features]
    selected_features = X.columns[feature_indices].tolist()
    selected_importance = feature_importance[feature_indices]

    return selected_features, selected_importance


def select_by_var(df: pd.DataFrame,
                   target_col: str,
                   max_features: int = 10,
                   max_lags: int = 5) -> Tuple[List[str], Dict]:
    """
    Отбор признаков методом векторной авторегрессии (VAR)
    Использует дисперсию остатков как критерий отбора

    Args:
        df: DataFrame с данными
        target_col: Целевая переменная
        max_features: Максимальное количество признаков
        max_lags: Максимальное количество лагов для VAR

    Returns:
        Tuple[List[str], Dict]: Список выбранных признаков и статистика
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    try:
        from statsmodels.tsa.api import VAR
    except ImportError:
        raise ImportError("statsmodels is required for VAR method")

    # Подготовка данных
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols].dropna()
    y = df.loc[X.index, target_col]

    # Начинаем с целевой переменной
    selected_features = []
    remaining_features = feature_cols.copy()
    feature_scores = {}

    for _ in range(min(max_features, len(remaining_features))):
        best_feature = None
        best_score = np.inf

        for feature in remaining_features:
            # Создаем временный набор признаков
            temp_features = selected_features + [feature]
            temp_df = df[[target_col] + temp_features].dropna()

            if len(temp_df) < max_lags + 10:
                continue

            try:
                # Обучаем VAR модель
                model = VAR(temp_df)
                results = model.fit(maxlags=max_lags, ic='aic')

                # Используем остаточную дисперсию как критерий
                residuals = results.resid
                score = np.var(residuals[target_col])

                if score < best_score:
                    best_score = score
                    best_feature = feature
            except:
                continue

        if best_feature is None:
            break

        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        feature_scores[best_feature] = best_score

    stats = {
        'selected_features': selected_features,
        'feature_scores': feature_scores,
        'num_selected': len(selected_features)
    }

    return selected_features, stats


def select_by_aic(df: pd.DataFrame,
                   target_col: str,
                   max_features: int = 10) -> Tuple[List[str], Dict]:
    """
    Отбор признаков на основе информационного критерия Акаике (AIC)
    Использует пошаговый forward selection

    Args:
        df: DataFrame с данными
        target_col: Целевая переменная
        max_features: Максимальное количество признаков

    Returns:
        Tuple[List[str], Dict]: Список выбранных признаков и их AIC scores
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    try:
        import statsmodels.api as sm
    except ImportError:
        raise ImportError("statsmodels is required for AIC method")

    # Подготовка данных
    feature_cols = [col for col in df.columns if col != target_col]
    X_full = df[feature_cols].dropna()
    y = df.loc[X_full.index, target_col]

    selected_features = []
    remaining_features = feature_cols.copy()
    aic_scores = {}

    for _ in range(min(max_features, len(remaining_features))):
        best_feature = None
        best_aic = np.inf

        for feature in remaining_features:
            # Создаем временный набор признаков
            temp_features = selected_features + [feature]
            X_temp = sm.add_constant(X_full[temp_features])

            try:
                # Обучаем OLS модель
                model = sm.OLS(y, X_temp).fit()
                aic = model.aic

                if aic < best_aic:
                    best_aic = aic
                    best_feature = feature
            except:
                continue

        if best_feature is None:
            break

        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        aic_scores[best_feature] = best_aic

    stats = {
        'selected_features': selected_features,
        'aic_scores': aic_scores,
        'final_aic': best_aic if selected_features else np.inf
    }

    return selected_features, stats


def select_by_bic(df: pd.DataFrame,
                   target_col: str,
                   max_features: int = 10) -> Tuple[List[str], Dict]:
    """
    Отбор признаков на основе байесовского информационного критерия (BIC)
    Использует пошаговый forward selection

    Args:
        df: DataFrame с данными
        target_col: Целевая переменная
        max_features: Максимальное количество признаков

    Returns:
        Tuple[List[str], Dict]: Список выбранных признаков и их BIC scores
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    try:
        import statsmodels.api as sm
    except ImportError:
        raise ImportError("statsmodels is required for BIC method")

    # Подготовка данных
    feature_cols = [col for col in df.columns if col != target_col]
    X_full = df[feature_cols].dropna()
    y = df.loc[X_full.index, target_col]

    selected_features = []
    remaining_features = feature_cols.copy()
    bic_scores = {}

    for _ in range(min(max_features, len(remaining_features))):
        best_feature = None
        best_bic = np.inf

        for feature in remaining_features:
            # Создаем временный набор признаков
            temp_features = selected_features + [feature]
            X_temp = sm.add_constant(X_full[temp_features])

            try:
                # Обучаем OLS модель
                model = sm.OLS(y, X_temp).fit()
                bic = model.bic

                if bic < best_bic:
                    best_bic = bic
                    best_feature = feature
            except:
                continue

        if best_feature is None:
            break

        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        bic_scores[best_feature] = best_bic

    stats = {
        'selected_features': selected_features,
        'bic_scores': bic_scores,
        'final_bic': best_bic if selected_features else np.inf
    }

    return selected_features, stats


def tournament_feature_selection(df: pd.DataFrame,
                                   target_col: str,
                                   methods: List[str] = ['correlation', 'rfr'],
                                   n_features: int = 10) -> Dict[str, List[str]]:
    """
    Турнирный отбор признаков - применяет несколько методов и сравнивает результаты

    Args:
        df: DataFrame с данными
        target_col: Целевая переменная
        methods: Список методов для применения
        n_features: Количество признаков для каждого метода

    Returns:
        Dict[str, List[str]]: Результаты каждого метода
    """
    results = {}

    for method in methods:
        try:
            if method == 'correlation':
                features, _ = select_by_correlation(df, target_col, top_n=n_features)
            elif method == 'svs':
                features, _ = select_by_svs(df, target_col, n_features=n_features)
            elif method == 'rfr':
                features, _ = select_by_rfr(df, target_col, n_features=n_features)
            elif method == 'var':
                features, _ = select_by_var(df, target_col, max_features=n_features)
            elif method == 'aic':
                features, _ = select_by_aic(df, target_col, max_features=n_features)
            elif method == 'bic':
                features, _ = select_by_bic(df, target_col, max_features=n_features)
            else:
                continue

            results[method] = features
        except Exception as e:
            print(f"Error in {method}: {str(e)}")
            results[method] = []

    return results


def get_consensus_features(tournament_results: Dict[str, List[str]],
                             min_votes: int = 2) -> List[str]:
    """
    Получение признаков, которые были выбраны несколькими методами

    Args:
        tournament_results: Результаты турнирного отбора
        min_votes: Минимальное количество "голосов" (методов)

    Returns:
        List[str]: Список признаков с минимальным количеством голосов
    """
    from collections import Counter

    # Подсчитываем, сколько раз каждый признак был выбран
    all_features = []
    for features in tournament_results.values():
        all_features.extend(features)

    feature_counts = Counter(all_features)

    # Выбираем признаки с минимальным количеством голосов
    consensus_features = [feature for feature, count in feature_counts.items()
                           if count >= min_votes]

    # Сортируем по количеству голосов
    consensus_features.sort(key=lambda x: feature_counts[x], reverse=True)

    return consensus_features
