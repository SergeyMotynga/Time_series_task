"""
Модуль для создания lag-признаков (сдвиги временных рядов)
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple


def create_lag_features(df: pd.DataFrame,
                         columns: List[str],
                         lags: Union[int, List[int]],
                         suffix: str = '_lag') -> pd.DataFrame:
    """
    Создание lag-признаков для указанных колонок

    Args:
        df: DataFrame с временными рядами
        columns: Список колонок для создания lag-признаков
        lags: Количество лагов (int) или список конкретных лагов (List[int])
        suffix: Суффикс для новых колонок

    Returns:
        DataFrame с добавленными lag-признаками
    """
    df_with_lags = df.copy()

    # Преобразуем lags в список, если это int
    if isinstance(lags, int):
        lags_list = list(range(1, lags + 1))
    else:
        lags_list = lags

    for col in columns:
        if col not in df.columns:
            continue

        for lag in lags_list:
            lag_col_name = f"{col}{suffix}{lag}"
            df_with_lags[lag_col_name] = df[col].shift(lag)

    return df_with_lags


def create_rolling_features(df: pd.DataFrame,
                              columns: List[str],
                              windows: Union[int, List[int]],
                              funcs: List[str] = ['mean'],
                              suffix: str = '_rolling') -> pd.DataFrame:
    """
    Создание скользящих статистических признаков

    Args:
        df: DataFrame с временными рядами
        columns: Список колонок для создания признаков
        windows: Размер окна (int) или список размеров (List[int])
        funcs: Список функций ('mean', 'std', 'min', 'max', 'median')
        suffix: Суффикс для новых колонок

    Returns:
        DataFrame с добавленными rolling-признаками
    """
    df_with_rolling = df.copy()

    # Преобразуем windows в список, если это int
    if isinstance(windows, int):
        windows_list = [windows]
    else:
        windows_list = windows

    for col in columns:
        if col not in df.columns:
            continue

        for window in windows_list:
            for func in funcs:
                rolling_col_name = f"{col}{suffix}_{func}{window}"

                if func == 'mean':
                    df_with_rolling[rolling_col_name] = df[col].rolling(window=window).mean()
                elif func == 'std':
                    df_with_rolling[rolling_col_name] = df[col].rolling(window=window).std()
                elif func == 'min':
                    df_with_rolling[rolling_col_name] = df[col].rolling(window=window).min()
                elif func == 'max':
                    df_with_rolling[rolling_col_name] = df[col].rolling(window=window).max()
                elif func == 'median':
                    df_with_rolling[rolling_col_name] = df[col].rolling(window=window).median()

    return df_with_rolling


def create_diff_features(df: pd.DataFrame,
                          columns: List[str],
                          periods: Union[int, List[int]] = 1,
                          suffix: str = '_diff') -> pd.DataFrame:
    """
    Создание разностных признаков (разница между текущим и предыдущим значением)

    Args:
        df: DataFrame с временными рядами
        columns: Список колонок для создания разностных признаков
        periods: Период разности (int) или список периодов (List[int])
        suffix: Суффикс для новых колонок

    Returns:
        DataFrame с добавленными diff-признаками
    """
    df_with_diff = df.copy()

    # Преобразуем periods в список, если это int
    if isinstance(periods, int):
        periods_list = [periods]
    else:
        periods_list = periods

    for col in columns:
        if col not in df.columns:
            continue

        for period in periods_list:
            diff_col_name = f"{col}{suffix}{period}"
            df_with_diff[diff_col_name] = df[col].diff(periods=period)

    return df_with_diff


def create_shift_features(df: pd.DataFrame,
                           target_col: str,
                           feature_cols: List[str],
                           shifts: Dict[str, int]) -> pd.DataFrame:
    """
    Создание сдвинутых признаков с индивидуальными сдвигами для каждого датчика
    (для учета технологической задержки)

    Args:
        df: DataFrame с временными рядами
        target_col: Целевая переменная (качество)
        feature_cols: Список датчиков-признаков
        shifts: Словарь {название_датчика: величина_сдвига}

    Returns:
        DataFrame с сдвинутыми признаками
    """
    df_shifted = df[[target_col]].copy()

    for col in feature_cols:
        if col not in df.columns:
            continue

        shift_value = shifts.get(col, 0)
        if shift_value != 0:
            shifted_col_name = f"{col}_shift{shift_value}"
            df_shifted[shifted_col_name] = df[col].shift(shift_value)
        else:
            df_shifted[col] = df[col]

    return df_shifted


def find_optimal_lag(df: pd.DataFrame,
                      target_col: str,
                      feature_col: str,
                      max_lag: int = 24,
                      method: str = 'correlation') -> Tuple[int, float]:
    """
    Поиск оптимального лага для признака по отношению к целевой переменной

    Args:
        df: DataFrame с данными
        target_col: Целевая переменная
        feature_col: Признак, для которого ищется оптимальный лаг
        max_lag: Максимальное значение лага для поиска
        method: Метод поиска ('correlation' - корреляция, 'mse' - минимальная MSE)

    Returns:
        Tuple[int, float]: Оптимальный лаг и значение метрики
    """
    if target_col not in df.columns or feature_col not in df.columns:
        raise ValueError(f"Columns {target_col} or {feature_col} not found in DataFrame")

    best_lag = 0
    best_score = -np.inf if method == 'correlation' else np.inf

    for lag in range(0, max_lag + 1):
        shifted_feature = df[feature_col].shift(lag)

        # Удаляем NaN значения для корректного расчета
        mask = ~(shifted_feature.isna() | df[target_col].isna())

        if mask.sum() == 0:
            continue

        if method == 'correlation':
            score = shifted_feature[mask].corr(df[target_col][mask])
            score = abs(score)  # Берем абсолютное значение корреляции

            if score > best_score:
                best_score = score
                best_lag = lag

        elif method == 'mse':
            from sklearn.metrics import mean_squared_error
            # Для MSE нужно построить простую линейную модель
            from sklearn.linear_model import LinearRegression

            X = shifted_feature[mask].values.reshape(-1, 1)
            y = df[target_col][mask].values

            model = LinearRegression()
            model.fit(X, y)
            predictions = model.predict(X)
            score = mean_squared_error(y, predictions)

            if score < best_score:
                best_score = score
                best_lag = lag

    return best_lag, best_score


def find_optimal_lags_for_all(df: pd.DataFrame,
                                target_col: str,
                                feature_cols: List[str],
                                max_lag: int = 24,
                                method: str = 'correlation') -> Dict[str, Tuple[int, float]]:
    """
    Поиск оптимальных лагов для всех признаков

    Args:
        df: DataFrame с данными
        target_col: Целевая переменная
        feature_cols: Список признаков
        max_lag: Максимальное значение лага для поиска
        method: Метод поиска ('correlation', 'mse')

    Returns:
        Dict[str, Tuple[int, float]]: Словарь {признак: (оптимальный_лаг, значение_метрики)}
    """
    optimal_lags = {}

    for col in feature_cols:
        if col not in df.columns or col == target_col:
            continue

        lag, score = find_optimal_lag(df, target_col, col, max_lag, method)
        optimal_lags[col] = (lag, score)

    return optimal_lags


def apply_optimal_lags(df: pd.DataFrame,
                        target_col: str,
                        optimal_lags: Dict[str, Tuple[int, float]]) -> pd.DataFrame:
    """
    Применение оптимальных лагов к признакам

    Args:
        df: DataFrame с данными
        target_col: Целевая переменная
        optimal_lags: Словарь с оптимальными лагами {признак: (лаг, метрика)}

    Returns:
        DataFrame с признаками, сдвинутыми на оптимальные лаги
    """
    df_lagged = df[[target_col]].copy()

    for col, (lag, _) in optimal_lags.items():
        if col not in df.columns:
            continue

        if lag > 0:
            lagged_col_name = f"{col}_opt_lag{lag}"
            df_lagged[lagged_col_name] = df[col].shift(lag)
        else:
            df_lagged[col] = df[col]

    return df_lagged


def create_interaction_features(df: pd.DataFrame,
                                  columns: List[str],
                                  interactions: List[Tuple[str, str]],
                                  operations: List[str] = ['multiply']) -> pd.DataFrame:
    """
    Создание признаков взаимодействия (произведение, сумма, разность)

    Args:
        df: DataFrame с данными
        columns: Список колонок
        interactions: Список пар колонок для взаимодействия
        operations: Список операций ('multiply', 'add', 'subtract', 'divide')

    Returns:
        DataFrame с добавленными признаками взаимодействия
    """
    df_with_interactions = df.copy()

    for col1, col2 in interactions:
        if col1 not in df.columns or col2 not in df.columns:
            continue

        for operation in operations:
            if operation == 'multiply':
                interaction_col_name = f"{col1}_x_{col2}"
                df_with_interactions[interaction_col_name] = df[col1] * df[col2]
            elif operation == 'add':
                interaction_col_name = f"{col1}_plus_{col2}"
                df_with_interactions[interaction_col_name] = df[col1] + df[col2]
            elif operation == 'subtract':
                interaction_col_name = f"{col1}_minus_{col2}"
                df_with_interactions[interaction_col_name] = df[col1] - df[col2]
            elif operation == 'divide':
                interaction_col_name = f"{col1}_div_{col2}"
                # Избегаем деления на ноль
                df_with_interactions[interaction_col_name] = df[col1] / (df[col2] + 1e-10)

    return df_with_interactions
