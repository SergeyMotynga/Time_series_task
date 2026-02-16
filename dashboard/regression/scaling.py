"""
Функции для стандартизации и нормализации данных
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def apply_standard_scaler(df: pd.DataFrame, columns: list = None) -> tuple[pd.DataFrame, dict]:
    """
    Применяет StandardScaler (Z-score нормализация)

    Преобразует данные так, чтобы mean=0 и std=1
    Формула: (x - mean) / std

    Args:
        df: DataFrame с данными
        columns: Список колонок для стандартизации (None = все числовые)

    Returns:
        Tuple[DataFrame, dict]: Стандартизованный DataFrame и параметры scaler
    """
    df_scaled = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    scaler = StandardScaler()
    df_scaled[columns] = scaler.fit_transform(df[columns])

    params = {
        'method': 'StandardScaler',
        'columns': columns,
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist()
    }

    return df_scaled, params


def apply_minmax_scaler(df: pd.DataFrame, columns: list = None, feature_range: tuple = (0, 1)) -> tuple[pd.DataFrame, dict]:
    """
    Применяет MinMaxScaler (нормализация в диапазон)

    Преобразует данные в заданный диапазон (обычно [0, 1])
    Формула: (x - min) / (max - min)

    Args:
        df: DataFrame с данными
        columns: Список колонок для нормализации (None = все числовые)
        feature_range: Диапазон для масштабирования (min, max)

    Returns:
        Tuple[DataFrame, dict]: Нормализованный DataFrame и параметры scaler
    """
    df_scaled = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    scaler = MinMaxScaler(feature_range=feature_range)
    df_scaled[columns] = scaler.fit_transform(df[columns])

    params = {
        'method': 'MinMaxScaler',
        'columns': columns,
        'feature_range': feature_range,
        'data_min': scaler.data_min_.tolist(),
        'data_max': scaler.data_max_.tolist()
    }

    return df_scaled, params


def apply_robust_scaler(df: pd.DataFrame, columns: list = None) -> tuple[pd.DataFrame, dict]:
    """
    Применяет RobustScaler (устойчивая нормализация)

    Использует медиану и IQR вместо mean и std
    Устойчив к выбросам
    Формула: (x - median) / IQR

    Args:
        df: DataFrame с данными
        columns: Список колонок для нормализации (None = все числовые)

    Returns:
        Tuple[DataFrame, dict]: Нормализованный DataFrame и параметры scaler
    """
    df_scaled = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    scaler = RobustScaler()
    df_scaled[columns] = scaler.fit_transform(df[columns])

    params = {
        'method': 'RobustScaler',
        'columns': columns,
        'center': scaler.center_.tolist(),
        'scale': scaler.scale_.tolist()
    }

    return df_scaled, params


def get_scaling_info(df_original: pd.DataFrame, df_scaled: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Возвращает сравнительную статистику до и после стандартизации

    Args:
        df_original: Исходный DataFrame
        df_scaled: Стандартизованный DataFrame
        columns: Список колонок для сравнения

    Returns:
        DataFrame со статистикой
    """
    stats = []

    for col in columns:
        stats.append({
            'Признак': col,
            'Mean (до)': f"{df_original[col].mean():.4f}",
            'Mean (после)': f"{df_scaled[col].mean():.4f}",
            'Std (до)': f"{df_original[col].std():.4f}",
            'Std (после)': f"{df_scaled[col].std():.4f}",
            'Min (до)': f"{df_original[col].min():.4f}",
            'Min (после)': f"{df_scaled[col].min():.4f}",
            'Max (до)': f"{df_original[col].max():.4f}",
            'Max (после)': f"{df_scaled[col].max():.4f}"
        })

    return pd.DataFrame(stats)
