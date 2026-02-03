"""
Модуль для определения и обработки выбросов в данных
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from scipy import stats


def detect_outliers_iqr(data: pd.Series, multiplier: float = 1.5) -> Tuple[pd.Series, Dict]:
    """
    Определение выбросов методом межквартильного размаха (IQR)

    Args:
        data: Временной ряд для анализа
        multiplier: Множитель для IQR (по умолчанию 1.5)

    Returns:
        Tuple[pd.Series, Dict]: Маска выбросов и статистика
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    outliers_mask = (data < lower_bound) | (data > upper_bound)

    stats_dict = {
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'outliers_count': outliers_mask.sum(),
        'outliers_percentage': (outliers_mask.sum() / len(data)) * 100
    }

    return outliers_mask, stats_dict


def detect_outliers_zscore(data: pd.Series, threshold: float = 3.0) -> Tuple[pd.Series, Dict]:
    """
    Определение выбросов методом Z-score

    Args:
        data: Временной ряд для анализа
        threshold: Пороговое значение Z-score (по умолчанию 3.0)

    Returns:
        Tuple[pd.Series, Dict]: Маска выбросов и статистика
    """
    z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
    outliers_mask = z_scores > threshold

    stats_dict = {
        'mean': data.mean(),
        'std': data.std(),
        'threshold': threshold,
        'outliers_count': outliers_mask.sum(),
        'outliers_percentage': (outliers_mask.sum() / len(data)) * 100
    }

    return outliers_mask, stats_dict


def detect_outliers_modified_zscore(data: pd.Series, threshold: float = 3.5) -> Tuple[pd.Series, Dict]:
    """
    Определение выбросов модифицированным методом Z-score (на основе медианы)
    Более устойчив к выбросам, чем классический Z-score

    Args:
        data: Временной ряд для анализа
        threshold: Пороговое значение modified Z-score (по умолчанию 3.5)

    Returns:
        Tuple[pd.Series, Dict]: Маска выбросов и статистика
    """
    median = data.median()
    mad = np.median(np.abs(data - median))
    modified_z_scores = 0.6745 * (data - median) / mad if mad != 0 else pd.Series([0] * len(data), index=data.index)

    outliers_mask = np.abs(modified_z_scores) > threshold

    stats_dict = {
        'median': median,
        'mad': mad,
        'threshold': threshold,
        'outliers_count': outliers_mask.sum(),
        'outliers_percentage': (outliers_mask.sum() / len(data)) * 100
    }

    return outliers_mask, stats_dict


def remove_outliers(df: pd.DataFrame, columns: List[str], method: str = 'iqr',
                     **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """
    Удаление выбросов из DataFrame

    Args:
        df: DataFrame с данными
        columns: Список колонок для обработки
        method: Метод определения выбросов ('iqr', 'zscore', 'modified_zscore')
        **kwargs: Дополнительные параметры для метода

    Returns:
        Tuple[pd.DataFrame, Dict]: DataFrame без выбросов и статистика по каждой колонке
    """
    df_clean = df.copy()
    stats = {}

    for col in columns:
        if col not in df.columns:
            continue

        if method == 'iqr':
            outliers_mask, col_stats = detect_outliers_iqr(df[col],
                                                            multiplier=kwargs.get('multiplier', 1.5))
        elif method == 'zscore':
            outliers_mask, col_stats = detect_outliers_zscore(df[col],
                                                                threshold=kwargs.get('threshold', 3.0))
        elif method == 'modified_zscore':
            outliers_mask, col_stats = detect_outliers_modified_zscore(df[col],
                                                                         threshold=kwargs.get('threshold', 3.5))
        else:
            raise ValueError(f"Unknown method: {method}")

        # Замена выбросов на NaN
        df_clean.loc[outliers_mask, col] = np.nan
        stats[col] = col_stats

    return df_clean, stats


def handle_outliers(df: pd.DataFrame, columns: List[str], method: str = 'iqr',
                     action: str = 'remove', **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """
    Обработка выбросов: удаление строк или замена значений

    Args:
        df: DataFrame с данными
        columns: Список колонок для обработки
        method: Метод определения выбросов ('iqr', 'zscore', 'modified_zscore')
        action: Действие с выбросами:
                - 'remove': удалить строки с выбросами
                - 'replace': заменить на NaN
                - 'median': заменить на медиану
                - 'mean': заменить на среднее
                - 'clip': ограничить значениями границ (clipping)
        **kwargs: Дополнительные параметры для метода

    Returns:
        Tuple[pd.DataFrame, Dict]: Обработанный DataFrame и статистика
    """
    df_processed = df.copy()
    stats = {}

    # Определяем маску выбросов для всех колонок
    combined_mask = pd.Series([False] * len(df), index=df.index)

    for col in columns:
        if col not in df.columns:
            continue

        if method == 'iqr':
            outliers_mask, col_stats = detect_outliers_iqr(df[col],
                                                            multiplier=kwargs.get('multiplier', 1.5))
        elif method == 'zscore':
            outliers_mask, col_stats = detect_outliers_zscore(df[col],
                                                                threshold=kwargs.get('threshold', 3.0))
        elif method == 'modified_zscore':
            outliers_mask, col_stats = detect_outliers_modified_zscore(df[col],
                                                                         threshold=kwargs.get('threshold', 3.5))
        else:
            raise ValueError(f"Unknown method: {method}")

        if action == 'remove':
            # Объединяем маски для всех колонок
            combined_mask = combined_mask | outliers_mask
        elif action == 'replace':
            # Замена на NaN
            df_processed.loc[outliers_mask, col] = np.nan
        elif action == 'median':
            # Замена на медиану
            df_processed.loc[outliers_mask, col] = df[col].median()
        elif action == 'mean':
            # Замена на среднее
            df_processed.loc[outliers_mask, col] = df[col].mean()
        elif action == 'clip':
            # Ограничение значений границами (clipping)
            if method == 'iqr':
                lower_bound = col_stats.get('lower_bound')
                upper_bound = col_stats.get('upper_bound')
                df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
            elif method in ['zscore', 'modified_zscore']:
                # Для z-score используем границы на основе threshold
                mean_val = df[col].mean()
                std_val = df[col].std()
                threshold = kwargs.get('threshold', 3.0)
                lower_bound = mean_val - threshold * std_val
                upper_bound = mean_val + threshold * std_val
                df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
        else:
            raise ValueError(f"Unknown action: {action}")

        stats[col] = col_stats

    # Удаляем строки с выбросами, если выбрано действие 'remove'
    if action == 'remove':
        df_processed = df_processed[~combined_mask]
        stats['total_removed_rows'] = combined_mask.sum()
        stats['removed_percentage'] = (combined_mask.sum() / len(df)) * 100

    return df_processed, stats


def analyze_outliers(df: pd.DataFrame, columns: List[str] = None,
                      method: str = 'iqr', **kwargs) -> Dict:
    """
    Анализ выбросов без изменения данных

    Args:
        df: DataFrame с данными
        columns: Список колонок для анализа (если None, то все числовые колонки)
        method: Метод определения выбросов ('iqr', 'zscore', 'modified_zscore')
        **kwargs: Дополнительные параметры для метода

    Returns:
        Dict: Статистика по выбросам для каждой колонки
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    stats = {}

    for col in columns:
        if col not in df.columns:
            continue

        if method == 'iqr':
            outliers_mask, col_stats = detect_outliers_iqr(df[col],
                                                            multiplier=kwargs.get('multiplier', 1.5))
        elif method == 'zscore':
            outliers_mask, col_stats = detect_outliers_zscore(df[col],
                                                                threshold=kwargs.get('threshold', 3.0))
        elif method == 'modified_zscore':
            outliers_mask, col_stats = detect_outliers_modified_zscore(df[col],
                                                                         threshold=kwargs.get('threshold', 3.5))
        else:
            raise ValueError(f"Unknown method: {method}")

        stats[col] = col_stats
        stats[col]['outlier_indices'] = df[outliers_mask].index.tolist()

    return stats
