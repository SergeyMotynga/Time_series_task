"""
Модуль для сдвига временных рядов и поиска оптимального сдвига
"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr


def find_optimal_shift(df, sensor_col, target_col, max_shift=24, metric='correlation'):
    """
    Находит оптимальный сдвиг для сенсора относительно целевой переменной

    Args:
        df: DataFrame с данными
        sensor_col: Название колонки сенсора
        target_col: Название целевой переменной
        max_shift: Максимальный сдвиг для проверки
        metric: Метрика для оценки ('correlation', 'mse', 'mae')

    Returns:
        dict с результатами:
            - optimal_shift: оптимальный сдвиг
            - shift_scores: словарь {shift: score}
            - best_score: лучший score
    """
    if sensor_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"Колонки {sensor_col} или {target_col} не найдены в DataFrame")

    shift_scores = {}

    for shift in range(0, max_shift + 1):
        if shift == 0:
            shifted = df[sensor_col]
        else:
            shifted = df[sensor_col].shift(shift)

        # Удаляем NaN после сдвига
        mask = ~(shifted.isna() | df[target_col].isna())

        if mask.sum() < 10:  # Минимум 10 точек для расчета
            continue

        sensor_clean = shifted[mask]
        target_clean = df[target_col][mask]

        if metric == 'correlation':
            score, _ = pearsonr(sensor_clean, target_clean)
            score = abs(score)  # Берем абсолютное значение корреляции
        elif metric == 'mse':
            score = -np.mean((sensor_clean - target_clean) ** 2)  # Отрицательный MSE (чем больше, тем лучше)
        elif metric == 'mae':
            score = -np.mean(np.abs(sensor_clean - target_clean))  # Отрицательный MAE
        else:
            raise ValueError(f"Неизвестная метрика: {metric}")

        shift_scores[shift] = score

    if not shift_scores:
        return {
            'optimal_shift': 0,
            'shift_scores': {},
            'best_score': 0.0
        }

    optimal_shift = max(shift_scores, key=shift_scores.get)
    best_score = shift_scores[optimal_shift]

    return {
        'optimal_shift': optimal_shift,
        'shift_scores': shift_scores,
        'best_score': best_score
    }


def apply_shifts(df, shift_config):
    """
    Применяет сдвиги к нескольким сенсорам

    Args:
        df: DataFrame с данными
        shift_config: Словарь {sensor_col: shift_periods}
            Например: {'temp_sensor_1': 2, 'pressure': 5}

    Returns:
        DataFrame с добавленными сдвинутыми колонками
    """
    df_shifted = df.copy()

    for sensor_col, shift in shift_config.items():
        if sensor_col not in df.columns:
            continue

        if shift > 0:
            new_col_name = f"{sensor_col}_shift_{shift}"
            df_shifted[new_col_name] = df[sensor_col].shift(shift)

    # Удаляем строки с NaN после сдвига
    df_shifted = df_shifted.dropna()

    return df_shifted


def analyze_shift_impact(df, sensor_col, target_col, shifts_to_test=None):
    """
    Анализирует влияние различных сдвигов на корреляцию с целевой переменной

    Args:
        df: DataFrame с данными
        sensor_col: Название сенсора
        target_col: Целевая переменная
        shifts_to_test: Список сдвигов для тестирования (по умолчанию 0-24)

    Returns:
        DataFrame с результатами анализа
    """
    if shifts_to_test is None:
        shifts_to_test = list(range(0, 25))

    results = []

    for shift in shifts_to_test:
        if shift == 0:
            shifted = df[sensor_col]
        else:
            shifted = df[sensor_col].shift(shift)

        # Удаляем NaN
        mask = ~(shifted.isna() | df[target_col].isna())

        if mask.sum() < 10:
            continue

        sensor_clean = shifted[mask]
        target_clean = df[target_col][mask]

        # Корреляция
        corr, p_value = pearsonr(sensor_clean, target_clean)

        # MSE и MAE
        mse = np.mean((sensor_clean - target_clean) ** 2)
        mae = np.mean(np.abs(sensor_clean - target_clean))

        results.append({
            'shift': shift,
            'correlation': corr,
            'abs_correlation': abs(corr),
            'p_value': p_value,
            'mse': mse,
            'mae': mae,
            'data_points': mask.sum()
        })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('abs_correlation', ascending=False)

    return df_results


def create_shifted_features(df, sensor_cols, target_col, auto_find_shifts=True, max_shift=24):
    """
    Создает сдвинутые признаки для списка сенсоров

    Args:
        df: DataFrame с данными
        sensor_cols: Список сенсоров
        target_col: Целевая переменная
        auto_find_shifts: Автоматически найти оптимальные сдвиги
        max_shift: Максимальный сдвиг

    Returns:
        tuple: (df_with_shifts, shift_info)
            - df_with_shifts: DataFrame с добавленными признаками
            - shift_info: Словарь с информацией о сдвигах
    """
    df_result = df.copy()
    shift_info = {}

    for sensor in sensor_cols:
        if sensor not in df.columns or sensor == target_col:
            continue

        if auto_find_shifts:
            # Находим оптимальный сдвиг
            result = find_optimal_shift(df, sensor, target_col, max_shift=max_shift)
            optimal_shift = result['optimal_shift']

            shift_info[sensor] = {
                'optimal_shift': optimal_shift,
                'correlation': result['best_score'],
                'all_shifts': result['shift_scores']
            }

            if optimal_shift > 0:
                new_col_name = f"{sensor}_shift_{optimal_shift}"
                df_result[new_col_name] = df[sensor].shift(optimal_shift)
        else:
            # Используем фиксированный сдвиг (можно расширить)
            shift_info[sensor] = {'optimal_shift': 0, 'correlation': 0.0}

    # Удаляем строки с NaN
    df_result = df_result.dropna()

    return df_result, shift_info


def visualize_shift_analysis_data(df, sensor_col, target_col, max_shift=24):
    """
    Подготавливает данные для визуализации анализа сдвигов

    Args:
        df: DataFrame с данными
        sensor_col: Название сенсора
        target_col: Целевая переменная
        max_shift: Максимальный сдвиг

    Returns:
        DataFrame с результатами для визуализации
    """
    return analyze_shift_impact(df, sensor_col, target_col, list(range(0, max_shift + 1)))
