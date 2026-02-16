"""
Универсальная временная агрегация для данных с разной частотой измерений

Работает с любыми временными рядами где:
- Целевая переменная измеряется редко (мало точек)
- Признаки измеряются часто (много точек)

Решение: агрегация признаков в окна вокруг измерений целевой переменной.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.stats import pearsonr


def aggregate_by_target_timestamps(
    df: pd.DataFrame,
    target_col: str,
    window_hours: int = 24,
    shift_hours: int = 0,
    agg_functions: List[str] = None
) -> pd.DataFrame:
    """
    Универсальная агрегация данных на основе измерений целевой переменной.

    **Когда использовать:**
    - Целевая переменная измеряется РЕДКО (например, лаб. анализы)
    - Другие сенсоры измеряются ЧАСТО (технологические параметры)
    - Нельзя просто заполнять пропуски в целевой переменной (это создает синтетику)

    **Что делает:**
    1. Находит все реальные измерения целевой переменной
    2. Для каждого измерения:
       - Берет временное окно ДО этого измерения
       - Агрегирует данные всех признаков в этом окне
    3. Создает датасет где кол-во строк = кол-во реальных измерений цели

    **Параметры:**
        df: DataFrame с DatetimeIndex
        target_col: Целевая переменная (та что предсказываем)
        window_hours: Размер окна усреднения в часах
        shift_hours: Сдвиг окна назад во времени (компенсация задержки)
        agg_functions: Функции ['mean', 'std', 'min', 'max', 'last', 'first', 'median', 'count']

    **Возвращает:**
        DataFrame где:
        - Строк = количество реальных измерений целевой переменной
        - Колонки = целевая + агрегированные признаки

    **Пример:**
        ```python
        # Целевая переменная: 10 измерений
        # Признак A: 1000 измерений
        # Признак B: 500 измерений

        df_agg = aggregate_by_target_timestamps(
            df,
            target_col='lab_analysis',
            window_hours=24,  # усреднить за последние 24 часа
            shift_hours=1     # со сдвигом 1 час назад
        )

        # Результат: 10 строк × (1 + N*функций) колонок
        # Признак A_mean, A_std, A_min, A_max, ...
        # Признак B_mean, B_std, B_min, B_max, ...
        ```
    """
    if agg_functions is None:
        agg_functions = ['mean', 'std', 'min', 'max', 'last']

    # Валидация
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame должен иметь DatetimeIndex. Используйте df.set_index('date_column')")

    if target_col not in df.columns:
        raise ValueError(f"Целевая колонка '{target_col}' не найдена. Доступные: {list(df.columns)}")

    # Сортируем по времени
    df = df.sort_index()

    # Находим реальные (не-NaN) измерения целевой переменной
    target_measurements = df[df[target_col].notna()].copy()

    if len(target_measurements) == 0:
        raise ValueError(f"В колонке '{target_col}' нет ненулевых значений")

    # Признаки для агрегации (все кроме целевой)
    feature_cols = [col for col in df.columns if col != target_col]

    # Агрегируем для каждого измерения
    aggregated_rows = []

    for timestamp, row in target_measurements.iterrows():
        target_value = row[target_col]

        # Определяем временное окно с учетом сдвига
        window_end = timestamp - pd.Timedelta(hours=shift_hours)
        window_start = window_end - pd.Timedelta(hours=window_hours)

        # Извлекаем данные в окне
        window_data = df.loc[window_start:window_end, feature_cols]

        if len(window_data) == 0:
            continue

        # Строка с агрегированными данными
        agg_row = {'timestamp': timestamp, target_col: target_value}

        # Агрегируем каждый признак
        for feature in feature_cols:
            feature_data = window_data[feature].dropna()

            if len(feature_data) == 0:
                for func in agg_functions:
                    agg_row[f'{feature}_{func}'] = np.nan
                continue

            for func in agg_functions:
                if func == 'mean':
                    agg_row[f'{feature}_mean'] = feature_data.mean()
                elif func == 'std':
                    agg_row[f'{feature}_std'] = feature_data.std()
                elif func == 'min':
                    agg_row[f'{feature}_min'] = feature_data.min()
                elif func == 'max':
                    agg_row[f'{feature}_max'] = feature_data.max()
                elif func == 'last':
                    agg_row[f'{feature}_last'] = feature_data.iloc[-1]
                elif func == 'first':
                    agg_row[f'{feature}_first'] = feature_data.iloc[0]
                elif func == 'median':
                    agg_row[f'{feature}_median'] = feature_data.median()
                elif func == 'count':
                    agg_row[f'{feature}_count'] = len(feature_data)

        aggregated_rows.append(agg_row)

    if not aggregated_rows:
        raise ValueError("Не удалось создать агрегированные строки. Проверьте параметры window_hours и shift_hours.")

    # Создаем DataFrame
    df_result = pd.DataFrame(aggregated_rows)
    df_result['timestamp'] = pd.to_datetime(df_result['timestamp'])
    df_result = df_result.set_index('timestamp')

    return df_result


def find_optimal_aggregation_params(
    df: pd.DataFrame,
    target_col: str,
    window_hours_candidates: List[int] = None,
    shift_hours_candidates: List[int] = None,
    metric: str = 'correlation'
) -> Dict:
    """
    Автоматически находит оптимальные параметры агрегации.

    Перебирает разные комбинации window_hours и shift_hours,
    создает агрегированные датасеты и оценивает качество по метрике.

    **Параметры:**
        df: DataFrame с данными
        target_col: Целевая переменная
        window_hours_candidates: Список кандидатов для окна (None = авто)
        shift_hours_candidates: Список кандидатов для сдвига (None = авто)
        metric: Метрика оценки ('correlation', 'avg_correlation')

    **Возвращает:**
        dict: {
            'optimal_window_hours': int,
            'optimal_shift_hours': int,
            'best_score': float,
            'all_results': DataFrame
        }
    """
    # Автоопределение кандидатов на основе частоты измерений целевой переменной
    if window_hours_candidates is None or shift_hours_candidates is None:
        info = get_target_measurements_info(df, target_col)
        avg_interval = info.get('avg_interval_hours', 168)

        if window_hours_candidates is None:
            # Окна: от часа до среднего интервала
            window_hours_candidates = [
                max(1, int(avg_interval * 0.1)),
                max(1, int(avg_interval * 0.25)),
                max(1, int(avg_interval * 0.5)),
                max(1, int(avg_interval * 0.75)),
                max(1, int(avg_interval))
            ]
            window_hours_candidates = sorted(list(set(window_hours_candidates)))

        if shift_hours_candidates is None:
            # Сдвиги: 0, 1, 2, 4, 8 часов
            shift_hours_candidates = [0, 1, 2, 4, 8]

    results = []

    for window_h in window_hours_candidates:
        for shift_h in shift_hours_candidates:
            try:
                df_agg = aggregate_by_target_timestamps(
                    df, target_col,
                    window_hours=window_h,
                    shift_hours=shift_h,
                    agg_functions=['mean']
                )

                # Оценка качества
                feature_cols = [col for col in df_agg.columns if col != target_col]
                correlations = []

                for col in feature_cols:
                    data = df_agg[[col, target_col]].dropna()
                    if len(data) >= 3:
                        corr, _ = pearsonr(data[col], data[target_col])
                        correlations.append(abs(corr))

                if metric == 'correlation':
                    score = max(correlations) if correlations else 0.0
                elif metric == 'avg_correlation':
                    score = np.mean(correlations) if correlations else 0.0
                else:
                    score = 0.0

                results.append({
                    'window_hours': window_h,
                    'shift_hours': shift_h,
                    'score': score,
                    'num_rows': len(df_agg)
                })

            except Exception:
                continue

    if not results:
        return {
            'optimal_window_hours': window_hours_candidates[0] if window_hours_candidates else 24,
            'optimal_shift_hours': shift_hours_candidates[0] if shift_hours_candidates else 0,
            'best_score': 0.0,
            'all_results': pd.DataFrame()
        }

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('score', ascending=False)

    best = df_results.iloc[0]

    return {
        'optimal_window_hours': int(best['window_hours']),
        'optimal_shift_hours': int(best['shift_hours']),
        'best_score': float(best['score']),
        'all_results': df_results
    }


def get_target_measurements_info(df: pd.DataFrame, target_col: str) -> Dict:
    """
    Анализирует частоту измерений целевой переменной.

    **Возвращает:**
        dict: {
            'count': кол-во измерений,
            'dates': список дат,
            'values': список значений,
            'min_interval_hours': минимальный интервал,
            'max_interval_hours': максимальный интервал,
            'avg_interval_hours': средний интервал,
            'suggested_window_hours': рекомендуемое окно
        }
    """
    if target_col not in df.columns:
        raise ValueError(f"Колонка '{target_col}' не найдена")

    measurements = df[df[target_col].notna()].copy()

    if len(measurements) == 0:
        return {
            'count': 0,
            'dates': [],
            'values': [],
            'min_interval_hours': None,
            'max_interval_hours': None,
            'avg_interval_hours': None,
            'suggested_window_hours': 24
        }

    # Интервалы между измерениями
    intervals = []
    timestamps = measurements.index.tolist()

    for i in range(1, len(timestamps)):
        interval = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600
        intervals.append(interval)

    avg_interval = np.mean(intervals) if intervals else 24
    suggested_window = max(1, int(avg_interval * 0.5))  # 50% от среднего интервала

    return {
        'count': len(measurements),
        'dates': [ts.strftime('%Y-%m-%d %H:%M') for ts in timestamps],
        'values': measurements[target_col].tolist(),
        'min_interval_hours': min(intervals) if intervals else None,
        'max_interval_hours': max(intervals) if intervals else None,
        'avg_interval_hours': avg_interval,
        'suggested_window_hours': suggested_window
    }


def analyze_feature_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Анализирует частоту измерений для каждой колонки.

    **Возвращает:**
        DataFrame с информацией о каждой колонке:
        - название
        - количество ненулевых значений
        - процент заполненности
        - средний интервал между измерениями (часы)
    """
    stats = []

    for col in df.columns:
        non_null = df[col].notna()
        count = non_null.sum()
        pct = (count / len(df)) * 100

        # Средний интервал
        measurements = df[non_null]
        if len(measurements) > 1:
            timestamps = measurements.index.tolist()
            intervals = [
                (timestamps[i] - timestamps[i-1]).total_seconds() / 3600
                for i in range(1, len(timestamps))
            ]
            avg_interval = np.mean(intervals)
        else:
            avg_interval = None

        stats.append({
            'Признак': col,
            'Измерений': count,
            'Заполненность %': round(pct, 2),
            'Средний интервал (ч)': round(avg_interval, 2) if avg_interval else None
        })

    df_stats = pd.DataFrame(stats)
    df_stats = df_stats.sort_values('Измерений', ascending=True)

    return df_stats
