"""
Страница регрессионного анализа
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
from io import StringIO

from dashboard.data_processing.info_about_dataframe import info_about_dataframe
from dashboard.data_processing.select_time_interval import start_date, end_date, filter_dataframe
from dashboard.regression.outlier_processing import (
    detect_outliers_iqr,
    detect_outliers_zscore,
    remove_outliers,
    handle_outliers,
    analyze_outliers
)
from dashboard.regression.lag_features import (
    create_lag_features,
    create_rolling_features,
    find_optimal_lag
)
from dashboard.regression.feature_selection import (
    select_by_correlation,
    select_by_rfr,
    tournament_feature_selection,
    get_consensus_features
)
from dashboard.regression.scaling import (
    apply_standard_scaler,
    apply_minmax_scaler,
    apply_robust_scaler,
    get_scaling_info
)
from dashboard.regression.time_shift import (
    find_optimal_shift,
    apply_shifts,
    analyze_shift_impact,
    create_shifted_features,
    visualize_shift_analysis_data
)
from dashboard.request_functions.regression_request_func import (
    get_regression_models,
    train_regression_model,
    train_multiple_regression_models
)
from dashboard.visualization.regression_plots import (
    plot_actual_vs_predicted,
    plot_residuals,
    plot_feature_importance,
    plot_prediction_timeline,
    plot_model_comparison,
    plot_metrics_comparison_table,
    plot_correlation_matrix,
    plot_sensor_graph,
    plot_multiple_sensors,
    plot_shift_correlation,
    plot_shifted_sensor_comparison,
    plot_multiple_shifts_heatmap
)


def render_regression_page(df, outlier_percentage):
    """
    Главная страница регрессионного анализа

    Args:
        df: Исходный DataFrame с данными
        outlier_percentage: Процент выбросов
    """
    st.title("Регрессионный анализ")

    if df is None or df.empty:
        st.warning("Загрузите данные для начала работы")
        return

    # Инициализация session state
    if 'regression_df' not in st.session_state:
        st.session_state.regression_df = df.copy()
    if 'regression_features' not in st.session_state:
        st.session_state.regression_features = []
    if 'regression_target' not in st.session_state:
        st.session_state.regression_target = None
    if 'regression_results' not in st.session_state:
        st.session_state.regression_results = {}

    # Боковая панель с настройками
    with st.sidebar:
        st.header("Настройки анализа")

        # Выбор целевой переменной
        target_col = st.selectbox(
            "Целевая переменная (Y)",
            options=df.columns.tolist(),
            help="Переменная, которую нужно предсказать"
        )
        st.session_state.regression_target = target_col

        # Размер тестовой выборки
        test_size = st.slider(
            "Размер тестовой выборки (%)",
            min_value=10,
            max_value=50,
            value=20,
            step=5,
            help="Процент данных для тестирования модели"
        ) / 100

    # Вкладки для разных этапов анализа
    tabs = st.tabs([
        "📊 Обзор данных",
        "📈 Визуализация сенсоров",
        "🔍 Анализ корреляций",
        "⚠️ Обработка выбросов",
        "⚖️ Стандартизация данных",
        "🔧 Создание признаков",
        "🎯 Отбор признаков",
        "🤖 Обучение моделей",
        "📉 Результаты"
    ])

    # ========== Вкладка 1: Обзор данных ==========
    with tabs[0]:
        st.subheader("Информация о датасете")
        info_about_dataframe(st.session_state.regression_df)

        st.markdown("---")
        st.subheader("Первые строки данных")
        st.dataframe(st.session_state.regression_df.head(20))

        st.markdown("---")
        st.subheader("Статистика по переменным")
        st.dataframe(st.session_state.regression_df.describe())

    # ========== Вкладка 2: Визуализация сенсоров ==========
    with tabs[1]:
        st.subheader("Графики сенсоров")

        # Выбор сенсоров для визуализации
        sensors_to_plot = st.multiselect(
            "Выберите сенсоры для отображения",
            options=st.session_state.regression_df.columns.tolist(),
            default=[target_col] if target_col else []
        )

        if sensors_to_plot:
            if len(sensors_to_plot) == 1:
                # Один сенсор
                fig = plot_sensor_graph(st.session_state.regression_df, sensors_to_plot[0])
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Несколько сенсоров
                fig = plot_multiple_sensors(st.session_state.regression_df, sensors_to_plot)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Выберите хотя бы один сенсор для визуализации")

    # ========== Вкладка 3: Анализ корреляций ==========
    with tabs[2]:
        st.subheader("Корреляционная матрица")

        # Выбор переменных для корреляционной матрицы
        corr_vars = st.multiselect(
            "Выберите переменные для анализа корреляций",
            options=st.session_state.regression_df.columns.tolist(),
            default=st.session_state.regression_df.columns.tolist()[:min(10, len(st.session_state.regression_df.columns))]
        )

        if corr_vars and len(corr_vars) >= 2:
            fig = plot_correlation_matrix(st.session_state.regression_df[corr_vars])
            st.plotly_chart(fig, use_container_width=True)

            # Показать топ корреляций с целевой переменной
            if target_col in corr_vars:
                st.markdown("---")
                st.subheader(f"Корреляция с целевой переменной '{target_col}'")
                corr_with_target = st.session_state.regression_df[corr_vars].corr()[target_col].sort_values(ascending=False)
                corr_df = pd.DataFrame({
                    'Признак': corr_with_target.index,
                    'Корреляция': corr_with_target.values
                })
                st.dataframe(corr_df)
        else:
            st.info("Выберите хотя бы 2 переменные для построения корреляционной матрицы")

    # ========== Вкладка 4: Обработка выбросов ==========
    with tabs[3]:
        st.subheader("Обнаружение и удаление выбросов")

        outlier_method = st.selectbox(
            "Метод обнаружения выбросов",
            options=["IQR (Межквартильный размах)", "Z-score", "Modified Z-score"],
            help="IQR: классический метод, Z-score: для нормального распределения"
        )

        col1, col2 = st.columns(2)
        with col1:
            if outlier_method == "IQR (Межквартильный размах)":
                multiplier = st.slider("Множитель IQR", 1.0, 3.0, 1.5, 0.1)
                params = {'multiplier': multiplier}
                method = 'iqr'
            else:
                threshold = st.slider("Порог Z-score", 2.0, 4.0, 3.0, 0.5)
                params = {'threshold': threshold}
                method = 'zscore' if outlier_method == "Z-score" else 'modified_zscore'

        with col2:
            if st.button("Анализировать выбросы"):
                # Анализ выбросов
                column_stats = analyze_outliers(st.session_state.regression_df, method=method, **params)

                # Подсчитываем общую статистику
                total_rows = len(st.session_state.regression_df)
                outliers_per_column = {col: stats['outliers_count'] for col, stats in column_stats.items()}

                # Находим строки с выбросами хотя бы в одной колонке
                all_outlier_indices = set()
                for col_stats in column_stats.values():
                    all_outlier_indices.update(col_stats.get('outlier_indices', []))
                rows_with_outliers = len(all_outlier_indices)
                outlier_percentage = (rows_with_outliers / total_rows * 100) if total_rows > 0 else 0

                st.markdown("**Статистика выбросов:**")
                stats_df = pd.DataFrame([
                    {"Метрика": "Всего строк", "Значение": total_rows},
                    {"Метрика": "Строк с выбросами", "Значение": rows_with_outliers},
                    {"Метрика": "Процент выбросов", "Значение": f"{outlier_percentage:.2f}%"}
                ])
                st.dataframe(stats_df, hide_index=True)

                st.markdown("**Выбросы по колонкам:**")
                col_df = pd.DataFrame([
                    {"Колонка": k, "Количество выбросов": v}
                    for k, v in outliers_per_column.items()
                    if v > 0
                ]).sort_values("Количество выбросов", ascending=False)
                if not col_df.empty:
                    st.dataframe(col_df, hide_index=True)
                else:
                    st.success("Выбросов не обнаружено!")

        st.markdown("---")
        remove_method = st.radio(
            "Метод обработки выбросов",
            options=["Удалить строки с выбросами", "Заменить на границы (clip)", "Заменить на NaN и заполнить медианой"],
            help="Удаление: убирает строки, Clip: ограничивает значения, NaN: заменяет и заполняет"
        )

        if st.button("Применить обработку выбросов"):
            with st.spinner("Обработка выбросов..."):
                action_map = {
                    "Удалить строки с выбросами": "remove",
                    "Заменить на границы (clip)": "clip",
                    "Заменить на NaN и заполнить медианой": "median"
                }

                # Получаем все числовые колонки
                numeric_columns = st.session_state.regression_df.select_dtypes(include=[np.number]).columns.tolist()

                # Запоминаем исходное количество строк
                rows_before = len(st.session_state.regression_df)

                df_cleaned, outlier_stats = handle_outliers(
                    st.session_state.regression_df,
                    columns=numeric_columns,
                    method=method,
                    action=action_map[remove_method],
                    **params
                )

                rows_after = len(df_cleaned)
                rows_removed = rows_before - rows_after

                st.session_state.regression_df = df_cleaned

            # Показываем результаты обработки
            st.success("✅ Обработка выбросов завершена!")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Было строк", rows_before)
            with col2:
                st.metric("Осталось строк", rows_after)
            with col3:
                st.metric("Удалено строк", rows_removed, delta=f"-{(rows_removed/rows_before*100):.1f}%" if rows_removed > 0 else "0%")

            st.info(f"Метод: {outlier_method}, Действие: {remove_method}")

            if st.button("Обновить страницу с новыми данными"):
                st.rerun()

    # ========== Вкладка 5: Стандартизация данных ==========
    with tabs[4]:
        st.subheader("Стандартизация и нормализация данных")

        st.info("""
        **Зачем нужна стандартизация?**

        Линейные модели (Linear Regression, Elastic Net) чувствительны к масштабу признаков.
        Если признаки имеют очень разный масштаб (например, температура ~100, давление ~0.1),
        модель может неправильно определить важность признаков и коэффициенты.

        Стандартизация приводит все признаки к одному масштабу, что помогает моделям работать лучше.
        """)

        # Выбор метода стандартизации
        scaling_method = st.selectbox(
            "Метод стандартизации",
            options=[
                "StandardScaler (Z-score)",
                "MinMaxScaler (0-1)",
                "RobustScaler (устойчивый к выбросам)"
            ],
            help="Выберите метод стандартизации данных"
        )

        # Описание методов
        method_descriptions = {
            "StandardScaler (Z-score)": """
            **StandardScaler** - наиболее популярный метод
            - Приводит данные к mean=0 и std=1
            - Формула: (x - mean) / std
            - Рекомендуется для большинства задач
            - Чувствителен к выбросам
            """,
            "MinMaxScaler (0-1)": """
            **MinMaxScaler** - нормализация в диапазон
            - Приводит данные к диапазону [0, 1]
            - Формула: (x - min) / (max - min)
            - Удобен для нейронных сетей
            - Очень чувствителен к выбросам
            """,
            "RobustScaler (устойчивый к выбросам)": """
            **RobustScaler** - устойчивая нормализация
            - Использует медиану и IQR вместо mean и std
            - Формула: (x - median) / IQR
            - Устойчив к выбросам
            - Рекомендуется если есть выбросы
            """
        }

        st.markdown(method_descriptions[scaling_method])

        st.markdown("---")

        # Выбор колонок для стандартизации
        numeric_columns = st.session_state.regression_df.select_dtypes(include=['float64', 'int64']).columns.tolist()

        # По умолчанию выбираем все колонки кроме целевой
        default_cols = [col for col in numeric_columns if col != target_col]

        columns_to_scale = st.multiselect(
            "Выберите колонки для стандартизации",
            options=numeric_columns,
            default=default_cols,
            help="Обычно стандартизуют только признаки, целевую переменную оставляют как есть"
        )

        if not columns_to_scale:
            st.warning("Выберите хотя бы одну колонку для стандартизации")
        else:
            # Показываем текущую статистику
            st.markdown("**Текущая статистика выбранных признаков:**")
            current_stats = st.session_state.regression_df[columns_to_scale].describe()
            st.dataframe(current_stats)

            col1, col2 = st.columns([1, 2])

            with col1:
                if st.button("Применить стандартизацию", type="primary"):
                    with st.spinner("Применение стандартизации..."):
                        # Применяем выбранный метод
                        if scaling_method == "StandardScaler (Z-score)":
                            df_scaled, scaler_params = apply_standard_scaler(
                                st.session_state.regression_df,
                                columns=columns_to_scale
                            )
                        elif scaling_method == "MinMaxScaler (0-1)":
                            df_scaled, scaler_params = apply_minmax_scaler(
                                st.session_state.regression_df,
                                columns=columns_to_scale
                            )
                        else:  # RobustScaler
                            df_scaled, scaler_params = apply_robust_scaler(
                                st.session_state.regression_df,
                                columns=columns_to_scale
                            )

                        # Сохраняем стандартизованные данные
                        st.session_state.regression_df = df_scaled
                        st.session_state['scaling_applied'] = True
                        st.session_state['scaling_params'] = scaler_params

                        st.success("✅ Стандартизация применена!")
                        st.rerun()

            with col2:
                if st.session_state.get('scaling_applied', False):
                    st.success("✅ Данные стандартизованы")
                    st.info(f"Метод: {st.session_state.get('scaling_params', {}).get('method', 'N/A')}")
                else:
                    st.info("Данные еще не стандартизованы")

            # Показываем сравнение если стандартизация применена
            if st.session_state.get('scaling_applied', False):
                st.markdown("---")
                st.markdown("**Статистика после стандартизации:**")
                scaled_stats = st.session_state.regression_df[columns_to_scale].describe()
                st.dataframe(scaled_stats)

                st.markdown("**Сравнение до/после:**")
                # Это показать не можем, т.к. исходные данные уже заменены
                # Но можем показать что mean близок к 0 и std близок к 1 для StandardScaler
                if scaling_method == "StandardScaler (Z-score)":
                    st.info("""
                    После StandardScaler:
                    - Mean должен быть близок к 0
                    - Std должен быть близок к 1
                    """)
                elif scaling_method == "MinMaxScaler (0-1)":
                    st.info("""
                    После MinMaxScaler:
                    - Min = 0
                    - Max = 1
                    """)

    # ========== Вкладка 6: Создание признаков ==========
    with tabs[5]:
        st.subheader("Инженерия признаков")

        # ===== РАЗДЕЛ 1: Анализ оптимальных сдвигов =====
        st.markdown("## 1️⃣ Анализ оптимальных сдвигов")
        st.info("""
        **Зачем нужен сдвиг?** В промышленных процессах показания сенсоров могут влиять на качество продукта
        с задержкой (например, температура 2 часа назад влияет на качество сейчас).
        Этот инструмент помогает найти оптимальный сдвиг для каждого сенсора.
        """)

        if not target_col:
            st.warning("Выберите целевую переменную в боковой панели")
        else:
            available_sensors = [col for col in st.session_state.regression_df.columns
                               if col != target_col and '_lag_' not in col and '_rolling_' not in col and '_shift_' not in col]

            if available_sensors:
                # Режим работы
                shift_mode = st.radio(
                    "Режим анализа",
                    options=["Анализ одного сенсора", "Анализ всех сенсоров (тепловая карта)", "Автоматическое создание"]
                )

                if shift_mode == "Анализ одного сенсора":
                    sensor_to_analyze = st.selectbox(
                        "Выберите сенсор для анализа",
                        options=available_sensors
                    )

                    max_shift = st.slider(
                        "Максимальный сдвиг для анализа",
                        min_value=1,
                        max_value=50,
                        value=24,
                        help="Количество периодов (строк) для проверки"
                    )

                    if st.button("Провести анализ сдвига"):
                        with st.spinner(f"Анализ сдвига для {sensor_to_analyze}..."):
                            # Анализ влияния сдвига
                            shift_analysis = visualize_shift_analysis_data(
                                st.session_state.regression_df,
                                sensor_to_analyze,
                                target_col,
                                max_shift=max_shift
                            )

                            # Находим оптимальный сдвиг
                            optimal_result = find_optimal_shift(
                                st.session_state.regression_df,
                                sensor_to_analyze,
                                target_col,
                                max_shift=max_shift
                            )

                            optimal_shift = optimal_result['optimal_shift']
                            best_corr = optimal_result['best_score']

                            # Показываем результаты
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Оптимальный сдвиг", f"{optimal_shift} периодов")
                            with col2:
                                st.metric("Корреляция", f"{best_corr:.4f}")
                            with col3:
                                original_corr = optimal_result['shift_scores'].get(0, 0)
                                improvement = ((best_corr - original_corr) / max(abs(original_corr), 0.001)) * 100
                                st.metric("Улучшение", f"{improvement:.1f}%")

                            # График корреляции vs сдвиг
                            st.markdown("**График зависимости корреляции от сдвига:**")
                            fig1 = plot_shift_correlation(shift_analysis, sensor_to_analyze, optimal_shift)
                            st.plotly_chart(fig1, use_container_width=True)

                            # Сравнение оригинального и сдвинутого сенсора
                            if optimal_shift > 0:
                                st.markdown("**Сравнение временных рядов:**")
                                fig2 = plot_shifted_sensor_comparison(
                                    st.session_state.regression_df,
                                    sensor_to_analyze,
                                    target_col,
                                    optimal_shift
                                )
                                st.plotly_chart(fig2, use_container_width=True)

                            # Таблица с детальными результатами
                            st.markdown("**Детальная статистика по сдвигам:**")
                            st.dataframe(shift_analysis.head(10))

                elif shift_mode == "Анализ всех сенсоров (тепловая карта)":
                    sensors_to_analyze = st.multiselect(
                        "Выберите сенсоры для анализа",
                        options=available_sensors,
                        default=available_sensors[:min(5, len(available_sensors))]
                    )

                    max_shift = st.slider(
                        "Максимальный сдвиг",
                        min_value=1,
                        max_value=50,
                        value=24,
                        key="heatmap_max_shift"
                    )

                    if st.button("Построить тепловую карту") and sensors_to_analyze:
                        with st.spinner("Построение тепловой карты..."):
                            fig = plot_multiple_shifts_heatmap(
                                st.session_state.regression_df,
                                sensors_to_analyze,
                                target_col,
                                max_shift=max_shift
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # Таблица с оптимальными сдвигами
                            st.markdown("**Оптимальные сдвиги для каждого сенсора:**")
                            optimal_shifts = []
                            for sensor in sensors_to_analyze:
                                result = find_optimal_shift(
                                    st.session_state.regression_df,
                                    sensor,
                                    target_col,
                                    max_shift=max_shift
                                )
                                optimal_shifts.append({
                                    'Сенсор': sensor,
                                    'Оптимальный сдвиг': result['optimal_shift'],
                                    'Корреляция': f"{result['best_score']:.4f}"
                                })

                            df_optimal = pd.DataFrame(optimal_shifts)
                            df_optimal = df_optimal.sort_values('Корреляция', ascending=False)
                            st.dataframe(df_optimal, hide_index=True)

                else:  # Автоматическое создание
                    st.info("""
                    Система автоматически найдет оптимальные сдвиги для выбранных сенсоров
                    и создаст новые признаки с этими сдвигами.
                    """)

                    sensors_for_shift = st.multiselect(
                        "Выберите сенсоры для автоматического сдвига",
                        options=available_sensors,
                        help="Для каждого сенсора будет найден оптимальный сдвиг"
                    )

                    max_shift_auto = st.slider(
                        "Максимальный сдвиг для поиска",
                        min_value=1,
                        max_value=50,
                        value=24,
                        key="auto_shift_max"
                    )

                    if st.button("Создать сдвинутые признаки автоматически") and sensors_for_shift:
                        with st.spinner("Поиск оптимальных сдвигов и создание признаков..."):
                            df_with_shifts, shift_info = create_shifted_features(
                                st.session_state.regression_df,
                                sensors_for_shift,
                                target_col,
                                auto_find_shifts=True,
                                max_shift=max_shift_auto
                            )

                            st.session_state.regression_df = df_with_shifts

                            # Показываем результаты
                            st.success(f"Добавлено {sum(1 for s in shift_info.values() if s['optimal_shift'] > 0)} сдвинутых признаков")

                            # Таблица с результатами
                            st.markdown("**Найденные оптимальные сдвиги:**")
                            results = []
                            for sensor, info in shift_info.items():
                                results.append({
                                    'Сенсор': sensor,
                                    'Оптимальный сдвиг': info['optimal_shift'],
                                    'Корреляция': f"{info['correlation']:.4f}",
                                    'Новый признак': f"{sensor}_shift_{info['optimal_shift']}" if info['optimal_shift'] > 0 else "-"
                                })

                            df_results = pd.DataFrame(results)
                            st.dataframe(df_results, hide_index=True)

                            st.markdown("**Обновленный датасет:**")
                            st.dataframe(st.session_state.regression_df.head(10))

        st.markdown("---")

        # ===== РАЗДЕЛ 2: Лаг-признаки =====
        st.markdown("## 2️⃣ Лаг-признаки (ручное создание)")
        st.info("Лаг-признаки позволяют использовать прошлые значения сенсоров для предсказания")

        col1, col2 = st.columns(2)
        with col1:
            lag_sensors = st.multiselect(
                "Выберите сенсоры для создания лагов",
                options=[col for col in st.session_state.regression_df.columns if col != target_col],
                help="Для каких сенсоров создать лаг-признаки"
            )

        with col2:
            lag_periods = st.multiselect(
                "Периоды лагов",
                options=list(range(1, 25)),
                default=[1, 2, 3],
                help="На сколько шагов назад создать признаки (например, 1 = предыдущее значение)"
            )

        if st.button("Создать лаг-признаки") and lag_sensors and lag_periods:
            df_with_lags = create_lag_features(
                st.session_state.regression_df,
                columns=lag_sensors,
                lags=lag_periods
            )
            st.session_state.regression_df = df_with_lags
            st.success(f"Добавлено {len(lag_sensors) * len(lag_periods)} лаг-признаков!")
            st.dataframe(st.session_state.regression_df.head())

        st.markdown("---")

        # ===== РАЗДЕЛ 3: Rolling-признаки =====
        st.markdown("## 3️⃣ Rolling-признаки (скользящие окна)")

        col1, col2, col3 = st.columns(3)
        with col1:
            rolling_sensors = st.multiselect(
                "Выберите сенсоры для rolling",
                options=[col for col in st.session_state.regression_df.columns if col != target_col and '_lag_' not in col],
                key="rolling_sensors"
            )

        with col2:
            rolling_window = st.number_input("Размер окна", min_value=2, max_value=50, value=5)

        with col3:
            rolling_funcs = st.multiselect(
                "Функции агрегации",
                options=['mean', 'std', 'min', 'max'],
                default=['mean']
            )

        if st.button("Создать rolling-признаки") and rolling_sensors and rolling_funcs:
            df_with_rolling = create_rolling_features(
                st.session_state.regression_df,
                columns=rolling_sensors,
                windows=rolling_window,
                funcs=rolling_funcs
            )
            st.session_state.regression_df = df_with_rolling
            st.success(f"Добавлено {len(rolling_sensors) * len(rolling_funcs)} rolling-признаков!")
            st.dataframe(st.session_state.regression_df.head())

        # ===== РАЗДЕЛ 4: Временная агрегация (для редких измерений) =====
        st.markdown("---")
        st.markdown("## 4️⃣ Временная агрегация (для данных с разной частотой)")

        # Проверка что есть целевая переменная
        if not target_col:
            st.warning("Выберите целевую переменную в боковой панели")
        else:
            # Импорт функций агрегации
            try:
                from dashboard.regression.time_aggregation import (
                    aggregate_by_target_timestamps,
                    find_optimal_aggregation_params,
                    get_target_measurements_info,
                    analyze_feature_frequency
                )

                # Анализ частоты измерений
                st.markdown("### 📊 Анализ частоты измерений")

                if st.button("Показать анализ частоты"):
                    with st.spinner("Анализ данных..."):
                        # Анализ для целевой переменной
                        target_info = get_target_measurements_info(st.session_state.regression_df, target_col)

                        st.markdown(f"**Целевая переменная: `{target_col}`**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Измерений", target_info['count'])
                        with col2:
                            avg_h = target_info.get('avg_interval_hours')
                            st.metric("Средний интервал", f"{avg_h:.1f} ч" if avg_h else "N/A")
                        with col3:
                            avg_d = avg_h / 24 if avg_h else 0
                            st.metric("Средний интервал (дни)", f"{avg_d:.1f}" if avg_h else "N/A")
                        with col4:
                            suggested = target_info.get('suggested_window_hours', 24)
                            st.metric("Рекомендуемое окно", f"{suggested} ч")

                        # Анализ всех признаков
                        st.markdown("**Частота измерений всех признаков:**")
                        freq_analysis = analyze_feature_frequency(st.session_state.regression_df)
                        st.dataframe(freq_analysis, use_container_width=True)

                # Параметры агрегации
                st.markdown("### ⚙️ Параметры агрегации")

                col1, col2 = st.columns(2)
                with col1:
                    window_hours = st.number_input(
                        "Окно усреднения (часы)",
                        min_value=1,
                        max_value=720,  # 30 дней
                        value=24,
                        help="Сколько часов данных использовать для усреднения ДО каждого измерения цели"
                    )

                    agg_funcs = st.multiselect(
                        "Функции агрегации",
                        options=['mean', 'std', 'min', 'max', 'last', 'first', 'median', 'count'],
                        default=['mean', 'std', 'min', 'max'],
                        help="Статистики для расчета по каждому признаку"
                    )

                with col2:
                    shift_hours = st.number_input(
                        "Временной сдвиг (часы)",
                        min_value=0,
                        max_value=72,
                        value=1,
                        help="Сдвиг окна назад (компенсация задержки отбора пробы)"
                    )

                    auto_find = st.checkbox(
                        "Автоматически найти оптимальные параметры",
                        value=False,
                        help="Перебрать разные window/shift и выбрать лучшие по корреляции"
                    )

                # Кнопка создания агрегированного датасета
                if auto_find:
                    if st.button("🔍 Найти оптимальные параметры и создать датасет"):
                        with st.spinner("Поиск оптимальных параметров..."):
                            result = find_optimal_aggregation_params(
                                st.session_state.regression_df,
                                target_col
                            )

                            st.success(f"Найдены оптимальные параметры!")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Окно", f"{result['optimal_window_hours']} ч")
                            with col2:
                                st.metric("Сдвиг", f"{result['optimal_shift_hours']} ч")
                            with col3:
                                st.metric("Корреляция", f"{result['best_score']:.4f}")

                            # Создаем датасет с оптимальными параметрами
                            df_aggregated = aggregate_by_target_timestamps(
                                st.session_state.regression_df,
                                target_col,
                                window_hours=result['optimal_window_hours'],
                                shift_hours=result['optimal_shift_hours'],
                                agg_functions=agg_funcs
                            )

                            # Заменяем датасет
                            st.session_state.regression_df = df_aggregated
                            st.success(f"✅ Создан агрегированный датасет: {len(df_aggregated)} строк × {len(df_aggregated.columns)} колонок")
                            st.dataframe(df_aggregated.head())
                else:
                    if st.button("📊 Создать агрегированный датасет"):
                        with st.spinner(f"Агрегация с окном {window_hours}ч и сдвигом {shift_hours}ч..."):
                            df_aggregated = aggregate_by_target_timestamps(
                                st.session_state.regression_df,
                                target_col,
                                window_hours=window_hours,
                                shift_hours=shift_hours,
                                agg_functions=agg_funcs
                            )

                            # Заменяем датасет
                            st.session_state.regression_df = df_aggregated
                            st.success(f"✅ Создан агрегированный датасет: {len(df_aggregated)} строк × {len(df_aggregated.columns)} колонок")
                            st.dataframe(df_aggregated.head())

                            # Статистика
                            st.markdown("**Статистика после агрегации:**")
                            st.dataframe(df_aggregated.describe())

            except ImportError as e:
                st.error(f"Ошибка импорта модуля time_aggregation: {e}")

    # ========== Вкладка 7: Отбор признаков ==========
    with tabs[6]:
        st.subheader("Отбор важных признаков")

        available_features = [col for col in st.session_state.regression_df.columns if col != target_col]

        if not available_features:
            st.warning("Нет доступных признаков для отбора")
        else:
            selection_method = st.selectbox(
                "Метод отбора признаков",
                options=[
                    "Корреляция",
                    "Random Forest (важность)",
                    "Турнирная валидация (несколько методов)"
                ]
            )

            if selection_method == "Корреляция":
                threshold = st.slider("Минимальная корреляция с целевой переменной", 0.0, 1.0, 0.3, 0.01)

                if st.button("Выполнить отбор"):
                    selected, correlations = select_by_correlation(
                        st.session_state.regression_df,
                        target_col,
                        threshold=threshold
                    )
                    st.session_state.regression_features = selected
                    st.success(f"Отобрано {len(selected)} признаков")
                    st.write("Выбранные признаки:", selected)

            elif selection_method == "Random Forest (важность)":
                top_n = st.slider("Количество топовых признаков", 1, min(50, len(available_features)), 10)

                if st.button("Выполнить отбор"):
                    with st.spinner("Обучение Random Forest..."):
                        selected, importances = select_by_rfr(
                            st.session_state.regression_df,
                            target_col,
                            n_features=top_n
                        )
                        st.session_state.regression_features = selected
                        st.success(f"Отобрано {len(selected)} признаков")
                        st.write("Выбранные признаки:", selected)

            else:  # Турнирная валидация
                min_consensus = st.slider(
                    "Минимальное согласие методов",
                    1, 3, 2,
                    help="Сколько методов должны выбрать признак для включения"
                )

                if st.button("Выполнить турнирную валидацию"):
                    with st.spinner("Выполнение турнирной валидации..."):
                        tournament_results = tournament_feature_selection(
                            st.session_state.regression_df,
                            target_col
                        )

                        consensus = get_consensus_features(tournament_results, min_consensus=min_consensus)
                        st.session_state.regression_features = consensus

                        st.success(f"Отобрано {len(consensus)} признаков с согласием >= {min_consensus}")

                        # Показать результаты каждого метода
                        st.markdown("**Результаты по методам:**")
                        for method, features in tournament_results.items():
                            st.write(f"- {method}: {len(features)} признаков")

                        st.write("Итоговые признаки:", consensus)

            # Показать текущий выбор
            if st.session_state.regression_features:
                st.markdown("---")
                st.markdown("**Текущие выбранные признаки:**")
                st.write(st.session_state.regression_features)

                # Возможность ручного редактирования
                # Фильтруем только те признаки, которые еще существуют
                # Конвертируем в списки для безопасного сравнения
                current_features = st.session_state.regression_features

                # Если это не список, делаем список
                if not isinstance(current_features, list):
                    current_features = list(current_features)

                # Распаковываем вложенные списки и фильтруем только строки
                flat_features = []
                for item in current_features:
                    if isinstance(item, str):
                        flat_features.append(item)
                    elif isinstance(item, (list, tuple)):
                        flat_features.extend([x for x in item if isinstance(x, str)])

                available_features_set = set(available_features)
                valid_defaults = [f for f in flat_features if f in available_features_set]

                manual_features = st.multiselect(
                    "Редактировать список признаков вручную",
                    options=available_features,
                    default=valid_defaults
                )
                if st.button("Применить ручной выбор"):
                    st.session_state.regression_features = manual_features
                    st.success("Список признаков обновлен")
                    st.rerun()

    # ========== Вкладка 8: Обучение моделей ==========
    with tabs[7]:
        st.subheader("Обучение регрессионных моделей")

        if not st.session_state.regression_features:
            st.warning("Сначала выберите признаки на вкладке 'Отбор признаков'")
        else:
            # Получение списка доступных моделей
            models_config = get_regression_models()

            if not models_config:
                st.error("Не удалось загрузить список моделей из API")
                st.info("Проверьте что API запущен на http://localhost:8000")
            else:
                # Фильтруем только регрессионные модели
                regression_models = {k: v for k, v in models_config.items() if v.get('type') == 'regression'}

                if not regression_models:
                    st.error("Не найдено регрессионных моделей в конфигурации API")
                    st.write("**Debug:** Полученные модели:", list(models_config.keys()))
                    st.write("**Debug:** Модели с типами:", {k: v.get('type') for k, v in models_config.items()})
                    return

                training_mode = st.radio(
                    "Режим обучения",
                    options=["Одна модель", "Турнирная валидация (все модели)"]
                )

                if training_mode == "Одна модель":
                    model_type = st.selectbox(
                        "Выберите модель",
                        options=list(regression_models.keys())
                    )

                    if model_type is None:
                        st.info("Выберите модель для обучения")
                        return

                    # Выбор режима настройки гиперпараметров
                    use_auto_tune = st.checkbox(
                        "Использовать автоподбор гиперпараметров (Grid Search)",
                        value=False,
                        help="GridSearchCV автоматически найдет лучшие параметры (займет больше времени)"
                    )

                    # Настройка гиперпараметров
                    if use_auto_tune:
                        st.info("Автоподбор будет искать оптимальные параметры в предопределенных диапазонах")

                        # Показываем какие диапазоны будут использованы
                        param_grids = {
                            "linear_regression": {"fit_intercept": [True, False]},
                            "random_forest": {
                                "n_estimators": [50, 100, 200],
                                "max_depth": [None, 10, 20, 30],
                                "min_samples_split": [2, 5, 10]
                            },
                            "gradient_boosting": {
                                "n_estimators": [50, 100, 200],
                                "learning_rate": [0.01, 0.1, 0.3],
                                "max_depth": [3, 5, 7]
                            },
                            "elastic_net": {
                                "alpha": [0.1, 1.0, 10.0],
                                "l1_ratio": [0.1, 0.5, 0.9]
                            },
                            "lightgbm": {
                                "n_estimators": [50, 100, 200],
                                "learning_rate": [0.01, 0.1, 0.3],
                                "max_depth": [3, 5, 7, -1],
                                "num_leaves": [15, 31, 63]
                            }
                        }

                        st.write("**Диапазоны параметров для поиска:**")
                        st.json(param_grids.get(model_type, {}))
                        hyper_params = param_grids.get(model_type, {})

                    else:
                        st.markdown("**Гиперпараметры:**")
                        hyper_params = {}
                        model_config = regression_models[model_type]

                        for param_name, param_info in model_config.get('hyper_params', {}).items():
                            dtype = param_info.get('dtype')
                            default = param_info.get('default_value')
                            description = param_info.get('description', '')

                            if dtype == 'int':
                                min_val = param_info.get('min_value', 1)
                                max_val = param_info.get('max_value', 1000)
                                if default is None:
                                    default = (min_val + max_val) // 2
                                hyper_params[param_name] = st.number_input(
                                    f"{param_name} ({description})",
                                    min_value=min_val,
                                    max_value=max_val,
                                    value=default,
                                    help=description
                                )
                            elif dtype == 'float':
                                min_val = param_info.get('min_value', 0.0)
                                max_val = param_info.get('max_value', 10.0)
                                if default is None:
                                    default = (min_val + max_val) / 2
                                hyper_params[param_name] = st.number_input(
                                    f"{param_name} ({description})",
                                    min_value=min_val,
                                    max_value=max_val,
                                    value=float(default),
                                    step=0.01,
                                    help=description
                                )
                            elif dtype == 'bool':
                                hyper_params[param_name] = st.checkbox(
                                    f"{param_name} ({description})",
                                    value=default if default is not None else False,
                                    help=description
                                )

                    if st.button("Обучить модель"):
                        spinner_text = f"Автоподбор параметров для {model_type}..." if use_auto_tune else f"Обучение модели {model_type}..."
                        with st.spinner(spinner_text):
                            result = train_regression_model(
                                df=st.session_state.regression_df,
                                target_col=target_col,
                                feature_cols=st.session_state.regression_features,
                                model_type=model_type,
                                hyper_params=hyper_params,
                                test_size=test_size,
                                use_auto_tune=use_auto_tune
                            )

                            if 'error' in result:
                                st.error(f"Ошибка обучения: {result['error']}")
                            else:
                                st.session_state.regression_results = {model_type: result}
                                st.success("✅ Модель обучена успешно! Перейдите на вкладку **'📊 Результаты'** для просмотра результатов.")
                                st.rerun()

                else:  # Турнирная валидация
                    st.info("Будут обучены все доступные регрессионные модели с параметрами по умолчанию")

                    if st.button("Запустить турнирную валидацию"):
                        # Собираем гиперпараметры по умолчанию для каждой модели
                        hyper_params_dict = {}
                        for model_name, model_config in regression_models.items():
                            params = {}
                            for param_name, param_info in model_config.get('hyper_params', {}).items():
                                default = param_info.get('default_value')
                                if default is not None:
                                    params[param_name] = default
                            hyper_params_dict[model_name] = params

                        with st.spinner("Обучение моделей..."):
                            results = train_multiple_regression_models(
                                df=st.session_state.regression_df,
                                target_col=target_col,
                                feature_cols=st.session_state.regression_features,
                                model_types=list(regression_models.keys()),
                                hyper_params_dict=hyper_params_dict,
                                test_size=test_size
                            )

                            st.session_state.regression_results = results
                            st.success("✅ Турнирная валидация завершена! Перейдите на вкладку **'📊 Результаты'** для просмотра сравнения моделей.")
                            st.rerun()

    # ========== Вкладка 9: Результаты ==========
    with tabs[8]:
        st.subheader("Результаты моделирования")

        if not st.session_state.regression_results:
            st.info("Обучите модели на вкладке 'Обучение моделей' для отображения результатов")
        else:
            results = st.session_state.regression_results

            # Если несколько моделей - показываем сравнение
            if len(results) > 1:
                st.markdown("### Сравнение моделей")

                # Таблица метрик
                fig_table = plot_metrics_comparison_table(results)
                st.plotly_chart(fig_table, use_container_width=True)

                # График сравнения по метрике
                metric_to_compare = st.selectbox(
                    "Метрика для сравнения",
                    options=['test_r2', 'test_mae', 'test_rmse']
                )
                fig_comparison = plot_model_comparison(results, metric=metric_to_compare)
                st.plotly_chart(fig_comparison, use_container_width=True)

                st.markdown("---")

            # Детальные результаты для каждой модели
            st.markdown("### Детальные результаты")

            for model_name, result in results.items():
                if 'error' in result:
                    st.error(f"**{model_name}**: {result['error']}")
                    continue

                with st.expander(f"**{model_name}**", expanded=len(results)==1):
                    model_params = result.get('model_params', {})
                    metrics = model_params.get('metrics', {})

                    # Метрики
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Train R²", f"{metrics.get('train_r2', 0):.4f}")
                    with col2:
                        st.metric("Test R²", f"{metrics.get('test_r2', 0):.4f}")
                    with col3:
                        st.metric("MAE", f"{metrics.get('test_mae', 0):.4f}")
                    with col4:
                        st.metric("MSE", f"{metrics.get('test_mse', 0):.4f}")
                    with col5:
                        st.metric("RMSE", f"{metrics.get('test_rmse', 0):.4f}")

                    # Парсинг предсказаний
                    try:
                        df_predictions = pd.read_json(StringIO(result.get('df_predictions', '{}')), orient='table')

                        # Графики
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**Фактические vs Предсказанные**")
                            fig1 = plot_actual_vs_predicted(df_predictions, model_name)
                            st.plotly_chart(fig1, use_container_width=True)

                        with col2:
                            st.markdown("**Анализ остатков**")
                            fig2 = plot_residuals(df_predictions, model_name)
                            st.plotly_chart(fig2, use_container_width=True)

                        # Timeline если есть индекс
                        if isinstance(df_predictions.index, pd.DatetimeIndex):
                            st.markdown("**Временной ряд предсказаний**")
                            fig3 = plot_prediction_timeline(df_predictions, model_name)
                            st.plotly_chart(fig3, use_container_width=True)

                        # Feature importance
                        feature_importance = model_params.get('feature_importance', {})
                        if feature_importance:
                            st.markdown("**Важность признаков**")
                            fig4 = plot_feature_importance(feature_importance, model_name)
                            st.plotly_chart(fig4, use_container_width=True)

                        # Формула (если есть)
                        formula = model_params.get('formula')
                        if formula:
                            st.markdown("**Формула модели:**")
                            st.code(formula, language="python")

                        # Таблица предсказаний
                        st.markdown("**Таблица предсказаний:**")
                        st.dataframe(df_predictions.head(50))

                    except Exception as e:
                        st.error(f"Ошибка при отображении результатов: {str(e)}")
