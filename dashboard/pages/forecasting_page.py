import streamlit as st
import pandas as pd
import asyncio
from dashboard.data_processing.info_about_dataframe import info_about_dataframe
from dashboard.data_processing.select_time_interval import start_date, end_date, filter_dataframe
from dashboard.visualization.plot_interactive_with_selection import plot_interactive_with_selection
from dashboard.request_functions.create_model_payload_func import create_model_payload
from dashboard.request_functions.model_request_func import get_prediction
from dashboard.request_functions.get_models_func import get_models
from dashboard.request_functions.metrics_request_func import get_metrics
from dashboard.request_functions.create_metrics_payload_func import create_metrics_payload
from dashboard.utils.data_limiting import limit_data_to_last_points, get_default_time_range
from typing import Optional, List
import json
from dashboard.data_processing.edit_dataset_by_quantiles import edit_dataset_by_quantiles
import os
from io import StringIO
import numpy as np

from ..schemas import ModelRequest, MetricsRequest


def render_forecast_results():
    """
    Отображает результаты прогнозирования, если они есть в session_state.
    Показывает таблицы, графики и метрики в зависимости от наличия df_test и длины.
    """
    # Проверяем, есть ли данные для отображения
    filtered_df = st.session_state.get('filtered_df')
    selected_sensors = st.session_state.get('selected_sensors', [])
    
    if filtered_df is None or filtered_df.empty or not selected_sensors:
        return  # Не отображаем результаты, если данные пустые или выделение очищено
    
    forecast_result = st.session_state.get('forecast_result')
    metrics_result = st.session_state.get('metrics_result')
    df_test = st.session_state.get('df_test')
    duration = st.session_state.get('duration')
    target_sensor = st.session_state.get('target_sensor')
    if forecast_result is not None:
        st.markdown('---')
        st.markdown('### Результаты прогнозирования')
        try:
            df_predict_json = forecast_result.get("df_predict")
            if df_predict_json is None or df_predict_json == '' or df_predict_json == 'null':
                st.error('Ошибка: Предсказание невозможно. Значения датчиков сложно предсказуемы либо неправильно настроены параметры')
                return
            df_predict = pd.read_json(StringIO(df_predict_json), orient='table')
            # Проверяем, содержит ли DataFrame только NaN значения
            all_nan = df_predict.isna().all()
            if isinstance(all_nan, bool):
                only_nan = all_nan
            else:
                only_nan = all_nan.all()
            if only_nan:
                st.error('Ошибка: Предсказание невозможно. Значения датчиков сложно предсказуемы либо неправильно настроены параметры')
                return
            # Приводим имя столбца к 'sensor', если нужно
            if 'sensor' not in df_predict.columns and len(df_predict.columns) == 1:
                df_predict = df_predict.rename(columns={str(df_predict.columns[0]): 'sensor'})
        except Exception as e:
            st.error(f"Ошибка при чтении предсказанных значений: {str(e)}")
            return
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('**Предсказанные значения:**')
            st.dataframe(df_predict)
        # Аналогично для df_test
        if df_test is not None and isinstance(df_test, pd.DataFrame) and len(df_test) == len(df_predict):
            if 'sensor' not in df_test.columns and len(df_test.columns) == 1:
                df_test = df_test.rename(columns={str(df_test.columns[0]): 'sensor'})
            with col2:
                st.markdown('**Истинные значения:**')
                st.dataframe(df_test)
        # График на всю ширину
        st.markdown('**График прогнозирования:**')
        import plotly.graph_objs as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_predict.index,
            y=df_predict['sensor'],
            mode='lines+markers',
            name='Прогноз'
        ))
        if df_test is not None and isinstance(df_test, pd.DataFrame) and len(df_test) == len(df_predict):
            fig.add_trace(go.Scatter(
                x=df_test.index,
                y=df_test['sensor'],
                mode='lines+markers',
                name='Истинные значения'
            ))
        fig.update_layout(
            xaxis_title='Время',
            yaxis_title=str(target_sensor) if target_sensor else 'Значение',
            legend_title='Легенда',
            width=1200,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        # Метрики и гиперпараметры
        if metrics_result is not None and df_test is not None and len(df_test) == len(df_predict):
            col_metrics, col_hyperparams = st.columns(2)
            with col_metrics:
                st.markdown('### Метрики качества прогноза')
                st.json(metrics_result)
            with col_hyperparams:
                st.markdown('### Гиперпараметры модели')
                hyper_params = forecast_result.get("hyper_params", {})
                if hyper_params:
                    st.json(hyper_params)
                else:
                    st.info("Гиперпараметры не найдены")
        elif df_test is None or len(df_predict) > (len(df_test) if df_test is not None else 0):
            st.info('Недостаточно тестовых данных для расчёта метрик. Отображаются только предсказанные значения.')
            # Показываем только гиперпараметры
            st.markdown('### Гиперпараметры модели')
            hyper_params = forecast_result.get("hyper_params", {})
            if hyper_params:
                st.json(hyper_params)
            else:
                st.info("Гиперпараметры не найдены")

def render_data_overview(df: pd.DataFrame, outlier_percentage: float) -> None:
    """
    Отображает верхнюю панель с общей информацией о данных
    """
    top_cols = st.columns([2, 2, 2, 2, 2])
    features_size, tuples_size, first_tuple, last_tuple = info_about_dataframe(df)
    with top_cols[0]:
        st.markdown(f"Кол-во записей: {tuples_size if tuples_size is not None else 'Нет информации'}")
    with top_cols[1]:
        st.markdown(f"Количество признаков: {features_size if features_size is not None else 'Нет информации'}")
    with top_cols[2]:
        st.markdown(f"Первая запись: {first_tuple if first_tuple is not None else 'Нет информации'}")
    with top_cols[3]:
        st.markdown(f"Последняя запись: {last_tuple if last_tuple is not None else 'Нет информации'}")
    with top_cols[4]:
        st.markdown(f"Количество выбросов: {f'{outlier_percentage}% от всех значений' if outlier_percentage is not None else 'Нет информации'}")

def render_forecasting_main_panel(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Основная панель: фильтрация, интерактивный график, предпросмотр, параметры, панель управления прогнозом справа
    Возвращает training_df для передачи в панель управления прогнозом
    """
    main_cols = st.columns([9, 3])
    training_df = None
    with main_cols[0]:
        if df is not None and not df.empty:
            filtered_df = st.session_state.get('filtered_df', df)
            selected_sensors = st.session_state.get('selected_sensors', df.columns.tolist())
            if selected_sensors:
                training_df = plot_interactive_with_selection(filtered_df, selected_sensors=selected_sensors, flag=True)
            else:
                st.error("Ошибка: Выберите хотя бы один параметр для отображения графика.")
        else:
            st.markdown("""<div class=\"block\" style=\"height: 420px;\"></div>""", unsafe_allow_html=True)
        lower_cols = st.columns([6, 6])
        with lower_cols[1]:
            st.markdown("#### Параметры:")
            if df is not None and not df.empty:
                sensor_df = pd.DataFrame({
                    'Датчики': df.columns,
                    'Отображать': [col in st.session_state.get('selected_sensors', df.columns.tolist()) for col in df.columns]
                })
                edited_sensor_df = st.data_editor(sensor_df, key="sensor_selector")
                st.session_state['sensor_editor_temp'] = edited_sensor_df[edited_sensor_df['Отображать']]['Датчики'].tolist()
            else:
                st.markdown("Нет информации", unsafe_allow_html=True)
        with lower_cols[0]:
            st.markdown("#### Предпросмотр:")
            filtered_df = st.session_state.get('filtered_df', df)
            if filtered_df is not None:
                st.dataframe(filtered_df)
            else:
                st.markdown("Нет информации", unsafe_allow_html=True)
    # Добавляем отображение результатов прогнозирования на всю ширину
    render_forecast_results()
    with main_cols[1]:
        render_forecasting_control_panel(df, training_df)
    return training_df

def show_training_data_dialog(training_df: Optional[pd.DataFrame]) -> None:
    @st.dialog("Тренировочные данные")
    def show_training_data():
        if training_df is not None and not training_df.empty:
            st.dataframe(training_df, height=300)
        else:
            st.markdown("Нет данных для отображения", unsafe_allow_html=True)
        if st.button("Выйти", key="exit"):
            st.rerun()
    if training_df is not None and not training_df.empty:
        if st.button("Отобразить тренировочные данные"):
            show_training_data()

def get_api_url():
    """Получает URL API из секретов Streamlit"""
    return st.secrets.get("api_url")

def render_forecasting_control_panel(df: pd.DataFrame, training_df: Optional[pd.DataFrame]) -> None:
    """
    Панель управления прогнозированием.
    Здесь находятся кнопки изменения исходного ряда, кнопка возврата исходного ряда,
    выбор временного интервала, целевого признака и модели, а также запуск прогноза.
    """
    st.markdown("## Прогнозирование")

    # Кнопки изменения исходного ряда располагаются сразу под заголовком панели справа от основного контента.
    btn_cols = st.columns([1, 1])
    with btn_cols[0]:
        if st.button("Изменить исходный ряд", key="edit_base_series_btn"):
            @st.dialog("Изменить исходный ряд")
            def edit_series_dialog():
                # Берём текущую рабочую версию данных. Все преобразования применяются к ней.
                work_df = st.session_state.get('working_df', df)

                numeric_cols = list(work_df.select_dtypes(include="number").columns)
                if not numeric_cols:
                    st.error("Нет числовых столбцов для редактирования")
                    if st.button("Закрыть"):
                        st.rerun()
                    return

                # Пользователь выбирает столбцы и параметры преобразования
                cols = st.multiselect("Столбцы для обработки", options=numeric_cols, default=numeric_cols, key="edit_cols")
                lower_q = st.slider("Нижний квантиль", 0.0, 1.0, 0.05, 0.01, key="edit_lower_q")
                upper_q = st.slider("Верхний квантиль", 0.0, 1.0, 0.95, 0.01, key="edit_upper_q")
                mode_human = st.radio("Режим", ["Обрезать значения по квантилям", "Фильтровать строки по квантилям"], horizontal=True, key="edit_mode")
                how_human = st.radio("Логика для нескольких столбцов", ["Все столбцы внутри диапазона", "Достаточно одного столбца"], horizontal=True, key="edit_how")
                inclusive = st.selectbox("Включение границ", ["both", "left", "right", "neither"], index=0, help="both соответствует включению обеих границ", key="edit_inclusive")

                # Подтверждение применения преобразования к рабочему датасету
                if st.button("Применить", key="apply_series_edit"):
                    try:
                        _, mod = edit_dataset_by_quantiles(
                            work_df,
                            lower_q=lower_q,
                            upper_q=upper_q,
                            cols=cols if cols else numeric_cols,
                            mode="clip" if mode_human.startswith("Обрезать") else "filter",
                            how="all" if how_human.startswith("Все") else "any",
                            inclusive=inclusive,
                        )
                        # Сохраняем изменённую версию как рабочую, чтобы графики и фильтры использовали именно её
                        st.session_state['working_df'] = mod
                        # Обновляем отображаемую часть согласно текущему режиму показа
                        if st.session_state.get('is_limited_view', False):
                            st.session_state['filtered_df'] = limit_data_to_last_points(mod, 500)
                        else:
                            st.session_state['filtered_df'] = mod
                        # Включаем флаг, чтобы показать кнопку возврата исходного ряда
                        st.session_state['is_series_modified'] = True
                        st.success("Исходный ряд обновлён. На графике отображается изменённая версия.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Не удалось применить изменения. Подробности: {e}")

                # Закрытие без применения
                if st.button("Отмена", key="cancel_series_edit"):
                    st.rerun()

            edit_series_dialog()

    with btn_cols[1]:
        # Кнопка возврата отображается только после любых изменений исходного ряда
        if st.session_state.get('is_series_modified', False):
            if st.button("Вернуть исходный ряд", key="restore_base_series_btn"):
                base = st.session_state.get('original_df', df)
                # Восстанавливаем рабочую версию из оригинала
                st.session_state['working_df'] = base.copy()
                # Обновляем отображаемую часть в зависимости от выбранного режима показа
                if st.session_state.get('is_limited_view', False):
                    st.session_state['filtered_df'] = limit_data_to_last_points(base, 500)
                else:
                    st.session_state['filtered_df'] = base
                # Сбрасываем флаг изменения ряда
                st.session_state['is_series_modified'] = False
                st.success("Вернули оригинальный ряд")
                st.rerun()

    # Информационный блок о текущем режиме показа данных. Логика прежняя.
    original_df = st.session_state.get('original_df', df)
    if original_df is not None and len(original_df) > 500:
        if st.session_state.get('is_limited_view', False):
            st.info(f"Отображаются последние 500 из {len(original_df)} записей. Используйте фильтр для просмотра других периодов.")
            if st.button("Отобразить все записи", key="show_all_data"):
                st.session_state['filtered_df'] = st.session_state.get('working_df', df)
                st.session_state['is_limited_view'] = False
                st.rerun()
        else:
            st.info(f"Отображаются все {len(original_df)} записей. Для лучшей производительности рекомендуется включать ограниченный вид.")
            if st.button("Отобразить последние 500 записей", key="show_limited_data"):
                limited_df = limit_data_to_last_points(st.session_state.get('working_df', df), 500)
                st.session_state['filtered_df'] = limited_df
                st.session_state['is_limited_view'] = True
                st.rerun()

    st.markdown("#### Рассматриваемый временной промежуток")

    # Счётчик нужен для независимого состояния виджетов дат при сбросе
    if 'reset_counter' not in st.session_state:
        st.session_state['reset_counter'] = 0
    context = f"display_panel_{st.session_state['reset_counter']}"

    # Даты берём из текущего отображаемого DataFrame
    filtered_df = st.session_state.get('filtered_df', df)

    start_cols = st.columns(2)
    with start_cols[0]:
        start_datetime = start_date(filtered_df, context=context)
    with start_cols[1]:
        end_datetime = end_date(filtered_df, context=context)

    # Применение и сброс фильтра по времени
    button_cols = st.columns(2)
    with button_cols[0]:
        if st.button("Применить фильтр"):
            if start_datetime is not None and end_datetime is not None:
                # Фильтруем рабочую версию данных, чтобы учесть внесённые изменения ряда
                base_for_filter = st.session_state.get('working_df', df)
                filtered_result = filter_dataframe(start_datetime, end_datetime, base_for_filter)
                if filtered_result is not None:
                    st.session_state['filtered_df'] = filtered_result
                    st.session_state['is_limited_view'] = False
                else:
                    st.error("Ошибка при применении фильтра")
                    return
            else:
                # Если даты не заданы, показываем рабочий датасет целиком
                st.session_state['filtered_df'] = st.session_state.get('working_df', df)
                st.session_state['is_limited_view'] = False

            # Актуализируем список выбранных сенсоров под текущий набор колонок
            if 'sensor_editor_temp' in st.session_state:
                if st.session_state['sensor_editor_temp']:
                    st.session_state['selected_sensors'] = st.session_state['sensor_editor_temp']
                else:
                    st.error("Выберите хотя бы один параметр для отображения графика")
                    st.session_state['selected_sensors'] = []
            else:
                st.session_state['selected_sensors'] = st.session_state['filtered_df'].columns.tolist()
            st.rerun()

    with button_cols[1]:
        if st.button("Сбросить фильтр"):
            # Возвращаемся к рабочему датасету, чтобы сохранить эффект редактирования ряда
            work = st.session_state.get('working_df', df)
            if st.session_state.get('is_limited_view', False) and st.session_state.get('original_df') is not None:
                limited_df = limit_data_to_last_points(work, 500)
                st.session_state['filtered_df'] = limited_df
            else:
                st.session_state['filtered_df'] = work
            # Сбрасываем выбор сенсоров к колонкам текущего рабочего датасета
            st.session_state['selected_sensors'] = work.columns.tolist()
            st.session_state['sensor_editor_temp'] = work.columns.tolist()
            # Обнуляем состояние виджетов выбора дат и времени
            st.session_state['reset_counter'] = st.session_state.get('reset_counter', 0) + 1
            old_context = f"display_panel_{st.session_state['reset_counter'] - 1}"
            for key in [f'start_date_{old_context}', f'start_time_{old_context}', f'end_date_{old_context}', f'end_time_{old_context}']:
                if key in st.session_state:
                    del st.session_state[key]
            # Чистим ранее рассчитанные результаты прогноза и метрики
            for key in ['forecast_result', 'metrics_result', 'df_test', 'duration']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    st.markdown("#### Целевые параметры:")
    target_sensor = None
    exog_sensors = []
    if df is not None and not df.empty:
        # Источник доступных сенсоров зависит от текущего отображаемого набора колонок
        available_sensors = st.session_state.get('selected_sensors', st.session_state.get('filtered_df', df).columns.tolist())
        if training_df is not None and not training_df.empty:
            available_sensors = list(training_df.columns)
        if len(available_sensors) == 0:
            st.markdown("Нет доступных параметров для прогнозирования")
        else:
            target_sensor = st.selectbox(
                "Выберите целевой признак",
                options=available_sensors,
                index=0,
                key="target_sensor"
            )

            # Выбор дополнительных признаков (экзогенных переменных)
            if target_sensor:
                other_sensors = [s for s in available_sensors if s != target_sensor]
                if len(other_sensors) > 0:
                    st.multiselect(
                        "Дополнительные признаки (опционально)",
                        options=other_sensors,
                        default=[],
                        key="exog_sensors",
                        help="Выберите дополнительные датчики, которые могут помочь улучшить прогноз. Работает для SARIMA и Prophet."
                    )
    else:
        st.markdown("Нет информации", unsafe_allow_html=True)

    show_training_data_dialog(training_df)

    st.markdown("#### Выбрать модель")
    api_url = get_api_url()
    if not api_url:
        st.error("Не удалось получить URL API")
        return

    available_models = None
    try:
        models_response = asyncio.run(get_models(api_url))
        if models_response.status_code == 200:
            models_data = models_response.json().get('models', {})
            available_models = list(models_data.keys()) if models_data else None
    except Exception:
        pass

    if not available_models:
        st.error("Не удалось получить список моделей от API")
        return

    option = st.selectbox("Выберите модель", available_models, key="model_select")

    if st.button("Начать прогнозирование"):
        @st.dialog("Настройка прогноза")
        def forecast_settings_dialog():
            # Пользователь выбирает способ подбора параметров и горизонт прогнозирования
            auto_params = st.checkbox("Автоподбор параметров", key="auto_params_dialog")
            duration = st.number_input("Количество предсказаний", min_value=1, value=st.session_state.get('duration', 1), step=1, format="%d", key="duration_dialog")
            params = st.session_state.get('params', {})

            # Формы параметров для разных моделей
            if auto_params and option in ["sarima", "ets"]:
                if option == "sarima":
                    params['S'] = st.number_input("Сезонность S", min_value=1, value=params.get('S', 12), step=1, format="%d", key="seasonality_S")
                elif option == "ets":
                    params['seasonal_periods'] = st.number_input("Сезонность seasonal_periods", min_value=1, value=params.get('seasonal_periods', 12), step=1, format="%d", key="seasonality_ets")
            if not auto_params:
                if option == "sarima":
                    params['S'] = st.number_input("Сезонность S", min_value=1, value=params.get('S', 12), step=1, format="%d", key="sarima_S")
                    params['p'] = st.number_input("Порядок p", min_value=0, value=params.get('p', 0), step=1, format="%d", key="sarima_p")
                    params['d'] = st.number_input("Порядок d", min_value=0, value=params.get('d', 0), step=1, format="%d", key="sarima_d")
                    params['q'] = st.number_input("Порядок q", min_value=0, value=params.get('q', 0), step=1, format="%d", key="sarima_q")
                    params['P'] = st.number_input("Сезонный порядок P", min_value=0, value=params.get('P', 0), step=1, format="%d", key="sarima_P")
                    params['D'] = st.number_input("Сезонный порядок D", min_value=0, value=params.get('D', 0), step=1, format="%d", key="sarima_D")
                    params['Q'] = st.number_input("Сезонный порядок Q", min_value=0, value=params.get('Q', 0), step=1, format="%d", key="sarima_Q")
                elif option == "ets":
                    params['error_type'] = st.selectbox("Тип ошибки", ["add", "mul"], index=["add", "mul"].index(params.get('error_type', 'add')), key="ets_error_type")
                    params['trend_type'] = st.selectbox("Тип тренда", ["None", "add", "mul"], index=["None", "add", "mul"].index(params.get('trend_type', 'None')), key="ets_trend_type")
                    params['season_type'] = st.selectbox("Тип сезонности", ["None", "add", "mul"], index=["None", "add", "mul"].index(params.get('season_type', 'None')), key="ets_season_type")
                    params['seasonal_periods'] = st.number_input("Сезонность seasonal_periods", min_value=1, value=params.get('seasonal_periods', 12), step=1, format="%d", key="ets_seasonal_periods")
                    params['damped_trend'] = st.selectbox("Демпфирование тренда", ["True", "False"], index=["True", "False"].index(str(params.get('damped_trend', 'True'))), key="ets_damped_trend") == "True"
                elif option == "prophet":
                    params['seasonality_mode'] = st.selectbox("Режим сезонности", ["additive", "multiplicative"], index=["additive", "multiplicative"].index(params.get('seasonality_mode', 'additive')), key="prophet_seasonality_mode")
                    params['yearly_seasonality'] = st.selectbox("Годовая сезонность", ["True", "False"], index=["True", "False"].index(str(params.get('yearly_seasonality', 'True'))), key="prophet_yearly_seasonality") == "True"
                    params['weekly_seasonality'] = st.selectbox("Недельная сезонность", ["True", "False"], index=["True", "False"].index(str(params.get('weekly_seasonality', 'True'))), key="prophet_weekly_seasonality") == "True"
                    params['daily_seasonality'] = st.selectbox("Дневная сезонность", ["True", "False"], index=["True", "False"].index(str(params.get('daily_seasonality', 'True'))), key="prophet_daily_seasonality") == "True"
                    params['seasonality_prior_scale'] = st.number_input("Интенсивность сезонности", value=float(params.get('seasonality_prior_scale', 10.0)), key="prophet_seasonality_prior_scale")
                    params['changepoint_prior_scale'] = st.number_input("Чувствительность к точкам излома", value=float(params.get('changepoint_prior_scale', 0.05)), key="prophet_changepoint_prior_scale")

            if st.button("Подтвердить", key="confirm_forecast_dialog"):
                st.session_state['duration'] = duration
                st.session_state['params'] = params

                if training_df is not None and not training_df.empty and target_sensor:
                    if 'duration' not in st.session_state:
                        st.error("Сначала задайте количество предсказаний")
                        return
                    if auto_params and option in ["sarima", "ets"] and 'params' not in st.session_state:
                        st.error("Для выбранной модели с автоподбором необходимо указать сезонность")
                        return
                    if not auto_params and 'params' not in st.session_state:
                        st.error("Сначала укажите параметры модели")
                        return

                    # Формируем обучающую выборку по выбранному целевому признаку
                    df_train = training_df[[target_sensor]].copy()
                    if isinstance(df_train, pd.DataFrame):
                        df_train = df_train.rename(columns={str(target_sensor): 'sensor'})
                    if not isinstance(df_train.index, pd.DatetimeIndex):
                        st.error("Индекс обучающей выборки должен быть типа DatetimeIndex")
                        return

                    # Формируем экзогенные переменные, если они выбраны
                    df_exog = None
                    exog_sensors_list = st.session_state.get('exog_sensors', [])
                    if exog_sensors_list and len(exog_sensors_list) > 0:
                        df_exog = training_df[exog_sensors_list].copy()

                    # Формируем тестовую часть из рабочей версии данных, чтобы она соответствовала тому, что видно на графике
                    end_training_time = df_train.index.max() if df_train is not None and not df_train.empty else None
                    df_test = None
                    work_all = st.session_state.get('working_df', df)
                    if work_all is not None and not work_all.empty and end_training_time is not None:
                        df_test = work_all[work_all.index > end_training_time][[target_sensor]]
                        if isinstance(df_test, pd.DataFrame):
                            df_test = df_test.rename(columns={str(target_sensor): 'sensor'})
                            if len(df_test) >= duration:
                                df_test = df_test.iloc[:duration]
                                st.session_state['df_test'] = df_test
                            else:
                                df_test = None
                                st.session_state['df_test'] = None
                        else:
                            df_test = None
                            st.session_state['df_test'] = None
                    else:
                        df_test = None
                        st.session_state['df_test'] = None

                    # Формируем полезную нагрузку для API
                    payload = create_model_payload(
                        auto_params=auto_params,
                        horizon=duration,
                        df_train=df_train,
                        hyper_params=params,
                        exog_vars=df_exog
                    )

                    try:
                        model_request = ModelRequest(**payload)
                        with st.spinner("Выполняется прогнозирование"):
                            response = asyncio.run(get_prediction(api_url, model_request.dict(), option))

                        if response.status_code == 200:
                            result = response.json()
                            st.session_state['forecast_result'] = result

                            # Запрашиваем метрики, только если тестовая часть сформирована полностью
                            if df_test is not None and len(df_test) == duration:
                                try:
                                    df_predict = pd.read_json(StringIO(result["df_predict"]), orient='table')
                                    if 'sensor' not in df_predict.columns and len(df_predict.columns) == 1:
                                        df_predict = df_predict.rename(columns={str(df_predict.columns[0]): 'sensor'})
                                    df_predict = df_predict.replace([np.inf, -np.inf, np.nan], 0)
                                    df_test_clean = df_test.replace([np.inf, -np.inf, np.nan], 0)
                                    metrics_payload = create_metrics_payload(df_predict=df_predict, df_test=df_test_clean)
                                    metrics_request = MetricsRequest(
                                        df_predict=str(metrics_payload["df_predict"]),
                                        df_test=str(metrics_payload["df_test"])
                                    )
                                    with st.spinner("Рассчитываются метрики"):
                                        metrics_response = asyncio.run(get_metrics(api_url, metrics_request.dict()))
                                    if metrics_response.status_code == 200:
                                        st.session_state['metrics_result'] = metrics_response.json()
                                    else:
                                        st.warning(f"Не удалось получить метрики. Код ответа {metrics_response.status_code}")
                                except Exception as e:
                                    st.warning(f"Ошибка при расчете метрик. Подробности {e}")

                            st.success(f"Прогноз выполнен для {target_sensor}")
                            st.rerun()
                        else:
                            st.error(f"Ошибка API. Код ответа {response.status_code}. Текст {response.text}")
                    except Exception as e:
                        st.error(f"Ошибка при выполнении прогноза. Подробности {e}")
                else:
                    st.error("Загрузите данные, выберите обучающую выборку и целевую переменную")

        forecast_settings_dialog()

def render_forecasting_page(df: pd.DataFrame, outlier_percentage: float) -> None:
    """
    Рендерит страницу "Прогнозирование"
    """
    st.set_page_config(page_title="Прогнозирование", layout="wide")
    # Не отображаем информацию о датасете, если df is None
    if df is None:
        st.session_state.clear()
        return
    render_data_overview(df, outlier_percentage)
    if df is not None:
        # Используем более простой способ хеширования DataFrame
        current_df_hash = hash(str(df.shape) + str(df.columns.tolist()) + str(df.index[-10:].tolist()) if len(df) > 10 else str(df.index.tolist()))
        if st.session_state.get('last_df_hash') != current_df_hash:
            st.session_state.clear()
            # При загрузке нового файла ограничиваем данные последними 500 точками
            limited_df = limit_data_to_last_points(df, 500)
            st.session_state['filtered_df'] = limited_df
            st.session_state['selected_sensors'] = df.columns.tolist()
            st.session_state['sensor_editor_temp'] = df.columns.tolist()
            st.session_state['target_sensor'] = df.columns[0]
            st.session_state['last_df_hash'] = current_df_hash
            st.session_state['original_df'] = df  # Сохраняем оригинальный DataFrame
            st.session_state['is_limited_view'] = True  # Флаг, что отображается ограниченный вид
            # Рабочая копия исходных данных. Сюда будут применяться преобразования ряда.
            st.session_state['working_df'] = df.copy()
            # Флаг факта изменения исходного ряда. Управляет показом кнопки возврата.
            st.session_state['is_series_modified'] = False
            # Сброс результатов прогнозирования, метрик и тестовых данных
            for key in ['forecast_result', 'metrics_result', 'df_test', 'duration']:
                if key in st.session_state:
                    del st.session_state[key]
    render_forecasting_main_panel(df)