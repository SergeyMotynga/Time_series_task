import streamlit as st
import pandas as pd
import json
import re
from typing import Tuple, Optional

# Импортируем утилиты из отдельных файлов
from dashboard.utils.date_utils import find_date_column, process_date_column, get_date_formats
from dashboard.utils.validation_utils import validate_numeric_columns
from dashboard.utils.validate_data import validate_dataframe_structure
from dashboard.utils.statistics_utils import calculate_outlier_percentage, fill_missing_values, sort_dataframe_by_index


def sanitize_column_name(name: str) -> str:
    """
    Очищает название колонки от специальных символов для совместимости с LightGBM.

    LightGBM не поддерживает некоторые спецсимволы в названиях признаков.
    Заменяем их на подчеркивания.

    Args:
        name: Исходное название колонки

    Returns:
        Очищенное название
    """
    # Заменяем двоеточия, точки и другие спецсимволы на подчеркивания
    # Оставляем только буквы, цифры и подчеркивания
    clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # Убираем множественные подчеркивания подряд
    clean_name = re.sub(r'_+', '_', clean_name)
    # Убираем подчеркивания в начале и конце
    clean_name = clean_name.strip('_')
    return clean_name


def parse_json_to_dataframe(json_data: list) -> pd.DataFrame:
    """
    Преобразует JSON данные временных рядов в pandas DataFrame.

    JSON формат:
    [
        {
            "tag": "sensor_name",
            "unit": "unit",
            "description": "description",
            "points": [
                {"d": "2025-11-10T00:00:00", "d_v": 99.95},
                ...
            ]
        },
        ...
    ]

    Результат - DataFrame, где:
    - Индекс: дата (из поля "d")
    - Столбцы: названия сенсоров (из поля "tag")
    - Значения: значения измерений (из поля "d_v")
    """
    all_series = {}

    for item in json_data:
        tag = item.get("tag", "unknown")
        points = item.get("points", [])

        if not points:
            continue

        # Создаем списки для валидных точек данных
        dates = []
        values = []

        for point in points:
            # Проверяем наличие обязательных полей
            if "d" not in point or "d_v" not in point:
                continue

            # Проверяем, что значение не None
            if point["d"] is None or point["d_v"] is None:
                continue

            try:
                # Пытаемся преобразовать значение в число
                value = float(point["d_v"])
                dates.append(point["d"])
                values.append(value)
            except (ValueError, TypeError):
                # Пропускаем точки с некорректными значениями
                continue

        # Если есть валидные точки, создаем Series
        if dates and values:
            try:
                # Сначала пытаемся распарсить все даты без потерь
                parsed_dates = pd.to_datetime(dates, utc=False)
                series = pd.Series(values, index=parsed_dates, name=tag)

                # Удаляем дубликаты дат (оставляем первое вхождение)
                series = series[~series.index.duplicated(keep='first')]

                all_series[tag] = series
            except Exception as e:
                # Если не получилось - пробуем с errors='coerce' и логируем потери
                try:
                    parsed_dates = pd.to_datetime(dates, errors='coerce', utc=False)

                    # Проверяем, сколько дат не удалось распарсить
                    invalid_count = parsed_dates.isna().sum()
                    total_count = len(dates)

                    if invalid_count > 0:
                        st.warning(f"Тег '{tag}': не удалось распарсить {invalid_count} из {total_count} дат ({invalid_count/total_count*100:.1f}%)")

                    # Создаем временный DataFrame для фильтрации
                    temp_df = pd.DataFrame({'date': parsed_dates, 'value': values})

                    # Удаляем строки с NaT (Not a Time) датами
                    temp_df = temp_df[temp_df['date'].notna()]

                    if temp_df.empty:
                        st.warning(f"Пропущен тег '{tag}': не удалось распарсить ни одной даты")
                        continue

                    series = pd.Series(temp_df['value'].values, index=temp_df['date'], name=tag)

                    # Удаляем дубликаты дат (оставляем первое вхождение)
                    series = series[~series.index.duplicated(keep='first')]

                    all_series[tag] = series
                except Exception as e2:
                    # Логируем проблему с преобразованием дат для этого тега
                    st.warning(f"Пропущен тег '{tag}': ошибка при обработке дат - {str(e2)}")
                    continue

    if not all_series:
        raise ValueError("JSON не содержит валидных данных временных рядов. Проверьте структуру файла.")

    # Объединяем все Series в DataFrame
    df = pd.DataFrame(all_series)

    # Очищаем названия колонок от специальных символов для совместимости с LightGBM
    df.columns = [sanitize_column_name(col) for col in df.columns]

    # Устанавливаем имя индекса
    df.index.name = "date"

    return df


def process_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """
    Обрабатывает DataFrame: валидирует данные, обрабатывает пропуски, вычисляет выбросы
    """
    # Валидируем базовую структуру
    validate_dataframe_structure(df)

    # Проверяем, является ли индекс уже DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        # Находим столбец с датами
        date_column, date_format = find_date_column(df)
        if date_column is None:
            st.error("Столбец с датой не найден. Поддерживаемые форматы: " +
                     ", ".join(get_date_formats()) + " или другие стандартные форматы дат")
            raise ValueError("Date column not found")

        # Обрабатываем столбец с датами
        df = process_date_column(df, date_column, date_format)

    # Убираем дубликаты по индексу (оставляем первую запись)
    df = df[~df.index.duplicated(keep="first")]

    # Сортируем по дате в порядке убывания
    df = sort_dataframe_by_index(df, ascending=False)

    # Валидируем числовые столбцы
    valid_columns = validate_numeric_columns(df)
    df = df[valid_columns]

    if df.empty:
        st.error("Нет валидных столбцов с данными датчиков.")
        raise ValueError("No valid columns found")

    # ВАЖНО: НЕ заполняем пропуски автоматически!
    # Если целевая переменная измеряется редко (LIMS), заполнение создаст синтетические данные.
    # Используйте "Временная агрегация" во вкладке "Создание признаков" вместо этого.
    # df = fill_missing_values(df, method="forward")  # ОТКЛЮЧЕНО

    # Вычисляем процент выбросов
    outlier_percentage = calculate_outlier_percentage(df)

    return df, outlier_percentage


def upload() -> Tuple[Optional[pd.DataFrame], Optional[float]]:
    """
    Основная функция загрузки и обработки CSV или JSON файла
    """
    upload_file = st.file_uploader("Загрузите CSV или JSON файл", type=["csv", "json"])

    if upload_file is not None:
        try:
            file_extension = upload_file.name.split(".")[-1].lower()

            if file_extension == "csv":
                # Загружаем CSV
                df = pd.read_csv(upload_file, sep=None, engine="python")
                # Удаляем полные дубликаты строк
                df = df.drop_duplicates()

            elif file_extension == "json":
                # Загружаем JSON
                json_content = upload_file.read()
                json_data = json.loads(json_content)

                # Парсим JSON в DataFrame
                df = parse_json_to_dataframe(json_data)

            else:
                st.error(f"Неподдерживаемый формат файла: {file_extension}")
                return None, None

            # Обрабатываем DataFrame
            df, outlier_percentage = process_dataframe(df)

            return df, outlier_percentage

        except json.JSONDecodeError as e:
            st.error(f"Ошибка при чтении JSON файла: {str(e)}")
            st.info("Убедитесь, что файл содержит корректный JSON")
            return None, None
        except ValueError as e:
            st.error(f"Ошибка валидации данных: {str(e)}")
            st.info("Проверьте структуру JSON: каждый объект должен содержать 'tag' и 'points' с полями 'd' и 'd_v'")
            return None, None
        except Exception as e:
            st.error(f"Ошибка при обработке данных: {str(e)}")
            st.info("Если проблема сохраняется, проверьте формат файла")
            return None, None

    return None, None
