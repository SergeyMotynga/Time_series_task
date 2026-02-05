"""
Функции для взаимодействия с regression API
"""
import httpx
import json
import streamlit as st


def get_regression_models():
    """
    Получение списка всех моделей (включая регрессионные)

    Returns:
        dict: Словарь с моделями
    """
    api_url = st.secrets.get("api_url", "http://localhost:8000/api/")

    try:
        response = httpx.get(f"{api_url}models")

        if response.status_code == 200:
            data = response.json()
            return data.get('models', {})
        else:
            st.error(f"Ошибка при получении списка моделей: {response.status_code}")
            return {}

    except Exception as e:
        st.error(f"Ошибка соединения с API: {str(e)}")
        return {}


def train_regression_model(
    df,
    target_col,
    feature_cols,
    model_type,
    hyper_params,
    test_size=0.2,
    use_auto_tune=False
):
    """
    Обучение регрессионной модели через API

    Args:
        df: DataFrame с данными
        target_col: Название целевой переменной
        feature_cols: Список признаков (или None для всех)
        model_type: Тип модели ('linear_regression', 'random_forest', etc.)
        hyper_params: Словарь с гиперпараметрами (или диапазонами для автоподбора)
        test_size: Размер тестовой выборки
        use_auto_tune: Использовать GridSearchCV для автоподбора

    Returns:
        dict: Результаты обучения модели
    """
    api_url = st.secrets.get("api_url", "http://localhost:8000/api/")

    # Подготовка данных
    df_json = df.to_json(orient='table', date_format='iso')

    # Преобразуем feature_cols в список если это pandas Series или Index
    if feature_cols is not None:
        import pandas as pd
        if isinstance(feature_cols, (pd.Series, pd.Index)):
            feature_cols_json = json.dumps(feature_cols.tolist())
        elif hasattr(feature_cols, 'tolist'):
            # На случай если это numpy array или другой объект с методом tolist
            feature_cols_json = json.dumps(feature_cols.tolist())
        else:
            feature_cols_json = json.dumps(feature_cols)
    else:
        feature_cols_json = None

    hyper_params_json = json.dumps(hyper_params)

    payload = {
        "df_train": df_json,
        "target_col": target_col,
        "feature_cols": feature_cols_json,
        "hyper_params": hyper_params_json,
        "test_size": test_size,
        "use_auto_tune": use_auto_tune
    }

    try:
        # Показываем размер отправляемых данных
        data_size_mb = len(df_json) / (1024 * 1024)
        if data_size_mb > 10:
            st.warning(f"⚠️ Размер данных: {data_size_mb:.2f} MB - это может занять время")

        response = httpx.post(
            f"{api_url}regression/{model_type}",
            json=payload,
            timeout=600.0  # 10 минут таймаут для больших данных
        )

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 504:
            st.error("⏱️ Сервер не успел обработать запрос (Gateway Timeout). Попробуйте уменьшить размер данных или отключить автоподбор.")
            return {"error": "Gateway Timeout"}
        elif response.status_code == 502:
            st.error("🔴 Сервер недоступен (Bad Gateway). Проверьте логи Railway.")
            return {"error": "Bad Gateway"}
        else:
            error_text = response.text[:500]
            st.error(f"❌ Ошибка API {response.status_code}: {error_text}")
            return {"error": f"HTTP {response.status_code}: {error_text}"}

    except httpx.TimeoutException:
        st.error("⏱️ Превышено время ожидания ответа от API (10 минут)")
        st.info("💡 Попробуйте: уменьшить размер данных, отключить автоподбор или увеличить test_size")
        return {"error": "Timeout"}
    except httpx.ConnectError as e:
        st.error(f"🔌 Не удалось подключиться к API: {api_url}. Проверьте что сервер запущен")
        return {"error": f"Connection error: {str(e)}"}
    except Exception as e:
        st.error(f"❌ Непредвиденная ошибка при обучении модели: {str(e)}")
        return {"error": str(e)}


def train_multiple_regression_models(
    df,
    target_col,
    feature_cols,
    model_types,
    hyper_params_dict,
    test_size=0.2
):
    """
    Обучение нескольких моделей для турнирной валидации

    Args:
        df: DataFrame с данными
        target_col: Целевая переменная
        feature_cols: Список признаков
        model_types: Список типов моделей
        hyper_params_dict: Словарь {model_type: hyper_params}
        test_size: Размер тестовой выборки

    Returns:
        dict: {model_type: результаты}
    """
    results = {}

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, model_type in enumerate(model_types):
        status_text.text(f"Обучение {model_type}...")

        hyper_params = hyper_params_dict.get(model_type, {})
        result = train_regression_model(
            df, target_col, feature_cols, model_type, hyper_params, test_size
        )

        results[model_type] = result
        progress_bar.progress((i + 1) / len(model_types))

    status_text.text("Обучение завершено!")
    return results
