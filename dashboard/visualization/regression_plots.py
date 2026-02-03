"""
Модуль визуализации для регрессионного анализа
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def plot_actual_vs_predicted(df_predictions, model_name="Model"):
    """
    График фактических vs предсказанных значений (scatter plot)

    Args:
        df_predictions: DataFrame с колонками 'actual' и 'predicted'
        model_name: Название модели для заголовка

    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure()

    # Scatter plot
    fig.add_trace(go.Scatter(
        x=df_predictions['actual'],
        y=df_predictions['predicted'],
        mode='markers',
        name='Предсказания',
        marker=dict(
            size=8,
            color='rgba(55, 128, 191, 0.7)',
            line=dict(width=1, color='rgba(55, 128, 191, 1)')
        ),
        text=[f"Факт: {a:.2f}<br>Пред: {p:.2f}"
              for a, p in zip(df_predictions['actual'], df_predictions['predicted'])],
        hovertemplate='%{text}<extra></extra>'
    ))

    # Идеальная линия (y=x)
    min_val = min(df_predictions['actual'].min(), df_predictions['predicted'].min())
    max_val = max(df_predictions['actual'].max(), df_predictions['predicted'].max())

    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Идеальное предсказание',
        line=dict(color='red', dash='dash', width=2)
    ))

    fig.update_layout(
        title=f'{model_name}: Фактические vs Предсказанные значения',
        xaxis_title='Фактические значения',
        yaxis_title='Предсказанные значения',
        hovermode='closest',
        showlegend=True,
        width=800,
        height=600
    )

    return fig


def plot_residuals(df_predictions, model_name="Model"):
    """
    График остатков (residuals)

    Args:
        df_predictions: DataFrame с колонками 'actual' и 'predicted'
        model_name: Название модели

    Returns:
        plotly.graph_objects.Figure
    """
    residuals = df_predictions['actual'] - df_predictions['predicted']

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Остатки по индексу', 'Распределение остатков'),
        vertical_spacing=0.15
    )

    # Residuals scatter
    fig.add_trace(
        go.Scatter(
            x=list(range(len(residuals))),
            y=residuals,
            mode='markers',
            name='Остатки',
            marker=dict(size=6, color='rgba(255, 127, 14, 0.7)')
        ),
        row=1, col=1
    )

    # Zero line
    fig.add_trace(
        go.Scatter(
            x=[0, len(residuals)],
            y=[0, 0],
            mode='lines',
            name='Ноль',
            line=dict(color='red', dash='dash')
        ),
        row=1, col=1
    )

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=residuals,
            name='Распределение',
            marker=dict(color='rgba(255, 127, 14, 0.7)'),
            nbinsx=30
        ),
        row=2, col=1
    )

    fig.update_xaxes(title_text="Индекс выборки", row=1, col=1)
    fig.update_yaxes(title_text="Остатки", row=1, col=1)
    fig.update_xaxes(title_text="Остатки", row=2, col=1)
    fig.update_yaxes(title_text="Частота", row=2, col=1)

    fig.update_layout(
        title_text=f'{model_name}: Анализ остатков',
        showlegend=False,
        width=800,
        height=800
    )

    return fig


def plot_feature_importance(feature_importance, model_name="Model", top_n=15):
    """
    График важности признаков

    Args:
        feature_importance: Dict {feature_name: importance_value}
        model_name: Название модели
        top_n: Количество топовых признаков для отображения

    Returns:
        plotly.graph_objects.Figure
    """
    # Сортировка и выбор топ N
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:top_n]

    features = [f[0] for f in top_features]
    importances = [f[1] for f in top_features]

    fig = go.Figure(go.Bar(
        x=importances,
        y=features,
        orientation='h',
        marker=dict(
            color=importances,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Важность")
        ),
        text=[f'{imp:.4f}' for imp in importances],
        textposition='auto'
    ))

    fig.update_layout(
        title=f'{model_name}: Топ {top_n} важных признаков',
        xaxis_title='Важность',
        yaxis_title='Признаки',
        yaxis=dict(autorange='reversed'),
        width=800,
        height=max(400, top_n * 30)
    )

    return fig


def plot_prediction_timeline(df_predictions, model_name="Model"):
    """
    График временного ряда предсказаний vs фактических значений

    Args:
        df_predictions: DataFrame с колонками 'actual' и 'predicted' и DatetimeIndex
        model_name: Название модели

    Returns:
        plotly.graph_objects.Figure
    """
    # ВАЖНО: сортируем данные по индексу для правильного отображения временного ряда
    df_sorted = df_predictions.sort_index()

    fig = go.Figure()

    # Фактические значения - сплошная синяя линия
    fig.add_trace(go.Scatter(
        x=df_sorted.index,
        y=df_sorted['actual'],
        mode='lines',
        name='Фактические',
        line=dict(color='rgb(31, 119, 180)', width=2),
        hovertemplate='Факт: %{y:.2f}<extra></extra>'
    ))

    # Предсказанные значения - пунктирная красная линия
    fig.add_trace(go.Scatter(
        x=df_sorted.index,
        y=df_sorted['predicted'],
        mode='lines',
        name='Предсказанные',
        line=dict(color='rgb(255, 65, 54)', width=2, dash='dash'),
        hovertemplate='Прогноз: %{y:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title=f'{model_name}: Временной ряд предсказаний',
        xaxis_title='Дата',
        yaxis_title='Значение',
        hovermode='x unified',
        width=1200,
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12)
        ),
        template='plotly_white'
    )

    return fig


def plot_model_comparison(results_dict, metric='test_r2'):
    """
    Сравнение метрик нескольких моделей

    Args:
        results_dict: Dict {model_name: result_dict}
        metric: Метрика для сравнения ('test_r2', 'test_mae', 'test_rmse', etc.)

    Returns:
        plotly.graph_objects.Figure
    """
    models = []
    values = []

    for model_name, result in results_dict.items():
        if 'error' not in result:
            model_params = result.get('model_params', {})
            metrics = model_params.get('metrics', {})
            if metric in metrics:
                models.append(model_name)
                values.append(metrics[metric])

    # Сортировка по значению метрики
    sorted_data = sorted(zip(models, values), key=lambda x: x[1], reverse=True)
    models = [d[0] for d in sorted_data]
    values = [d[1] for d in sorted_data]

    # Цветовая шкала: зеленый для лучших, красный для худших
    colors = px.colors.diverging.RdYlGn
    color_indices = np.linspace(0, len(colors)-1, len(values)).astype(int)
    bar_colors = [colors[i] for i in color_indices]

    fig = go.Figure(go.Bar(
        x=models,
        y=values,
        marker=dict(color=bar_colors),
        text=[f'{v:.4f}' for v in values],
        textposition='auto'
    ))

    metric_names = {
        'test_r2': 'R² (Тест)',
        'train_r2': 'R² (Обучение)',
        'test_mae': 'MAE (Тест)',
        'test_mse': 'MSE (Тест)',
        'test_rmse': 'RMSE (Тест)'
    }

    fig.update_layout(
        title=f'Сравнение моделей: {metric_names.get(metric, metric)}',
        xaxis_title='Модель',
        yaxis_title=metric_names.get(metric, metric),
        width=800,
        height=500
    )

    return fig


def plot_metrics_comparison_table(results_dict):
    """
    Таблица сравнения всех метрик для всех моделей

    Args:
        results_dict: Dict {model_name: result_dict}

    Returns:
        plotly.graph_objects.Figure
    """
    # Подготовка данных для таблицы
    rows = []

    for model_name, result in results_dict.items():
        if 'error' not in result:
            model_params = result.get('model_params', {})
            metrics = model_params.get('metrics', {})

            row = {
                'Модель': model_name,
                'Train R²': f"{metrics.get('train_r2', 0):.4f}",
                'Test R²': f"{metrics.get('test_r2', 0):.4f}",
                'MAE': f"{metrics.get('test_mae', 0):.4f}",
                'MSE': f"{metrics.get('test_mse', 0):.4f}",
                'RMSE': f"{metrics.get('test_rmse', 0):.4f}"
            }
            rows.append(row)
        else:
            rows.append({
                'Модель': model_name,
                'Train R²': 'ERROR',
                'Test R²': 'ERROR',
                'MAE': 'ERROR',
                'MSE': 'ERROR',
                'RMSE': 'ERROR'
            })

    df = pd.DataFrame(rows)

    # Сортировка по Test R² (если возможно)
    try:
        df['_sort_key'] = df['Test R²'].apply(lambda x: float(x) if x != 'ERROR' else -1)
        df = df.sort_values('_sort_key', ascending=False)
        df = df.drop('_sort_key', axis=1)
    except:
        pass

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(df.columns),
            fill_color='paleturquoise',
            align='left',
            font=dict(size=12, color='black')
        ),
        cells=dict(
            values=[df[col] for col in df.columns],
            fill_color='lavender',
            align='left',
            font=dict(size=11)
        )
    )])

    fig.update_layout(
        title='Сравнение метрик моделей',
        width=900,
        height=max(300, len(rows) * 40 + 100)
    )

    return fig


def plot_correlation_matrix(df, title="Корреляционная матрица"):
    """
    Тепловая карта корреляционной матрицы

    Args:
        df: DataFrame с признаками
        title: Заголовок графика

    Returns:
        plotly.graph_objects.Figure
    """
    corr_matrix = df.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 8},
        colorbar=dict(title="Корреляция")
    ))

    fig.update_layout(
        title=title,
        width=800,
        height=800,
        xaxis=dict(tickangle=-45)
    )

    return fig


def plot_sensor_graph(df, sensor_col, title=None):
    """
    График одного сенсора во времени

    Args:
        df: DataFrame с DatetimeIndex
        sensor_col: Название колонки сенсора
        title: Заголовок (если None, будет использовано имя сенсора)

    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[sensor_col],
        mode='lines',
        name=sensor_col,
        line=dict(width=2)
    ))

    fig.update_layout(
        title=title or f'Сенсор: {sensor_col}',
        xaxis_title='Дата',
        yaxis_title='Значение',
        hovermode='x unified',
        width=1000,
        height=400
    )

    return fig


def plot_multiple_sensors(df, sensor_cols, title="Множественные сенсоры"):
    """
    График нескольких сенсоров на одном графике

    Args:
        df: DataFrame с DatetimeIndex
        sensor_cols: Список названий колонок сенсоров
        title: Заголовок графика

    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure()

    for sensor_col in sensor_cols:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[sensor_col],
            mode='lines',
            name=sensor_col,
            line=dict(width=2)
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Дата',
        yaxis_title='Значение',
        hovermode='x unified',
        width=1200,
        height=600,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.05
        )
    )

    return fig


def plot_shift_correlation(shift_analysis_df, sensor_name, optimal_shift=None):
    """
    График зависимости корреляции от величины сдвига

    Args:
        shift_analysis_df: DataFrame с результатами analyze_shift_impact
        sensor_name: Название сенсора
        optimal_shift: Оптимальный сдвиг (будет выделен на графике)

    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure()

    # Абсолютная корреляция
    fig.add_trace(go.Scatter(
        x=shift_analysis_df['shift'],
        y=shift_analysis_df['abs_correlation'],
        mode='lines+markers',
        name='|Корреляция|',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))

    # Выделяем оптимальный сдвиг
    if optimal_shift is not None:
        optimal_row = shift_analysis_df[shift_analysis_df['shift'] == optimal_shift]
        if not optimal_row.empty:
            fig.add_trace(go.Scatter(
                x=[optimal_shift],
                y=[optimal_row['abs_correlation'].values[0]],
                mode='markers',
                name=f'Оптимальный сдвиг: {optimal_shift}',
                marker=dict(
                    size=15,
                    color='red',
                    symbol='star',
                    line=dict(width=2, color='darkred')
                )
            ))

    fig.update_layout(
        title=f'Корреляция vs Временной сдвиг для {sensor_name}',
        xaxis_title='Временной сдвиг (периоды)',
        yaxis_title='Абсолютная корреляция',
        hovermode='x unified',
        width=1000,
        height=500,
        showlegend=True
    )

    return fig


def plot_shifted_sensor_comparison(df, sensor_col, target_col, shift):
    """
    Сравнение оригинального и сдвинутого сенсора с целевой переменной

    Args:
        df: DataFrame с данными
        sensor_col: Название сенсора
        target_col: Целевая переменная
        shift: Величина сдвига

    Returns:
        plotly.graph_objects.Figure
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f'{sensor_col} (Оригинал) vs {target_col}',
            f'{sensor_col} (Сдвиг на {shift}) vs {target_col}'
        ),
        vertical_spacing=0.12
    )

    # Оригинальный сенсор
    fig.add_trace(
        go.Scatter(x=df.index, y=df[sensor_col], name=sensor_col, line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df[target_col], name=target_col, line=dict(color='red')),
        row=1, col=1
    )

    # Сдвинутый сенсор
    shifted = df[sensor_col].shift(shift)
    fig.add_trace(
        go.Scatter(x=df.index, y=shifted, name=f'{sensor_col} (сдвиг {shift})', line=dict(color='green')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df[target_col], name=target_col, line=dict(color='red'), showlegend=False),
        row=2, col=1
    )

    fig.update_layout(
        height=800,
        width=1200,
        title_text=f"Анализ сдвига сенсора: {sensor_col}",
        hovermode='x unified'
    )

    fig.update_xaxes(title_text="Дата", row=2, col=1)
    fig.update_yaxes(title_text="Значение", row=1, col=1)
    fig.update_yaxes(title_text="Значение", row=2, col=1)

    return fig


def plot_multiple_shifts_heatmap(df, sensor_cols, target_col, max_shift=24):
    """
    Тепловая карта корреляций для разных сдвигов нескольких сенсоров

    Args:
        df: DataFrame с данными
        sensor_cols: Список сенсоров
        target_col: Целевая переменная
        max_shift: Максимальный сдвиг

    Returns:
        plotly.graph_objects.Figure
    """
    from scipy.stats import pearsonr

    # Подготовка данных для heatmap
    shifts = list(range(0, max_shift + 1))
    correlation_matrix = []

    for sensor in sensor_cols:
        correlations = []
        for shift in shifts:
            if shift == 0:
                shifted = df[sensor]
            else:
                shifted = df[sensor].shift(shift)

            mask = ~(shifted.isna() | df[target_col].isna())
            if mask.sum() >= 10:
                corr, _ = pearsonr(shifted[mask], df[target_col][mask])
                correlations.append(abs(corr))
            else:
                correlations.append(0)

        correlation_matrix.append(correlations)

    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=shifts,
        y=sensor_cols,
        colorscale='Viridis',
        text=np.array(correlation_matrix),
        texttemplate='%{text:.2f}',
        textfont={"size": 8},
        colorbar=dict(title="|Корреляция|")
    ))

    fig.update_layout(
        title=f'Тепловая карта корреляций: Сенсоры vs Сдвиги<br>(Цель: {target_col})',
        xaxis_title='Временной сдвиг (периоды)',
        yaxis_title='Сенсор',
        width=1200,
        height=max(400, len(sensor_cols) * 50),
    )

    return fig
