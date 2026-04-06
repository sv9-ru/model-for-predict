#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для предсказания с использованием загруженной модели
Поддерживает предсказания из датасета или ручной ввод
Работает в локальной директории (Raspberry Pi / любой Linux)
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn import preprocessing
import joblib
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                           r2_score, mean_absolute_percentage_error,
                           explained_variance_score)
import warnings
import os
import sys
warnings.filterwarnings('ignore')

# ==================================================
# ОПРЕДЕЛЕНИЕ РАБОЧЕЙ ДИРЕКТОРИИ
# ==================================================
# Получаем директорию, где находится скрипт
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Если скрипт запущен не из своей директории, используем текущую рабочую
WORK_DIR = os.getcwd()

print(f"Рабочая директория: {WORK_DIR}")
print(f"Директория скрипта: {SCRIPT_DIR}")

# ==================================================
# ФУНКЦИЯ ЗАГРУЗКИ КОНФИГУРАЦИИ
# ==================================================
def load_config(config_path=None):
    """Загрузка конфигурации из текстового файла"""
    if config_path is None:
        config_path = os.path.join(WORK_DIR, 'config.txt')
    
    config = {}

    if not os.path.exists(config_path):
        print(f"Файл конфигурации не найден: {config_path}")
        print("Использую настройки по умолчанию")
        return get_default_config()

    with open(config_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Пропускаем комментарии и пустые строки
            if line and not line.startswith('#'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()

                    # Преобразование типов
                    if value.lower() in ['true', 'false']:
                        config[key] = value.lower() == 'true'
                    elif value.isdigit():
                        config[key] = int(value)
                    elif value.replace('.', '').replace('-', '').isdigit():
                        config[key] = float(value)
                    else:
                        config[key] = value

    print(f"Конфигурация загружена из {config_path}")
    return config

def get_default_config():
    """Настройки по умолчанию (используем относительные пути)"""
    return {
        'model_path': os.path.join(WORK_DIR, 'SVR_MFR_dataset1 (1).pkl'),
        'dataset_path': os.path.join(WORK_DIR, 'dataset1_full.xlsx'),
        'MFR_obs_col': 5,
        'DD_col': 7,
        'MFR_err_col': 8,
        'DD_err_col': 10,
        'row_start': 3,
        'prediction_mode': 'dataset',
        'target_metric': 'MFR',
        'prediction_interval': 0,
        'normalize_data': True,
        'show_progress': True,
        'max_display_samples': 10,
        'save_predictions': True,
        'predictions_output': os.path.join(WORK_DIR, 'predictions_results.csv'),
        'save_plots': True,
        'plots_dir': os.path.join(WORK_DIR, 'plots')
    }

# ==================================================
# ФУНКЦИЯ РУЧНОГО ВВОДА ДАННЫХ
# ==================================================
def manual_prediction_loop(model, X_scaler, y_scaler, config, feature_names=['MFR_obs', 'DD']):
    """Цикл ручного ввода данных для предсказания"""
    print("\n" + "="*60)
    print("РЕЖИМ РУЧНОГО ВВОДА ДАННЫХ")
    print("="*60)
    print("Введите значения признаков для предсказания")
    print("Для выхода введите 'q' или 'quit'")
    print("-"*60)

    predictions_history = []
    input_history = []

    while True:
        try:
            print(f"\nВведите {feature_names[0]} и {feature_names[1]}:")
            user_input = input("-> ").strip()

            if user_input.lower() in ['q', 'quit', 'exit']:
                print("\nВыход из режима ручного ввода")
                break

            # Парсинг ввода
            values = user_input.split()
            if len(values) < 2:
                print("Ошибка: введите два числа через пробел")
                continue

            val1 = float(values[0])
            val2 = float(values[1])

            # Создание массива признаков
            X_input = np.array([[val1, val2]])

            # Нормализация (если включена)
            if config['normalize_data'] and X_scaler is not None:
                X_input_scaled = X_scaler.transform(X_input)
            else:
                X_input_scaled = X_input

            # Предсказание
            start_time = time.perf_counter()
            pred_scaled = model.predict(X_input_scaled)
            pred_time = time.perf_counter() - start_time

            # Обратное преобразование
            if config['normalize_data'] and y_scaler is not None:
                prediction = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
            else:
                prediction = pred_scaled[0]

            # Сохранение истории
            input_history.append([val1, val2])
            predictions_history.append(prediction)

            # Вывод результата
            print(f"\nРЕЗУЛЬТАТ ПРЕДСКАЗАНИЯ:")
            print(f"   * {feature_names[0]}: {val1}")
            print(f"   * {feature_names[1]}: {val2}")
            print(f"   * Предсказанный {config['target_metric']}: {prediction:.6f}")
            print(f"   * Время предсказания: {pred_time*1000:.2f} мс")

            # Задержка (если задана)
            if config['prediction_interval'] > 0:
                time.sleep(config['prediction_interval'])

        except ValueError:
            print("Ошибка: введите корректные числа")
        except Exception as e:
            print(f"Ошибка при предсказании: {e}")

    return np.array(input_history) if input_history else np.array([]), np.array(predictions_history) if predictions_history else np.array([])

# ==================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ==================================================
def main():
    print("="*60)
    print("СИСТЕМА ПРЕДСКАЗАНИЯ НА ОСНОВЕ ML МОДЕЛИ")
    print("="*60)
    print(f"Рабочая директория: {WORK_DIR}")

    # Загрузка конфигурации
    config_path = os.path.join(WORK_DIR, 'config.txt')
    config = load_config(config_path)

    # Вывод конфигурации
    print("\nТЕКУЩАЯ КОНФИГУРАЦИЯ:")
    print("-"*40)
    for key, value in config.items():
        print(f"   {key}: {value}")
    print("-"*40)

    # Создание директории для графиков
    if config.get('save_plots', True):
        plots_dir = config.get('plots_dir', os.path.join(WORK_DIR, 'plots'))
        os.makedirs(plots_dir, exist_ok=True)
        print(f"Графики будут сохранены в: {plots_dir}")

    # Проверка режима предсказания
    if config['prediction_interval'] == 0:
        print("Режим предсказания: МГНОВЕННЫЙ (без задержки)")
    else:
        print(f"Режим предсказания: ЗАДЕРЖКА {config['prediction_interval']} сек")

    # ==================================================
    # ЗАГРУЗКА МОДЕЛИ
    # ==================================================
    print("\nЗагрузка модели...")
    try:
        if not os.path.exists(config['model_path']):
            raise FileNotFoundError(f"Файл модели не найден: {config['model_path']}")
        model = joblib.load(config['model_path'])
        print(f"Модель загружена: {type(model).__name__}")
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return

    # ==================================================
    # ПОДГОТОВКА НОРМАЛИЗАТОРОВ (если нужны)
    # ==================================================
    X_scaler = None
    y_scaler = None

    if config['normalize_data'] and config['prediction_mode'] == 'manual':
        print("\nДля ручного режима с нормализацией нужны эталонные данные")
        print("   Будут использованы данные из датасета для обучения нормализаторов")

        if os.path.exists(config['dataset_path']):
            try:
                dataset = pd.read_excel(config['dataset_path'])
                X_temp = dataset.iloc[config['row_start']:, [config['MFR_obs_col'], config['DD_col']]].values.astype(float)
                y_temp = dataset.iloc[config['row_start']:, [config['MFR_err_col']]].values.astype(float)

                X_scaler = preprocessing.StandardScaler().fit(X_temp)
                y_scaler = preprocessing.StandardScaler().fit(y_temp)
                print("Нормализаторы обучены на датасете")
            except Exception as e:
                print(f"Ошибка при обучении нормализаторов: {e}")
                config['normalize_data'] = False
        else:
            print("Датасет не найден, нормализация отключена")
            config['normalize_data'] = False

    # ==================================================
    # РЕЖИМ ПРЕДСКАЗАНИЯ: ИЗ ДАТАСЕТА
    # ==================================================
    if config['prediction_mode'] == 'dataset':
        print("\nРЕЖИМ ПРЕДСКАЗАНИЯ: ИЗ ДАТАСЕТА")

        # Загрузка датасета
        print("\nЗагрузка датасета...")
        try:
            if not os.path.exists(config['dataset_path']):
                raise FileNotFoundError(f"Файл датасета не найден: {config['dataset_path']}")
            dataset = pd.read_excel(config['dataset_path'])
            print(f"Датасет загружен: {dataset.shape[0]} строк")
        except Exception as e:
            print(f"Ошибка загрузки датасета: {e}")
            return

        # Подготовка данных
        print("\nПодготовка данных...")
        X_data = dataset.iloc[config['row_start']:, [config['MFR_obs_col'], config['DD_col']]].values.astype(float)

        if config['target_metric'] == 'MFR':
            y_data = dataset.iloc[config['row_start']:, [config['MFR_err_col']]].values.astype(float)
        else:  # DD
            y_data = dataset.iloc[config['row_start']:, [config['DD_err_col']]].values.astype(float)

        print(f"Подготовлено {len(X_data)} образцов")

        # Нормализация
        if config['normalize_data']:
            print("Применяется нормализация...")
            X_scaler = preprocessing.StandardScaler().fit(X_data)
            X_data_scaled = X_scaler.transform(X_data)

            y_scaler = preprocessing.StandardScaler().fit(y_data)
            y_data_scaled = y_scaler.transform(y_data)
            print("Данные нормализованы")
            use_scaled = True
        else:
            X_data_scaled = X_data
            y_data_scaled = y_data
            use_scaled = False

        # Пошаговое предсказание
        print(f"\n{'='*60}")
        print("НАЧАЛО ПРЕДСКАЗАНИЙ")
        print(f"{'='*60}")

        predictions = []
        prediction_times = []
        y_true_list = []

        for i in range(len(X_data)):
            start_time = time.perf_counter()

            current_sample = X_data_scaled[i:i+1] if use_scaled else X_data[i:i+1]
            pred_scaled = model.predict(current_sample)

            if use_scaled and y_scaler is not None:
                pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
            else:
                pred = pred_scaled[0]

            pred_time = time.perf_counter() - start_time

            predictions.append(pred)
            prediction_times.append(pred_time)
            y_true_list.append(y_data[i][0])

            if config['show_progress'] and (i < config['max_display_samples'] or i % 50 == 0 or i == len(X_data)-1):
                print(f"Образец {i+1:3d}/{len(X_data)} | "
                      f"Истинное: {y_data[i][0]:8.4f} | "
                      f"Предсказание: {pred:8.4f} | "
                      f"Время: {pred_time*1000:6.2f} мс")

            if config['prediction_interval'] > 0 and i < len(X_data) - 1:
                time.sleep(config['prediction_interval'])

        # Преобразование в numpy массивы
        predictions = np.array(predictions)
        y_true = np.array(y_true_list)
        prediction_times = np.array(prediction_times)

        # Расчет метрик
        print(f"\n{'='*60}")
        print("РАСЧЕТ МЕТРИК КАЧЕСТВА")
        print(f"{'='*60}")

        mse = mean_squared_error(y_true, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, predictions)
        mape = mean_absolute_percentage_error(y_true, predictions)
        r2 = r2_score(y_true, predictions)
        evs = explained_variance_score(y_true, predictions)

        n = len(y_true)
        p = X_data.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p else r2

        # Таблица метрик
        metrics_data = {
            'Метрика': ['MSE', 'RMSE', 'MAE', 'MAPE', 'R²', 'Adj R²', 'EVS'],
            'Значение': [
                f"{mse:.6f}", f"{rmse:.6f}", f"{mae:.6f}",
                f"{mape:.4%}", f"{r2:.6f}", f"{adj_r2:.6f}", f"{evs:.6f}"
            ]
        }
        metrics_df = pd.DataFrame(metrics_data)
        print("\n", metrics_df.to_string(index=False))

        # Временные метрики
        print(f"\nВРЕМЕННЫЕ МЕТРИКИ:")
        print(f"   * Среднее время: {np.mean(prediction_times)*1000:.2f} мс")
        print(f"   * Стд. отклонение: {np.std(prediction_times)*1000:.2f} мс")
        print(f"   * Мин. время: {np.min(prediction_times)*1000:.2f} мс")
        print(f"   * Макс. время: {np.max(prediction_times)*1000:.2f} мс")
        print(f"   * Общее время: {np.sum(prediction_times):.2f} сек")

        # Сохранение результатов
        if config['save_predictions']:
            results_df = pd.DataFrame({
                'Sample': range(1, len(predictions)+1),
                'True_Value': y_true,
                'Prediction': predictions,
                'Absolute_Error': np.abs(y_true - predictions),
                'Prediction_Time_ms': prediction_times * 1000
            })
            results_df.to_csv(config['predictions_output'], index=False)
            print(f"\nРезультаты сохранены в: {config['predictions_output']}")

            # Также сохраняем метрики в отдельный файл
            metrics_path = os.path.join(WORK_DIR, 'metrics_summary.csv')
            metrics_df.to_csv(metrics_path, index=False)
            print(f"Метрики сохранены в: {metrics_path}")

        # Построение и сохранение графиков
        residuals = y_true - predictions
        plot_results(y_true, predictions, prediction_times, residuals, config)

    # ==================================================
    # РЕЖИМ ПРЕДСКАЗАНИЯ: РУЧНОЙ ВВОД
    # ==================================================
    else:
        print("\nРЕЖИМ ПРЕДСКАЗАНИЯ: РУЧНОЙ ВВОД")
        input_data, predictions_history = manual_prediction_loop(model, X_scaler, y_scaler, config)

        if len(predictions_history) > 0:
            # Сохранение истории ручных предсказаний
            if config['save_predictions']:
                results_df = pd.DataFrame({
                    'MFR_obs': input_data[:, 0],
                    'DD': input_data[:, 1],
                    f'Predicted_{config["target_metric"]}': predictions_history
                })
                results_df.to_csv(config['predictions_output'], index=False)
                print(f"\nИстория предсказаний сохранена в: {config['predictions_output']}")
        else:
            print("\nНет данных для сохранения")

    print("\n" + "="*60)
    print("РАБОТА ЗАВЕРШЕНА")
    print("="*60)

# ==================================================
# ФУНКЦИЯ ПОСТРОЕНИЯ И СОХРАНЕНИЯ ГРАФИКОВ
# ==================================================
def plot_results(y_true, predictions, prediction_times, residuals, config):
    """Построение и сохранение графиков результатов"""
    print("\nПостроение и сохранение графиков...")

    # Создаем директорию для графиков, если её нет
    plots_dir = config.get('plots_dir', os.path.join(WORK_DIR, 'plots'))
    os.makedirs(plots_dir, exist_ok=True)

    # Список для хранения имен сохраненных файлов
    saved_files = []

    # 1. ГРАФИК: Предсказанные vs Реальные
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, predictions, alpha=0.6, edgecolors='k', linewidth=0.5)
    min_val = min(y_true.min(), predictions.min())
    max_val = max(y_true.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Идеальное предсказание')
    plt.xlabel('Реальные значения', fontsize=12)
    plt.ylabel('Предсказанные значения', fontsize=12)
    plt.title(f'Предсказанные vs Реальные значения\n{config["target_metric"]}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    filepath1 = os.path.join(plots_dir, '1_predicted_vs_actual.png')
    plt.savefig(filepath1, dpi=300, bbox_inches='tight')
    saved_files.append(filepath1)
    plt.close()

    # 2. ГРАФИК: Временной ряд
    plt.figure(figsize=(12, 6))
    indices = np.arange(len(y_true))
    plt.plot(indices, y_true, 'b-', label='Реальные значения', marker='o', markersize=3, linewidth=1, alpha=0.7)
    plt.plot(indices, predictions, 'r-', label='Предсказанные значения', marker='x', markersize=3, linewidth=1, alpha=0.7)
    plt.xlabel('Номер образца', fontsize=12)
    plt.ylabel('Значение', fontsize=12)
    plt.title('Сравнение предсказаний с реальными данными', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    filepath2 = os.path.join(plots_dir, '2_time_series.png')
    plt.savefig(filepath2, dpi=300, bbox_inches='tight')
    saved_files.append(filepath2)
    plt.close()

    # 3. ГРАФИК: Время предсказания по образцам
    plt.figure(figsize=(12, 6))
    plt.plot(indices, [t*1000 for t in prediction_times], 'g-', marker='s', markersize=3, linewidth=1, alpha=0.7)
    plt.xlabel('Номер образца', fontsize=12)
    plt.ylabel('Время предсказания (мс)', fontsize=12)
    plt.title('Время предсказания для каждого образца', fontsize=14)
    plt.grid(True, alpha=0.3)
    filepath3 = os.path.join(plots_dir, '3_prediction_time_series.png')
    plt.savefig(filepath3, dpi=300, bbox_inches='tight')
    saved_files.append(filepath3)
    plt.close()

    # Вывод информации о сохраненных файлах
    print("\n" + "="*60)
    print("СОХРАНЕННЫЕ ГРАФИКИ:")
    print("="*60)
    for i, filepath in enumerate(saved_files, 1):
        print(f"{i}. {filepath}")
    print(f"\nВсе графики сохранены в директории: {plots_dir}")
    print("="*60)

# ==================================================
# ЗАПУСК СКРИПТА
# ==================================================
if __name__ == "__main__":
    main()
