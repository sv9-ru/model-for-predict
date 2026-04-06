import numpy as np
import pandas as pd
from sklearn import preprocessing
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import matplotlib.pyplot as plt

# ЗАГРУЗКА КОНФИГУРАЦИИ
with open('config.txt', 'r') as f:
    exec(f.read())

print("="*50)
print("ЗАПУСК ПРЕДСКАЗАНИЯ")
print("="*50)

# ЗАГРУЗКА МОДЕЛИ И ДАННЫХ
model = joblib.load(MODEL_PATH)
dataset = pd.read_excel(DATA_PATH)
print(f"✓ Загружено {len(dataset)} строк")

# ВЫБОР ЦЕЛЕВОЙ ПЕРЕМЕННОЙ
if TARGET == 'MFR':
    target_col = MFR_ERR_COL
else:
    target_col = DD_ERR_COL

# ПОДГОТОВКА ДАННЫХ
X = dataset.iloc[ROW_START:, FEATURES].values.astype(float)
y = dataset.iloc[ROW_START:, [target_col]].values.astype(float)

print(f"✓ Данные: {len(X)} образцов, {len(FEATURES)} признаков")
print(f"✓ Предсказываем: {TARGET}")

# НОРМАЛИЗАЦИЯ
X_scaler = preprocessing.StandardScaler().fit(X)
X_scaled = X_scaler.transform(X)

y_scaler = preprocessing.StandardScaler().fit(y)
y_scaled = y_scaler.transform(y)

# ПРЕДСКАЗАНИЕ
print("\n--- Предсказание ---")
predictions = []
times = []

for i in range(len(X_scaled)):
    start = time.time()
    pred_scaled = model.predict(X_scaled[i].reshape(1, -1))
    elapsed_time = time.time() - start
    times.append(elapsed_time)

    pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1))
    predictions.append(pred[0][0])

    if (i+1) % 100 == 0:
        print(f"  {i+1}/{len(X_scaled)}")

# РЕЗУЛЬТАТЫ
y_true = y.flatten()
predictions = np.array(predictions)
times_ms = np.array(times) * 1000  # конвертируем в миллисекунды

# ТАБЛИЦА (с добавленным временем вычисления)
results = pd.DataFrame({
    '№': range(1, len(predictions)+1),
    'Реальное': y_true,
    'Предсказанное': predictions,
    'Ошибка': np.abs(y_true - predictions),
    'Время_мс': times_ms  # время вычисления для каждого пункта в миллисекундах
})
results.to_csv(OUTPUT_CSV, index=False)
print(f"\n✓ Результаты сохранены в {OUTPUT_CSV}")

# СТАТИСТИКА
if VERBOSE:
    print(f"\n--- СТАТИСТИКА ---")
    print(f"Образцов: {len(predictions)}")
    print(f"\nВремя (мс):")
    print(f"  Среднее: {np.mean(times)*1000:.3f}")
    print(f"  Мин/Макс: {min(times)*1000:.3f} / {max(times)*1000:.3f}")
    print(f"\nКачество:")
    print(f"  MAE: {mean_absolute_error(y_true, predictions):.6f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_true, predictions)):.6f}")
    print(f"  R2: {r2_score(y_true, predictions):.6f}")
    print(f"\nВремя вычисления по пунктам:")
    print(f"  Общее время: {sum(times)*1000:.3f} мс")
    print(f"  Медиана: {np.median(times)*1000:.3f} мс")
    print(f"  Стандартное отклонение: {np.std(times)*1000:.3f} мс")

# ГРАФИКИ
if DRAW_PLOTS:
    print("\n--- Рисуем графики ---")

    # График 1: предсказание vs реальность
    plt.figure(figsize=(10,8))
    plt.scatter(y_true, predictions, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Реальные значения')
    plt.ylabel('Предсказанные значения')
    plt.title(f'Предсказание vs Реальность ({TARGET})')
    plt.grid(True, alpha=0.3)
    plt.savefig('./plot_predictions.png', dpi=300)
    plt.close()

    # График 2: сравнение по точкам
    plt.figure(figsize=(12,6))
    plt.plot(y_true, 'b-', label='Реальные', marker='o', markersize=2)
    plt.plot(predictions, 'r-', label='Предсказанные', marker='x', markersize=2)
    plt.xlabel('Номер образца')
    plt.ylabel(f'{TARGET}_error')
    plt.title('Сравнение предсказаний с реальными данными')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('./plot_comparison.png', dpi=300)
    plt.close()

    # График 3: время вычисления по пунктам
    plt.figure(figsize=(12,6))
    plt.plot(times_ms, 'g-', marker='o', markersize=2)
    plt.xlabel('Номер образца')
    plt.ylabel('Время вычисления (мс)')
    plt.title('Время предсказания для каждого образца')
    plt.grid(True, alpha=0.3)
    plt.savefig('./plot_inference_time.png', dpi=300)
    plt.close()

    print("✓ Графики сохранены: plot_predictions.png, plot_comparison.png, plot_inference_time.png")

print("\n" + "="*50)
print("ГОТОВО!")
