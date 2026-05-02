import numpy as np
import pandas as pd
import joblib
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ================== ЗАГРУЗКА КОНФИГА ==================
with open('config.txt', 'r') as f:
    exec(f.read())

print("="*50)
print("ЗАПУСК ПРЕДСКАЗАНИЯ")
print("="*50)

# ================== ЗАГРУЗКА МОДЕЛИ И ДАННЫХ ==================
model = joblib.load(MODEL_PATH)
dataset = pd.read_excel(DATA_PATH)

model_name = model.__class__.__name__

print(f"✓ Загружено строк: {len(dataset)}")
print(f"✓ Модель: {model_name}")
print(f"✓ Цель: {TARGET}")
print(f"✓ Target в датасете: {'есть' if HAS_TARGET else 'нет'}")

# ================== ВЫБОР ЦЕЛЕВОЙ ==================
if TARGET == 'MFR':
    target_col = MFR_ERR_COL
elif TARGET == 'DD':
    target_col = DD_ERR_COL
else:
    target_col = None

# ================== ПОДГОТОВКА ДАННЫХ ==================
X = dataset.iloc[ROW_START:, FEATURES].values.astype(float)

if HAS_TARGET:
    y = dataset.iloc[ROW_START:, [target_col]].values.astype(float)
    y_true = y.flatten()
else:
    y = None
    y_true = None

print(f"✓ Данные: {len(X)} образцов, {len(FEATURES)} признаков")

# ================== НОРМАЛИЗАЦИЯ ==================
if USE_NORMALIZATION:
    X_scaler = joblib.load(X_SCALER_PATH)
    y_scaler = joblib.load(Y_SCALER_PATH)

    X_scaled = X_scaler.transform(X)

    print("✓ Нормализация включена")
else:
    X_scaled = X.copy()
    y_scaler = None

    print("✓ Нормализация выключена")

# ================== ПРЕДСКАЗАНИЕ ==================
predictions = []
times = []

print("\n--- Предсказание ---")

for i in range(len(X_scaled)):
    start = time.time()

    pred_scaled = model.predict(X_scaled[i].reshape(1, -1))

    elapsed = time.time() - start
    times.append(elapsed)

    if USE_NORMALIZATION:
        pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1))
    else:
        pred = pred_scaled.reshape(-1, 1)

    predictions.append(pred[0][0])

    if (i + 1) % 100 == 0:
        print(f"{i + 1}/{len(X_scaled)}")

predictions = np.array(predictions)
times_ms = np.array(times) * 1000

# ================== СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ==================
if HAS_TARGET:
    results = pd.DataFrame({
        '№': range(1, len(predictions) + 1),
        'Реальное': y_true,
        'Предсказанное': predictions,
        'Ошибка': np.abs(y_true - predictions),
        'Время_мс': times_ms
    })
else:
    results = pd.DataFrame({
        '№': range(1, len(predictions) + 1),
        'Предсказанное': predictions,
        'Время_мс': times_ms
    })

results.to_csv(OUTPUT_CSV, index=False)

print(f"\n✓ Результаты сохранены: {OUTPUT_CSV}")

# ================== МЕТРИКИ ==================
if HAS_TARGET and VERBOSE:
    mae = mean_absolute_error(y_true, predictions)
    mse = mean_squared_error(y_true, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, predictions)

    print("\n--- МЕТРИКИ ---")
    print(f"MAE:  {mae:.6f}")
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R2:   {r2:.6f}")

elif not HAS_TARGET:
    print("\n--- МЕТРИКИ ---")
    print("Метрики не рассчитаны: в датасете нет target значений")

# ================== ВРЕМЯ ==================
if VERBOSE:
    print("\n--- ВРЕМЯ ---")
    print(f"Среднее: {np.mean(times_ms):.3f} мс")
    print(f"Мин: {np.min(times_ms):.3f} мс")
    print(f"Макс: {np.max(times_ms):.3f} мс")

# ================== ГРАФИКИ ==================
if DRAW_PLOTS:
    print("\n--- Рисуем графики ---")

    # Исправление: берем директорию из OUTPUT_CSV
    output_dir = '/'.join(OUTPUT_CSV.split('/')[:-1])  # получаем './results_predict'
    filename = f"plot_time_{model_name}_{TARGET}.png"
    full_path = f"{output_dir}/{filename}"

    plt.figure(figsize=(10, 5))
    plt.plot(times_ms)
    plt.title(f"Время предсказания (мс)\n{model_name} | {TARGET}")
    plt.xlabel("Индекс")
    plt.ylabel("мс")
    plt.grid()
    plt.savefig(full_path)
    plt.close()

    print(f"✓ График сохранен: {full_path}")

print("\nГОТОВО")
