import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ================== ЗАГРУЗКА CONFIG.TXT ==================
config = {}

with open("config.txt", "r", encoding="utf-8") as f:
    exec(f.read(), config)

DATA_PATH = config["DATA_PATH"]
MODEL_PATH = config["MODEL_PATH"]
FEATURES = config["FEATURES"]

TARGET = config.get("TARGET", "").upper()

USE_NORMALIZATION = config.get("USE_NORMALIZATION", False)
X_SCALER_PATH = config.get("X_SCALER_PATH", None)
Y_SCALER_PATH = config.get("Y_SCALER_PATH", None)

OUTPUT_CSV = config.get(
    "STREAM_OUTPUT_CSV",
    "./results_predict/interval_predictions.csv"
)

INTERVAL_MS = config.get("PREDICT_INTERVAL_MS", 1000)
ROW_START = config.get("ROW_START", 0)

WARMUP_RUNS = config.get("WARMUP_RUNS", 3)

# Что строить на графике:
# inference_ms — только model.predict()
# compute_ms   — подготовка + нормализация + predict + обратная нормализация
# cycle_ms     — полный цикл обработки строки без sleep
PLOT_TIME_COLUMN = config.get("PLOT_TIME_COLUMN", "inference_ms")

SAVE_PLOT = config.get("SAVE_PLOT", True)


# ================== ПОДГОТОВКА ПАПКИ РЕЗУЛЬТАТОВ ==================
output_dir = os.path.dirname(OUTPUT_CSV)

if output_dir:
    os.makedirs(output_dir, exist_ok=True)
else:
    output_dir = "."

model_label = Path(MODEL_PATH).stem

PLOT_PATH = config.get(
    "RESULT_PLOT_PATH",
    os.path.join(output_dir, f"prediction_time_{model_label}.png")
)


# ================== ЗАГРУЗКА ДАННЫХ ==================
df = pd.read_excel(DATA_PATH)
df = df.iloc[ROW_START:].reset_index(drop=True)

if len(df) == 0:
    raise ValueError("После применения ROW_START датасет не содержит строк.")


# ================== ЗАГРУЗКА МОДЕЛИ ОДИН РАЗ ==================
model = joblib.load(MODEL_PATH)


# ================== ЗАГРУЗКА НОРМАЛИЗАТОРОВ ОДИН РАЗ ==================
if USE_NORMALIZATION:
    if X_SCALER_PATH is None or Y_SCALER_PATH is None:
        raise ValueError(
            "Для USE_NORMALIZATION=True нужно указать "
            "X_SCALER_PATH и Y_SCALER_PATH в config.txt"
        )

    x_scaler = joblib.load(X_SCALER_PATH)
    y_scaler = joblib.load(Y_SCALER_PATH)
else:
    x_scaler = None
    y_scaler = None


# ================== ПРОВЕРКА ПРИЗНАКОВ ==================
if not isinstance(FEATURES, (list, tuple)):
    raise TypeError("FEATURES должен быть списком или кортежем индексов столбцов.")

if len(FEATURES) == 0:
    raise ValueError("FEATURES не должен быть пустым.")

max_feature_index = max(FEATURES)

if max_feature_index >= df.shape[1]:
    raise IndexError(
        f"В FEATURES указан столбец {max_feature_index}, "
        f"но в датасете только {df.shape[1]} столбцов."
    )


# ================== ФУНКЦИЯ ПРЕДСКАЗАНИЯ ==================
def predict_one(features):
    """
    prediction    — предсказание в исходном масштабе;
    inference_ms  — время только model.predict();
    compute_ms    — подготовка + нормализация + predict + inverse_transform.
    """

    compute_start = time.perf_counter()

    x = np.array(features, dtype=float).reshape(1, -1)

    if USE_NORMALIZATION:
        x_input = x_scaler.transform(x)
    else:
        x_input = x

    inference_start = time.perf_counter()
    raw_prediction = model.predict(x_input)
    inference_ms = (time.perf_counter() - inference_start) * 1000

    if USE_NORMALIZATION:
        prediction = y_scaler.inverse_transform(
            np.asarray(raw_prediction).reshape(-1, 1)
        )[0][0]
    else:
        prediction = np.asarray(raw_prediction).reshape(-1)[0]

    compute_ms = (time.perf_counter() - compute_start) * 1000

    return float(prediction), inference_ms, compute_ms


# ================== ПРОГРЕВ МОДЕЛИ ==================
first_features = [df.iloc[0, col] for col in FEATURES]

for _ in range(WARMUP_RUNS):
    predict_one(first_features)


# ================== ИНТЕРВАЛЬНОЕ ПРЕДСКАЗАНИЕ ==================
print("=" * 70)
print("ИНТЕРВАЛЬНОЕ ПРЕДСКАЗАНИЕ В ОДНОМ ПРОЦЕССЕ")
print("=" * 70)
print(f"Модель: {MODEL_PATH}")
print(f"Название модели для графика: {model_label}")
print(f"Target: {TARGET}")
print(f"Строк: {len(df)}")
print(f"Интервал: {INTERVAL_MS} мс")
print(f"Нормализация: {'включена' if USE_NORMALIZATION else 'выключена'}")
print(f"Метрика времени для графика: {PLOT_TIME_COLUMN}")
print(f"CSV: {OUTPUT_CSV}")
print(f"График: {PLOT_PATH}")
print("=" * 70)

interval_s = INTERVAL_MS / 1000
next_tick = time.perf_counter()

results = []
missed_deadlines = 0

all_inference_times = []
all_compute_times = []
all_cycle_times = []

for i, row in df.iterrows():
    cycle_start = time.perf_counter()

    features = [row.iloc[col] for col in FEATURES]

    prediction, inference_ms, compute_ms = predict_one(features)

    cycle_ms = (time.perf_counter() - cycle_start) * 1000

    next_tick += interval_s
    remaining_s = next_tick - time.perf_counter()

    if remaining_s > 0:
        sleep_ms = remaining_s * 1000
        deadline_miss = 0
        time.sleep(remaining_s)
    else:
        sleep_ms = 0.0
        deadline_miss = 1
        missed_deadlines += 1
        next_tick = time.perf_counter()

    result_row = {
        "row_index": i + ROW_START,
        "prediction": prediction,
        "inference_ms": inference_ms,
        "compute_ms": compute_ms,
        "cycle_ms": cycle_ms,
        "interval_ms": INTERVAL_MS,
        "sleep_ms": sleep_ms,
        "deadline_miss": deadline_miss,
    }

    for col, value in zip(FEATURES, features):
        result_row[f"feature_col_{col}"] = value

    results.append(result_row)

    all_inference_times.append(inference_ms)
    all_compute_times.append(compute_ms)
    all_cycle_times.append(cycle_ms)

    print(
        f"row={i + ROW_START:03d} | "
        f"prediction={prediction:.6f} | "
        f"inference={inference_ms:.6f} ms | "
        f"compute={compute_ms:.6f} ms | "
        f"cycle={cycle_ms:.6f} ms | "
        f"sleep={sleep_ms:.3f} ms | "
        f"miss={deadline_miss}"
    )


# ================== СОХРАНЕНИЕ CSV ПОСЛЕ ВСЕХ ИЗМЕРЕНИЙ ==================
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")


# ================== ВЫБОР МЕТРИКИ ДЛЯ ГРАФИКА ==================
if PLOT_TIME_COLUMN not in results_df.columns:
    raise ValueError(
        "PLOT_TIME_COLUMN должен быть одним из значений: "
        "'inference_ms', 'compute_ms', 'cycle_ms'"
    )

x_plot = results_df["row_index"].values
y_plot = results_df[PLOT_TIME_COLUMN].values


# ================== ПОСТРОЕНИЕ ГРАФИКА ПОСЛЕ ИЗМЕРЕНИЙ ==================
plt.figure(figsize=(12, 6))

plt.plot(
    x_plot,
    y_plot,
    marker="o",
    linewidth=1.5,
    markersize=4
)

plt.title(
    f"Время выполнения предсказаний\n"
    f"Модель: {model_label} | Метрика: {PLOT_TIME_COLUMN}",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Номер измерения")
plt.ylabel("Время, мс")
plt.grid(True)
plt.tight_layout()

if SAVE_PLOT:
    plt.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
    print(f"\nГрафик сохранён: {PLOT_PATH}")

plt.show()


# ================== ИТОГОВАЯ СТАТИСТИКА ==================
all_inference_times = np.array(all_inference_times)
all_compute_times = np.array(all_compute_times)
all_cycle_times = np.array(all_cycle_times)

print("\n" + "=" * 70)
print("ИТОГОВАЯ СТАТИСТИКА")
print("=" * 70)

print(f"Количество измерений: {len(all_inference_times)}")
print(f"Пропусков интервала: {missed_deadlines}")

print("\n--- Только model.predict() ---")
print(f"Среднее: {np.mean(all_inference_times):.6f} мс")
print(f"Медиана: {np.median(all_inference_times):.6f} мс")
print(f"Минимум: {np.min(all_inference_times):.6f} мс")
print(f"Максимум: {np.max(all_inference_times):.6f} мс")
print(f"95-й процентиль: {np.percentile(all_inference_times, 95):.6f} мс")

print("\n--- Подготовка + нормализация + predict + обратная нормализация ---")
print(f"Среднее: {np.mean(all_compute_times):.6f} мс")
print(f"Медиана: {np.median(all_compute_times):.6f} мс")
print(f"Минимум: {np.min(all_compute_times):.6f} мс")
print(f"Максимум: {np.max(all_compute_times):.6f} мс")
print(f"95-й процентиль: {np.percentile(all_compute_times, 95):.6f} мс")

print("\n--- Полный цикл без sleep ---")
print(f"Среднее: {np.mean(all_cycle_times):.6f} мс")
print(f"Медиана: {np.median(all_cycle_times):.6f} мс")
print(f"Минимум: {np.min(all_cycle_times):.6f} мс")
print(f"Максимум: {np.max(all_cycle_times):.6f} мс")
print(f"95-й процентиль: {np.percentile(all_cycle_times, 95):.6f} мс")

print(f"\nРезультаты сохранены в {OUTPUT_CSV}")
