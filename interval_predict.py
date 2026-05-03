'''import time
import subprocess
import pandas as pd
import os

# ================== ЗАГРУЗКА CONFIG.TXT ==================
config = {}
with open("config.txt", "r", encoding="utf-8") as f:
    exec(f.read(), config)

DATA_PATH = config["DATA_PATH"]
FEATURES = config["FEATURES"]
OUTPUT_CSV = config.get(
    "STREAM_OUTPUT_CSV",
    "./results_predict/interval_predictions.csv"
)

INTERVAL_MS = config.get("PREDICT_INTERVAL_MS", 1000)
ROW_START = config.get("ROW_START", 0)

# ================== ЗАГРУЗКА ДАННЫХ ==================
df = pd.read_excel(DATA_PATH)

# если ROW_START задан как номер строки Excel
df = df.iloc[ROW_START:].reset_index(drop=True)

results = []

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# ================== ПОСТРОЧНОЕ ПРЕДСКАЗАНИЕ ==================
for i, row in df.iterrows():
    features = [row.iloc[col] for col in FEATURES]

    command = [
        "python",
        "single_predict.py",
        *[str(x) for x in features]
    ]

    output = subprocess.check_output(command, text=True).strip()

    prediction, prediction_time_ms = output.split()
    prediction = float(prediction)
    prediction_time_ms = float(prediction_time_ms)

    result_row = {
        "row_index": i + ROW_START,
        "prediction": prediction,
        "prediction_time_ms": prediction_time_ms
    }

    for col, value in zip(FEATURES, features):
        result_row[f"feature_col_{col}"] = value

    results.append(result_row)

    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

    print(
        f"row={i + ROW_START}, "
        f"prediction={prediction:.6f}, "
        f"time={prediction_time_ms:.6f} ms"
    )

    time.sleep(INTERVAL_MS / 1000)

print(f"Готово. Результаты сохранены в {OUTPUT_CSV}")'''
import os
import csv
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

# Какая величина отображается на графике:
# inference_ms — только model.predict()
# compute_ms   — подготовка + нормализация + predict + inverse_transform
# cycle_ms     — полный цикл обработки строки без sleep
PLOT_TIME_COLUMN = config.get("PLOT_TIME_COLUMN", "inference_ms")

# Частота обновления графика в измерениях
PLOT_UPDATE_EVERY = config.get("PLOT_UPDATE_EVERY", 1)

# Сохранять график после завершения
SAVE_PLOT = config.get("SAVE_PLOT", True)


# ================== ПОДГОТОВКА ИМЁН И ПАПОК ==================
output_dir = os.path.dirname(OUTPUT_CSV)

if output_dir:
    os.makedirs(output_dir, exist_ok=True)
else:
    output_dir = "."

model_label = Path(MODEL_PATH).stem

PLOT_PATH = config.get(
    "REALTIME_PLOT_PATH",
    os.path.join(output_dir, f"realtime_prediction_time_{model_label}.png")
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

max_feature_index = max(FEATURES)

if max_feature_index >= df.shape[1]:
    raise IndexError(
        f"В FEATURES указан столбец {max_feature_index}, "
        f"но в датасете только {df.shape[1]} столбцов."
    )


# ================== ФУНКЦИЯ ПРЕДСКАЗАНИЯ ==================
def predict_one(features):
    """
    prediction   — предсказание в исходном масштабе;
    inference_ms — время только model.predict();
    compute_ms   — подготовка + нормализация + predict + inverse_transform.
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


# ================== НАСТРОЙКА CSV ==================
fieldnames = [
    "row_index",
    "prediction",
    "inference_ms",
    "compute_ms",
    "cycle_ms",
    "interval_ms",
    "sleep_ms",
    "deadline_miss",
]

for col in FEATURES:
    fieldnames.append(f"feature_col_{col}")


# ================== НАСТРОЙКА ГРАФИКА ==================
plt.ion()

fig, ax = plt.subplots(figsize=(12, 6))

x_plot = []
y_plot = []

line, = ax.plot(x_plot, y_plot, marker="o", linewidth=1.5, markersize=4)

ax.set_title(
    f"Время выполнения предсказаний в реальном времени\n"
    f"Модель: {model_label}",
    fontsize=14,
    fontweight="bold"
)

ax.set_xlabel("Номер измерения")
ax.set_ylabel("Время выполнения предсказания, мс")
ax.grid(True)

fig.tight_layout()


# ================== ИНТЕРВАЛЬНОЕ ПРЕДСКАЗАНИЕ ==================
print("=" * 70)
print("ИНТЕРВАЛЬНОЕ ПРЕДСКАЗАНИЕ В ОДНОМ ПРОЦЕССЕ")
print("=" * 70)
print(f"Модель: {MODEL_PATH}")
print(f"Название на графике: {model_label}")
print(f"Target: {TARGET}")
print(f"Строк: {len(df)}")
print(f"Интервал: {INTERVAL_MS} мс")
print(f"Нормализация: {'включена' if USE_NORMALIZATION else 'выключена'}")
print(f"Метрика времени на графике: {PLOT_TIME_COLUMN}")
print(f"CSV: {OUTPUT_CSV}")
print(f"График: {PLOT_PATH}")
print("=" * 70)

interval_s = INTERVAL_MS / 1000
next_tick = time.perf_counter()

all_inference_times = []
all_compute_times = []
all_cycle_times = []
missed_deadlines = 0

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for i, row in df.iterrows():
        cycle_start = time.perf_counter()

        features = [row.iloc[col] for col in FEATURES]

        prediction, inference_ms, compute_ms = predict_one(features)

        cycle_ms = (time.perf_counter() - cycle_start) * 1000

        if PLOT_TIME_COLUMN == "inference_ms":
            plot_value = inference_ms
        elif PLOT_TIME_COLUMN == "compute_ms":
            plot_value = compute_ms
        elif PLOT_TIME_COLUMN == "cycle_ms":
            plot_value = cycle_ms
        else:
            raise ValueError(
                "PLOT_TIME_COLUMN должен быть 'inference_ms', "
                "'compute_ms' или 'cycle_ms'"
            )

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

        writer.writerow(result_row)
        f.flush()

        all_inference_times.append(inference_ms)
        all_compute_times.append(compute_ms)
        all_cycle_times.append(cycle_ms)

        # ================== ОБНОВЛЕНИЕ ГРАФИКА ==================
        x_plot.append(i + ROW_START)
        y_plot.append(plot_value)

        if (i + 1) % PLOT_UPDATE_EVERY == 0:
            line.set_data(x_plot, y_plot)

            ax.relim()
            ax.autoscale_view()

            fig.canvas.draw()
            fig.canvas.flush_events()

        print(
            f"row={i + ROW_START:03d} | "
            f"prediction={prediction:.6f} | "
            f"inference={inference_ms:.6f} ms | "
            f"compute={compute_ms:.6f} ms | "
            f"cycle={cycle_ms:.6f} ms | "
            f"sleep={sleep_ms:.3f} ms | "
            f"miss={deadline_miss}"
        )


# ================== СОХРАНЕНИЕ ГРАФИКА ==================
line.set_data(x_plot, y_plot)
ax.relim()
ax.autoscale_view()

fig.canvas.draw()
fig.canvas.flush_events()

if SAVE_PLOT:
    fig.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
    print(f"\nГрафик сохранён: {PLOT_PATH}")

plt.ioff()
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
