import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


with open('config.txt', 'r') as f:
    exec(f.read())


def get_model_name(path):
    return Path(path).stem


def find_model_files(models_dir):
    model_files = []

    for root, _, files in os.walk(models_dir):
        for file in files:
            if file.endswith(".pkl") or file.endswith(".joblib"):
                model_files.append(os.path.join(root, file))

    return sorted(model_files)


def model_matches_target(model_name, target):
    name = model_name.upper()
    target = target.upper()

    if target == "MFR":
        return "MFR" in name and "DD" not in name

    if target == "DD":
        return "DD" in name

    raise ValueError("TARGET должен быть 'MFR' или 'DD'")


def get_model_settings(model_name):
    # сначала точные совпадения
    if model_name in NORMALIZATION_CONFIG:
        return NORMALIZATION_CONFIG[model_name]

    # потом частичные совпадения
    for key, value in NORMALIZATION_CONFIG.items():
        if key.lower() in model_name.lower():
            return value

    return {
        "use_normalization": DEFAULT_USE_NORMALIZATION,
        "use_poly": DEFAULT_USE_POLY
    }


def safe_predict_value(raw_pred):
    arr = np.asarray(raw_pred).reshape(-1)
    return arr[0]


print("=" * 70)
print("СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ МОДЕЛЕЙ")
print("=" * 70)

dataset = pd.read_excel(DATA_PATH)

X = dataset.iloc[ROW_START:, FEATURES].values.astype(float)

mfr_true = dataset.iloc[ROW_START:, [MFR_ERR_COL]].values.astype(float).flatten()
dd_true = dataset.iloc[ROW_START:, [DD_ERR_COL]].values.astype(float).flatten()

if TARGET.upper() == "MFR":
    y_true = mfr_true
elif TARGET.upper() == "DD":
    y_true = dd_true
else:
    raise ValueError("TARGET должен быть 'MFR' или 'DD'")

all_models = find_model_files(MODELS_DIR)

model_files = [
    path for path in all_models
    if model_matches_target(get_model_name(path), TARGET)
]

if len(model_files) == 0:
    raise FileNotFoundError(f"Не найдено моделей для TARGET={TARGET} в {MODELS_DIR}")

print(f"Dataset: {DATA_PATH}")
print(f"Строк для сравнения: {len(X)}")
print(f"TARGET для сравнения: {TARGET}")
print(f"Найдено моделей для TARGET={TARGET}: {len(model_files)}")

for path in model_files:
    print(f" - {get_model_name(path)}")


prediction_table = pd.DataFrame({
    "MFRerr": mfr_true,
    "DDerr": dd_true
})

metrics_rows = []
time_data = {}

# Список маркеров для разных линий
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', '+', 'x', '|', '_', 'd']

for idx, model_path in enumerate(model_files):
    model_name = get_model_name(model_path)

    print("\n" + "=" * 60)
    print(f"Модель: {model_name}")
    print("=" * 60)

    model = joblib.load(model_path)
    settings = get_model_settings(model_name)

    use_poly = settings.get("use_poly", DEFAULT_USE_POLY)
    poly_degree = settings.get("poly_degree", 5)

    if use_poly or "poly" in model_name.lower():
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        X_model = poly.fit_transform(X)
        print(f"PolynomialFeatures: ON, degree={poly_degree}")
    else:
        X_model = X.copy()
        print("PolynomialFeatures: OFF")

    use_norm = settings.get("use_normalization", DEFAULT_USE_NORMALIZATION)

    if use_norm:
        X_scaler = joblib.load(settings["x_scaler_path"])
        y_scaler = joblib.load(settings["y_scaler_path"])
        X_input = X_scaler.transform(X_model)
        print("Нормализация: ON")
    else:
        y_scaler = None
        X_input = X_model
        print("Нормализация: OFF")

    predictions = []
    times_ms = []

    for i in range(len(X_input)):
        x_one = X_input[i].reshape(1, -1)

        start = time.perf_counter()
        raw_pred = model.predict(x_one)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if use_norm:
            pred = y_scaler.inverse_transform(
                np.asarray(raw_pred).reshape(-1, 1)
            )[0][0]
        else:
            pred = safe_predict_value(raw_pred)

        predictions.append(pred)
        times_ms.append(elapsed_ms)

    predictions = np.asarray(predictions)
    times_ms = np.asarray(times_ms)

    prediction_table[f"Предсказанное({model_name})"] = predictions
    prediction_table[f"Время_мс({model_name})"] = times_ms

    time_data[model_name] = times_ms

    mse = mean_squared_error(y_true, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, predictions)
    r2 = r2_score(y_true, predictions)

    metrics_rows.append({
        "Model": model_name,
        "Target": TARGET,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Mean_Time_ms": np.mean(times_ms),
        "Min_Time_ms": np.min(times_ms),
        "Max_Time_ms": np.max(times_ms),
        "Std_Time_ms": np.std(times_ms),
        "Total_Time_ms": np.sum(times_ms)
    })


prediction_table.to_csv(OUTPUT_PREDICTIONS_CSV, index=False)

metrics_table = pd.DataFrame(metrics_rows)
metrics_table = metrics_table.sort_values(
    by=["MAE", "RMSE", "Mean_Time_ms"],
    ascending=True
)
metrics_table.to_csv(OUTPUT_METRICS_CSV, index=False)

print("\n" + "=" * 70)
print("ТАБЛИЦЫ СОХРАНЕНЫ")
print("=" * 70)
print(f"Предсказания: {OUTPUT_PREDICTIONS_CSV}")
print(f"Метрики: {OUTPUT_METRICS_CSV}")

print("\nРейтинг моделей:")
print(metrics_table.to_string(index=False))

# ПОСТРОЕНИЕ ГРАФИКА С ТОЧКАМИ И ОТСЕЧЕНИЕМ ВЫБРОСОВ
plt.figure(figsize=(14, 7))

# Собираем все времена для определения пределов с отсечением выбросов
all_times = []
for model_name, times_ms in time_data.items():
    all_times.extend(times_ms)

# Расчет процентилей для отсечения выбросов (отображаем 1-й до 99-й процентиль)
lower_bound = np.percentile(all_times, 1)
upper_bound = np.percentile(all_times, 99)

print(f"\nДиапазон отображения (1-99 процентили): {lower_bound:.3f} - {upper_bound:.3f} мс")
print(f"Выбросы выше {upper_bound:.3f} мс будут обрезаны")

for idx, (model_name, times_ms) in enumerate(time_data.items()):
    marker = markers[idx % len(markers)]

    # Обрезаем выбросы для отображения (но не для данных)
    times_clipped = np.clip(times_ms, lower_bound, upper_bound)

    # Рисуем линию с точками
    plt.plot(times_clipped,
             marker=marker,
             markersize=3,
             markevery=5,  # Ставим точку каждые 5 измерений для читаемости
             linewidth=1.5,
             label=f"{model_name} (max: {np.max(times_ms):.2f} мс)",  # Показываем реальный максимум
             alpha=0.8)

    # Отмечаем выбросы красными точками (опционально)
    outliers = times_ms > upper_bound
    if np.any(outliers):
        outlier_indices = np.where(outliers)[0]
        outlier_values = times_ms[outliers]
        # Рисуем выбросы на верхней границе графика
        plt.scatter(outlier_indices, [upper_bound] * len(outlier_indices),
                   color='red', s=20, zorder=5, alpha=0.5)
        print(f"  {model_name}: {np.sum(outliers)} выбросов > {upper_bound:.3f} мс (макс: {np.max(times_ms):.3f} мс)")

plt.title(f"Сравнение времени предсказания моделей | Target: {TARGET}\n(Отображены значения от {lower_bound:.2f} до {upper_bound:.2f} мс, выбросы обрезаны)",
          fontsize=12)
plt.xlabel("Номер измерения")
plt.ylabel("Время предсказания, мс")
plt.grid(True, linestyle="--", alpha=0.6)

# Устанавливаем пределы графика (с небольшим запасом)
plt.ylim(lower_bound * 0.9, upper_bound * 1.05)

plt.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    borderaxespad=0,
    fontsize=9
)

plt.tight_layout(rect=[0, 0, 0.78, 1])
plt.savefig(OUTPUT_TIME_PLOT, dpi=300, bbox_inches="tight")
plt.close()

print(f"\nГрафик сохранён: {OUTPUT_TIME_PLOT}")
print("\nГОТОВО")
