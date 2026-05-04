
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ================== ЗАГРУЗКА CONFIG ==================

CONFIG_PATH = "config_realtime.txt"

config = {}

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    exec(f.read(), config)


# ================== ОСНОВНЫЕ ПАРАМЕТРЫ ==================

DATA_PATH = config["DATA_PATH"]
MODEL_PATHS = config["MODEL_PATHS"]
TARGET_ORDER = config.get("TARGET_ORDER", ["MFR", "DD"])

BASE_RESULTS_DIR = config.get("BASE_RESULTS_DIR", "./results_realtime")

PREDICT_INTERVAL_MS = config.get("PREDICT_INTERVAL_MS", 20)
REPEAT_DATASET = config.get("REPEAT_DATASET", 1)
WARMUP_RUNS = config.get("WARMUP_RUNS", 5)
VERBOSE_EVERY = config.get("VERBOSE_EVERY", 40)

ROW_START = config.get("ROW_START", 0)
ROW_END = config.get("ROW_END", None)

FEATURES = config["FEATURES"]

HAS_TARGET = config.get("HAS_TARGET", False)
MFR_ERR_COL = config.get("MFR_ERR_COL", None)
DD_ERR_COL = config.get("DD_ERR_COL", None)

PLOT_TIME_COLUMN = config.get("PLOT_TIME_COLUMN", "cycle_ms")
SAVE_PLOTS = config.get("SAVE_PLOTS", True)
SAVE_FULL_DEBUG_CSV = config.get("SAVE_FULL_DEBUG_CSV", False)

NORMALIZATION_CONFIG = config.get("NORMALIZATION_CONFIG", {})

DEFAULT_USE_NORMALIZATION = config.get("DEFAULT_USE_NORMALIZATION", False)
DEFAULT_USE_POLY = config.get("DEFAULT_USE_POLY", False)


# ================== ЗАГРУЗКА ДАТАСЕТА ==================

df_full = pd.read_excel(DATA_PATH)
df = df_full.iloc[ROW_START:ROW_END].reset_index(drop=True)

if len(df) == 0:
    raise ValueError("После применения ROW_START/ROW_END датасет не содержит строк.")

os.makedirs(BASE_RESULTS_DIR, exist_ok=True)


# ================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==================

def get_target_col(target):
    target = target.upper()

    if target == "MFR":
        return MFR_ERR_COL

    if target == "DD":
        return DD_ERR_COL

    raise ValueError("TARGET должен быть 'MFR' или 'DD'.")


def get_runtime_config(model_path):
    model_label = Path(model_path).stem

    candidates = [
        model_label,
        model_label.replace("_MFR", ""),
        model_label.replace("_DD", ""),
        model_label.split("_dataset")[0],
    ]

    runtime_config = {
        "use_normalization": DEFAULT_USE_NORMALIZATION,
        "use_poly": DEFAULT_USE_POLY,
        "poly_degree": None,
        "x_scaler_path": None,
        "y_scaler_path": None,
        "matched_key": None
    }

    for key in candidates:
        if key in NORMALIZATION_CONFIG:
            runtime_config.update(NORMALIZATION_CONFIG[key])
            runtime_config["matched_key"] = key
            return runtime_config

    return runtime_config


def make_features(row):
    return [float(row.iloc[col]) for col in FEATURES]


def prepare_poly_transformer(use_poly, poly_degree, n_features):
    if not use_poly:
        return None

    if poly_degree is None:
        raise ValueError("Для use_poly=True нужно указать poly_degree.")

    poly = PolynomialFeatures(
        degree=poly_degree,
        include_bias=False
    )

    poly.fit(np.zeros((1, n_features)))

    return poly


def predict_one(features, model, x_scaler, y_scaler, poly, use_normalization):
    compute_start = time.perf_counter()

    x = np.array(features, dtype=float).reshape(1, -1)

    if poly is not None:
        x = poly.transform(x)

    if use_normalization:
        x_input = x_scaler.transform(x)
    else:
        x_input = x

    inference_start = time.perf_counter()
    raw_prediction = model.predict(x_input)
    inference_ms = (time.perf_counter() - inference_start) * 1000

    if use_normalization and y_scaler is not None:
        prediction = y_scaler.inverse_transform(
            np.asarray(raw_prediction).reshape(-1, 1)
        )[0][0]
    else:
        prediction = np.asarray(raw_prediction).reshape(-1)[0]

    compute_ms = (time.perf_counter() - compute_start) * 1000

    return float(prediction), inference_ms, compute_ms


def get_stats(values):
    values = np.asarray(values, dtype=float)

    return {
        "mean": np.mean(values),
        "median": np.median(values),
        "min": np.min(values),
        "max": np.max(values),
        "p95": np.percentile(values, 95),
        "p99": np.percentile(values, 99)
    }


def add_repeat_columns(results_df, dataset_len):
    results_df["repeat_number"] = (
        (results_df["measurement_id"] - 1) // dataset_len
    ) + 1

    results_df["measurement_in_repeat"] = (
        (results_df["measurement_id"] - 1) % dataset_len
    ) + 1

    results_df["repeat_measurement_range"] = results_df["repeat_number"].apply(
        lambda r: f"{(int(r) - 1) * dataset_len + 1}–{int(r) * dataset_len}"
    )

    return results_df


def save_export_table(results_df, output_csv, dataset_len):
    results_df = add_repeat_columns(results_df, dataset_len)

    if REPEAT_DATASET > 1:
        export_columns = [
            "repeat_number",
            "repeat_measurement_range",
            "measurement_in_repeat",
            "dataset_row",
            "inference_ms",
            "compute_ms",
            "cycle_ms",
            "period_ms",
            "start_jitter_ms",
            "lateness_ms",
            "deadline_miss",
        ]
    else:
        export_columns = [
            "measurement_in_repeat",
            "dataset_row",
            "inference_ms",
            "compute_ms",
            "cycle_ms",
            "period_ms",
            "start_jitter_ms",
            "lateness_ms",
            "deadline_miss",
        ]

    export_df = results_df[export_columns]
    export_df.to_csv(output_csv, index=False, encoding="utf-8")

    return results_df


def plot_detail_time(results_df, target, model_label, output_dir):
    if PLOT_TIME_COLUMN not in results_df.columns:
        raise ValueError(
            "PLOT_TIME_COLUMN должен быть одним из: "
            "'inference_ms', 'compute_ms', 'cycle_ms', "
            "'start_jitter_ms', 'lateness_ms'."
        )

    plt.figure(figsize=(14, 7))

    if REPEAT_DATASET > 1:
        for repeat_number, group in results_df.groupby("repeat_number"):
            plt.plot(
                group["measurement_in_repeat"],
                group[PLOT_TIME_COLUMN],
                marker="o",
                linewidth=1.3,
                markersize=3,
                alpha=0.85,
                label=f"Повтор {int(repeat_number)}"
            )

        plt.xlabel("Номер измерения внутри повтора")
    else:
        plt.plot(
            results_df["measurement_in_repeat"],
            results_df[PLOT_TIME_COLUMN],
            marker="o",
            linewidth=1.5,
            markersize=4,
            label=PLOT_TIME_COLUMN
        )

        plt.xlabel("Номер измерения")

    plt.title(
        f"Время выполнения модели\n"
        f"Цель: {target} | Модель: {model_label} | Метрика: {PLOT_TIME_COLUMN}",
        fontsize=14,
        fontweight="bold"
    )

    plt.ylabel("Время, мс")
    plt.grid(True)

    y_max = results_df[PLOT_TIME_COLUMN].max() * 1.2

    if y_max <= 0:
        y_max = 1

    plt.ylim(0, y_max)

    if REPEAT_DATASET <= 15:
        plt.legend()

    plt.tight_layout()

    plot_path = os.path.join(
        output_dir,
        f"detail_time_{model_label}_{PLOT_TIME_COLUMN}.png"
    )

    if SAVE_PLOTS:
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")

    plt.close()

    return plot_path


def plot_period_check(results_df, target, model_label, output_dir):
    plt.figure(figsize=(14, 7))

    if REPEAT_DATASET > 1:
        for repeat_number, group in results_df.groupby("repeat_number"):
            plt.plot(
                group["measurement_in_repeat"],
                group[PLOT_TIME_COLUMN],
                marker="o",
                linewidth=1.2,
                markersize=3,
                alpha=0.8,
                label=f"Повтор {int(repeat_number)}"
            )

        plt.xlabel("Номер измерения внутри повтора")
    else:
        plt.plot(
            results_df["measurement_in_repeat"],
            results_df[PLOT_TIME_COLUMN],
            marker="o",
            linewidth=1.5,
            markersize=4,
            label=PLOT_TIME_COLUMN
        )

        plt.xlabel("Номер измерения")

    plt.axhline(
        y=PREDICT_INTERVAL_MS,
        linestyle="--",
        linewidth=1.5,
        label=f"Период опроса: {PREDICT_INTERVAL_MS} мс"
    )

    plt.title(
        f"Проверка выполнения в реальном времени\n"
        f"Цель: {target} | Модель: {model_label} | Метрика: {PLOT_TIME_COLUMN}",
        fontsize=14,
        fontweight="bold"
    )

    plt.ylabel("Время, мс")
    plt.grid(True)

    if REPEAT_DATASET <= 15:
        plt.legend()
    else:
        plt.legend(["Период опроса"])

    plt.tight_layout()

    plot_path = os.path.join(
        output_dir,
        f"period_check_{model_label}_{PLOT_TIME_COLUMN}.png"
    )

    if SAVE_PLOTS:
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")

    plt.close()

    return plot_path


def save_summary(results_df, predictions, y_true, target, model_label, output_dir):
    inference_stats = get_stats(results_df["inference_ms"])
    compute_stats = get_stats(results_df["compute_ms"])
    cycle_stats = get_stats(results_df["cycle_ms"])
    jitter_stats = get_stats(results_df["start_jitter_ms"])
    lateness_stats = get_stats(results_df["lateness_ms"])

    deadline_misses = int(results_df["deadline_miss"].sum())
    total_measurements = len(results_df)

    summary = {
        "target": target,
        "model": model_label,
        "measurements": total_measurements,
        "dataset_rows": len(df),
        "repeat_dataset": REPEAT_DATASET,
        "period_ms": PREDICT_INTERVAL_MS,
        "deadline_misses": deadline_misses,
        "deadline_miss_percent": deadline_misses / total_measurements * 100,

        "inference_mean_ms": inference_stats["mean"],
        "inference_median_ms": inference_stats["median"],
        "inference_min_ms": inference_stats["min"],
        "inference_max_ms": inference_stats["max"],
        "inference_p95_ms": inference_stats["p95"],
        "inference_p99_ms": inference_stats["p99"],

        "compute_mean_ms": compute_stats["mean"],
        "compute_median_ms": compute_stats["median"],
        "compute_min_ms": compute_stats["min"],
        "compute_max_ms": compute_stats["max"],
        "compute_p95_ms": compute_stats["p95"],
        "compute_p99_ms": compute_stats["p99"],

        "cycle_mean_ms": cycle_stats["mean"],
        "cycle_median_ms": cycle_stats["median"],
        "cycle_min_ms": cycle_stats["min"],
        "cycle_max_ms": cycle_stats["max"],
        "cycle_p95_ms": cycle_stats["p95"],
        "cycle_p99_ms": cycle_stats["p99"],

        "start_jitter_mean_ms": jitter_stats["mean"],
        "start_jitter_max_ms": jitter_stats["max"],

        "lateness_mean_ms": lateness_stats["mean"],
        "lateness_max_ms": lateness_stats["max"],

        "max_cycle_less_than_period": bool(cycle_stats["max"] < PREDICT_INTERVAL_MS),
        "p99_cycle_less_than_80_percent_period": bool(cycle_stats["p99"] < 0.8 * PREDICT_INTERVAL_MS),
        "realtime_passed": bool(cycle_stats["max"] < PREDICT_INTERVAL_MS and deadline_misses == 0)
    }

    if HAS_TARGET and y_true is not None and len(y_true) == len(predictions):
        mae = mean_absolute_error(y_true, predictions)
        mse = mean_squared_error(y_true, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, predictions)

        summary.update({
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        })

    summary_df = pd.DataFrame([summary])

    summary_csv = os.path.join(output_dir, f"summary_{model_label}.csv")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")

    return summary_df, summary_csv


# ================== ОСНОВНАЯ ФУНКЦИЯ ТЕСТИРОВАНИЯ ==================

def run_target_test(target):
    target = target.upper()

    if target not in MODEL_PATHS:
        raise KeyError(f"Для цели {target} не указан путь в MODEL_PATHS.")

    model_path = MODEL_PATHS[target]
    model_label = Path(model_path).stem

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Файл модели не найден: {model_path}")

    output_dir = os.path.join(BASE_RESULTS_DIR, target, model_label)
    os.makedirs(output_dir, exist_ok=True)

    output_csv = os.path.join(output_dir, f"realtime_results_{model_label}.csv")
    full_debug_csv = os.path.join(output_dir, f"full_debug_{model_label}.csv")

    runtime_config = get_runtime_config(model_path)

    use_normalization = runtime_config.get("use_normalization", DEFAULT_USE_NORMALIZATION)
    use_poly = runtime_config.get("use_poly", DEFAULT_USE_POLY)
    poly_degree = runtime_config.get("poly_degree", None)

    x_scaler_path = runtime_config.get("x_scaler_path", None)
    y_scaler_path = runtime_config.get("y_scaler_path", None)

    model = joblib.load(model_path)

    if use_normalization:
        if x_scaler_path is None or y_scaler_path is None:
            raise ValueError(
                f"Для модели {model_label} включена нормализация, "
                f"но не указаны x_scaler_path/y_scaler_path."
            )

        x_scaler = joblib.load(x_scaler_path)
        y_scaler = joblib.load(y_scaler_path)
    else:
        x_scaler = None
        y_scaler = None

    poly = prepare_poly_transformer(
        use_poly=use_poly,
        poly_degree=poly_degree,
        n_features=len(FEATURES)
    )

    target_col = get_target_col(target)

    if HAS_TARGET:
        if target_col is None:
            raise ValueError(f"Для цели {target} не указан столбец target.")

        y_base = df.iloc[:, target_col].values.astype(float)
    else:
        y_base = None

    first_features = make_features(df.iloc[0])

    for _ in range(WARMUP_RUNS):
        predict_one(
            features=first_features,
            model=model,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            poly=poly,
            use_normalization=use_normalization
        )

    total_measurements = len(df) * REPEAT_DATASET
    period_s = PREDICT_INTERVAL_MS / 1000

    print("\n" + "=" * 90)
    print(f"ТЕСТ МОДЕЛИ В РЕЖИМЕ РЕАЛЬНОГО ВРЕМЕНИ: {target}")
    print("=" * 90)
    print(f"Модель: {model_path}")
    print(f"Название модели: {model_label}")
    print(f"Конфигурация нормализации: {runtime_config.get('matched_key')}")
    print(f"Нормализация: {'включена' if use_normalization else 'выключена'}")
    print(f"Полиномиальные признаки: {'включены' if use_poly else 'выключены'}")
    print(f"Степень полинома: {poly_degree if use_poly else '-'}")
    print(f"Строк в датасете: {len(df)}")
    print(f"Повторов датасета: {REPEAT_DATASET}")
    print(f"Всего измерений: {total_measurements}")
    print(f"Период опроса: {PREDICT_INTERVAL_MS} мс")
    print(f"Папка результатов: {output_dir}")
    print("=" * 90)

    results = []
    predictions = []
    y_true_repeated = []

    test_start = time.perf_counter()

    for measurement_id in range(total_measurements):
        scheduled_start = test_start + measurement_id * period_s

        now = time.perf_counter()

        if now < scheduled_start:
            time.sleep(scheduled_start - now)

        actual_start = time.perf_counter()
        start_jitter_ms = (actual_start - scheduled_start) * 1000

        row_id = measurement_id % len(df)
        row = df.iloc[row_id]

        cycle_start = time.perf_counter()

        features = make_features(row)

        prediction, inference_ms, compute_ms = predict_one(
            features=features,
            model=model,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            poly=poly,
            use_normalization=use_normalization
        )

        cycle_ms = (time.perf_counter() - cycle_start) * 1000

        deadline_time = scheduled_start + period_s
        finish_time = time.perf_counter()

        lateness_ms = max(0.0, (finish_time - deadline_time) * 1000)
        deadline_miss = int(finish_time > deadline_time)

        result_row = {
            "measurement_id": measurement_id + 1,
            "dataset_row": row_id + ROW_START,
            "prediction": prediction,
            "inference_ms": inference_ms,
            "compute_ms": compute_ms,
            "cycle_ms": cycle_ms,
            "period_ms": PREDICT_INTERVAL_MS,
            "start_jitter_ms": start_jitter_ms,
            "lateness_ms": lateness_ms,
            "deadline_miss": deadline_miss,
        }

        for j, value in enumerate(features):
            result_row[f"feature_{j + 1}"] = value

        results.append(result_row)
        predictions.append(prediction)

        if HAS_TARGET and y_base is not None:
            y_true_repeated.append(y_base[row_id])

        if VERBOSE_EVERY and (measurement_id + 1) % VERBOSE_EVERY == 0:
            print(
                f"{measurement_id + 1}/{total_measurements} | "
                f"inference={inference_ms:.6f} ms | "
                f"compute={compute_ms:.6f} ms | "
                f"cycle={cycle_ms:.6f} ms | "
                f"miss={deadline_miss}"
            )

    results_df = pd.DataFrame(results)
    results_df = save_export_table(
        results_df=results_df,
        output_csv=output_csv,
        dataset_len=len(df)
    )

    if SAVE_FULL_DEBUG_CSV:
        results_df.to_csv(full_debug_csv, index=False, encoding="utf-8")

    predictions = np.asarray(predictions, dtype=float)

    if HAS_TARGET and len(y_true_repeated) == len(predictions):
        y_true_repeated = np.asarray(y_true_repeated, dtype=float)
    else:
        y_true_repeated = None

    summary_df, summary_csv = save_summary(
        results_df=results_df,
        predictions=predictions,
        y_true=y_true_repeated,
        target=target,
        model_label=model_label,
        output_dir=output_dir
    )

    detail_plot_path = plot_detail_time(
        results_df=results_df,
        target=target,
        model_label=model_label,
        output_dir=output_dir
    )

    period_plot_path = plot_period_check(
        results_df=results_df,
        target=target,
        model_label=model_label,
        output_dir=output_dir
    )

    print("\n--- ИТОГИ ---")
    print(summary_df.T)
    print(f"\nТаблица времени: {output_csv}")
    print(f"Сводка: {summary_csv}")
    print(f"Детальный график: {detail_plot_path}")
    print(f"График проверки периода: {period_plot_path}")

    return summary_df


# ================== ПОСЛЕДОВАТЕЛЬНЫЙ ЗАПУСК MFR -> DD ==================

all_summaries = []

for target in TARGET_ORDER:
    summary_df = run_target_test(target)
    all_summaries.append(summary_df)

all_summaries_df = pd.concat(all_summaries, ignore_index=True)

combined_summary_csv = os.path.join(
    BASE_RESULTS_DIR,
    "combined_realtime_summary.csv"
)

all_summaries_df.to_csv(combined_summary_csv, index=False, encoding="utf-8")

print("\n" + "=" * 90)
print("ОБЩАЯ СВОДКА ПО ВСЕМ ЦЕЛЕВЫМ ПЕРЕМЕННЫМ")
print("=" * 90)
print(all_summaries_df)
print(f"\nОбщая сводка сохранена: {combined_summary_csv}")
