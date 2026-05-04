
import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ================= CONFIG =================
with open('config.txt', 'r') as f:
    exec(f.read())


# ================= UTILS =================

def get_model_name(path):
    return Path(path).stem


def find_model_files(models_dir):
    model_files = []
    for root, _, files in os.walk(models_dir):
        for file in files:
            if file.endswith((".pkl", ".joblib")):
                model_files.append(os.path.join(root, file))
    return sorted(model_files)


def model_matches_target(model_name, target):
    name = model_name.upper()

    if target == "MFR":
        return "MFR" in name and "DD" not in name
    if target == "DD":
        return "DD" in name

    return False


def get_model_settings(model_name):
    if model_name in NORMALIZATION_CONFIG:
        return NORMALIZATION_CONFIG[model_name]

    for key, value in NORMALIZATION_CONFIG.items():
        if key.lower() in model_name.lower():
            return value

    return {
        "use_normalization": DEFAULT_USE_NORMALIZATION
    }


def predict_one(x, model, x_scaler, y_scaler, use_norm):
    compute_start = time.perf_counter()

    x = np.array(x).reshape(1, -1)

    if use_norm:
        x_input = x_scaler.transform(x)
    else:
        x_input = x

    inf_start = time.perf_counter()
    raw = model.predict(x_input)
    inference_ms = (time.perf_counter() - inf_start) * 1000

    if use_norm:
        pred = y_scaler.inverse_transform(np.array(raw).reshape(-1, 1))[0][0]
    else:
        pred = float(raw[0])

    compute_ms = (time.perf_counter() - compute_start) * 1000

    return pred, inference_ms, compute_ms


def stats(arr):
    arr = np.array(arr)
    return {
        "mean": np.mean(arr),
        "p95": np.percentile(arr, 95),
        "p99": np.percentile(arr, 99),
        "max": np.max(arr)
    }


# ================= MAIN FUNCTION =================

def run_target(target):

    print(f"\n{'='*90}")
    print(f"TARGET = {target}")
    print(f"{'='*90}")

    # ===== DATA =====
    df = pd.read_excel(DATA_PATH)

    X = df.iloc[ROW_START:, FEATURES].values.astype(float)

    mfr_true = df.iloc[ROW_START:, [MFR_ERR_COL]].values.flatten()
    dd_true = df.iloc[ROW_START:, [DD_ERR_COL]].values.flatten()

    y_true = mfr_true if target == "MFR" else dd_true

    # ===== MODELS =====
    all_models = find_model_files(MODELS_DIR)

    model_files = [
        m for m in all_models
        if model_matches_target(get_model_name(m), target)
    ]

    if not model_files:
        print("❌ Нет моделей")
        return None

    print(f"Найдено моделей: {len(model_files)}")

    # ===== OUTPUT DIR =====
    target_dir = os.path.join(BASE_RESULTS_DIR, target)
    os.makedirs(target_dir, exist_ok=True)

    results_summary = []
    time_series = {}

    # ===== BENCHMARK =====
    for model_path in model_files:
        model_name = get_model_name(model_path)

        print(f"\n→ {model_name}")

        model = joblib.load(model_path)
        cfg = get_model_settings(model_name)

        use_norm = cfg.get("use_normalization", False)

        if use_norm:
            x_scaler = joblib.load(cfg["x_scaler_path"])
            y_scaler = joblib.load(cfg["y_scaler_path"])
        else:
            x_scaler = None
            y_scaler = None

        # ===== WARMUP =====
        for _ in range(10):
            predict_one(X[0], model, x_scaler, y_scaler, use_norm)

        predictions = []
        cycle_times = []
        deadline_miss = 0

        period_s = PREDICT_INTERVAL_MS / 1000
        start_time = time.perf_counter()

        for i in range(len(X)):
            scheduled = start_time + i * period_s

            now = time.perf_counter()
            if now < scheduled:
                time.sleep(scheduled - now)

            cycle_start = time.perf_counter()

            pred, _, _ = predict_one(
                X[i], model, x_scaler, y_scaler, use_norm
            )

            cycle_ms = (time.perf_counter() - cycle_start) * 1000

            finish = time.perf_counter()
            if finish > scheduled + period_s:
                deadline_miss += 1

            predictions.append(pred)
            cycle_times.append(cycle_ms)

        predictions = np.array(predictions)

        # ===== METRICS =====
        mse = mean_squared_error(y_true, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, predictions)
        r2 = r2_score(y_true, predictions)

        cycle_stat = stats(cycle_times)

        results_summary.append({
            "model": model_name,
            "target": target,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "cycle_mean_ms": cycle_stat["mean"],
            "cycle_p95_ms": cycle_stat["p95"],
            "cycle_p99_ms": cycle_stat["p99"],
            "cycle_max_ms": cycle_stat["max"],
            "deadline_miss_%": deadline_miss / len(X) * 100,
            "realtime_ok": cycle_stat["max"] < PREDICT_INTERVAL_MS and deadline_miss == 0
        })

        time_series[model_name] = cycle_times

    # ===== SAVE TABLE =====
    df_summary = pd.DataFrame(results_summary)
    df_summary = df_summary.sort_values(by=["MAE", "cycle_mean_ms"])

    metrics_path = os.path.join(target_dir, f"metrics_{target}.csv")
    df_summary.to_csv(metrics_path, index=False)

    print("\nRESULT:")
    print(df_summary)

    # ================= PLOT (FIXED) =================
    plt.figure(figsize=(15, 7))

    markers = [
        "o", "s", "^", "D", "v", "<", ">", "p", "*", "h",
        "X", "P", "8", "d", "|", "_"
    ]

    # более насыщенные цвета (вместо pastel tab20)
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_series)))

    for idx, (name, series) in enumerate(time_series.items()):
        plt.plot(
            series,
            label=name,

            # 🔥 усиливаем визуал
            linewidth=2.2,          # толще линии
            alpha=1.0,              # без прозрачности
            color=colors[idx % len(colors)],

            marker=markers[idx % len(markers)],
            markevery=max(1, len(series)//30),
            markersize=6,
        )

    plt.axhline(
        y=PREDICT_INTERVAL_MS,
        linestyle="--",
        linewidth=1.5,
        color="red",
        label="deadline"
    )

    plt.title(f"REAL PERFORMANCE | {target}", fontsize=13, fontweight="bold")
    plt.xlabel("measurement")
    plt.ylabel("cycle ms")
    plt.grid(True, alpha=0.4)

    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.tight_layout()

    plot_path = os.path.join(target_dir, f"time_{target}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved: {metrics_path}")
    print(f"Graph: {plot_path}")

    return df_summary


# ================= RUN =================

all_results = []

for target in ["MFR", "DD"]:
    result = run_target(target)
    if result is not None:
        all_results.append(result)

if all_results:
    combined = pd.concat(all_results, ignore_index=True)
    combined_path = os.path.join(BASE_RESULTS_DIR, "combined_metrics.csv")
    combined.to_csv(combined_path, index=False)

    print("\n=== COMBINED ===")
    print(combined)
    print(f"\nSaved: {combined_path}")
