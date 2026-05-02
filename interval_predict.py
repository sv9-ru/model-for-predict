import time
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

print(f"Готово. Результаты сохранены в {OUTPUT_CSV}")
