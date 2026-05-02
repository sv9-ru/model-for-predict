import sys
import time
import joblib
import numpy as np

# ================== ЗАГРУЗКА CONFIG.TXT ==================
config = {}
with open("config.txt", "r", encoding="utf-8") as f:
    exec(f.read(), config)

MODEL_PATH = config["MODEL_PATH"]
TARGET = config["TARGET"].upper()
FEATURES = config["FEATURES"]

USE_NORMALIZATION = config.get("USE_NORMALIZATION", False)
X_SCALER_PATH = config.get("X_SCALER_PATH", None)
Y_SCALER_PATH = config.get("Y_SCALER_PATH", None)

# ================== ПРОВЕРКА TARGET ==================
if TARGET not in ("MFR", "DD"):
    raise ValueError("TARGET должен быть 'MFR' или 'DD'")

# ================== ПРОВЕРКА ВВОДА ==================
if len(sys.argv) - 1 != len(FEATURES):
    print(f"Ошибка: нужно ввести {len(FEATURES)} признака")
    print("Пример:")
    print("python single_predict.py 881 3.2")
    sys.exit(1)

# ================== ЧТЕНИЕ ПРИЗНАКОВ ==================
X = np.array(
    [float(value) for value in sys.argv[1:]],
    dtype=float
).reshape(1, -1)

# ================== ЗАГРУЗКА МОДЕЛИ ==================
model = joblib.load(MODEL_PATH)

# ================== НОРМАЛИЗАЦИЯ X ==================
if USE_NORMALIZATION:
    if X_SCALER_PATH is None or Y_SCALER_PATH is None:
        raise ValueError("Для нормализации нужны X_SCALER_PATH и Y_SCALER_PATH")

    x_scaler = joblib.load(X_SCALER_PATH)
    y_scaler = joblib.load(Y_SCALER_PATH)

    X_input = x_scaler.transform(X)
else:
    y_scaler = None
    X_input = X

# ================== ПРЕДСКАЗАНИЕ ==================
start_time = time.perf_counter()

raw_prediction = model.predict(X_input)

prediction_time_ms = (time.perf_counter() - start_time) * 1000

# ================== ОБРАТНАЯ НОРМАЛИЗАЦИЯ Y ==================
if USE_NORMALIZATION:
    prediction = y_scaler.inverse_transform(
        np.asarray(raw_prediction).reshape(-1, 1)
    )[0][0]
else:
    prediction = np.asarray(raw_prediction).reshape(-1)[0]

# ================== ВЫВОД В ОДНУ СТРОКУ ==================
print(f"{prediction:.6f} {prediction_time_ms:.6f}")
