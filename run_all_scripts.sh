#!/bin/bash

# run_all_scripts.sh - запускает все скрипты предсказаний

echo "Запуск predict_script.py"
python predict_script.py

echo "Запуск compare_models_performance.py"
python compare_models_performance.py

echo "Запуск interval_predict.py"
python interval_predict.py

echo "Запуск run_multiple_predictions.py"
python run_multiple_predictions.py

echo "Все скрипты завершены"
