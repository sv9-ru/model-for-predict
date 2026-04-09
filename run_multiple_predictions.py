import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import shutil

# Загрузка конфигурации
with open('config.txt', 'r') as f:
    exec(f.read())

print("="*60)
print("МНОГОКРАТНЫЙ ЗАПУСК ПРЕДСКАЗАНИЙ")
print("="*60)
print(f"Количество запусков: {NUM_RUNS}")
print(f"Нормализация: {'Включена' if USE_NORMALIZATION else 'Выключена'}")
print("="*60)

# Хранилище для времени всех запусков
all_times = []
successful_runs = 0

for run_num in range(1, NUM_RUNS + 1):
    print(f"\n--- ЗАПУСК {run_num}/{NUM_RUNS} ---")

    try:
        # Обновляем config.txt с текущими настройками нормализации
        with open('config.txt', 'r') as f:
            config_content = f.read()

        # Обновляем параметр нормализации в конфиге
        config_lines = config_content.split('\n')
        new_config_lines = []
        for line in config_lines:
            if line.strip().startswith('USE_NORMALIZATION'):
                new_config_lines.append(f'USE_NORMALIZATION = {USE_NORMALIZATION}')
            else:
                new_config_lines.append(line)

        with open('config.txt', 'w') as f:
            f.write('\n'.join(new_config_lines))

        # Запускаем predict_script.py
        result = subprocess.run(
            ['python', 'predict_script.py'],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            print(f"  ✗ Ошибка: {result.stderr[:200]}")
            continue

        # Читаем результаты
        if Path(OUTPUT_CSV).exists():
            df_results = pd.read_csv(OUTPUT_CSV)

            # Извлекаем время предсказания
            if 'Время_мс' in df_results.columns:
                times_ms = df_results['Время_мс'].values
                all_times.append(times_ms)
                successful_runs += 1
                print(f"  ✓ Запуск {run_num} завершен, {len(times_ms)} измерений")
                print(f"    Среднее время: {np.mean(times_ms):.3f} мс")
            else:
                print(f"  ✗ Столбец 'Время_мс' не найден в {OUTPUT_CSV}")
        else:
            print(f"  ✗ Файл {OUTPUT_CSV} не найден")

    except subprocess.TimeoutExpired:
        print(f"  ✗ Запуск {run_num} превысил таймаут")
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")

print("\n" + "="*60)
print("ПОСТРОЕНИЕ ГРАФИКА СРАВНЕНИЯ")
print("="*60)

if successful_runs < 2:
    print(f"Предупреждение: Успешных запусков: {successful_runs}. Для сравнения нужно минимум 2 запуска.")
    if successful_runs == 0:
        sys.exit(1)

# Находим максимальную длину среди всех запусков
max_length = max([len(times) for times in all_times])
print(f"Максимальное количество измерений: {max_length}")

# Выравниваем все массивы до одинаковой длины (заполняем NaN)
aligned_times = []
for i, times in enumerate(all_times):
    if len(times) < max_length:
        # Дополняем NaN для коротких массивов
        padded = np.pad(times, (0, max_length - len(times)), constant_values=np.nan)
        aligned_times.append(padded)
    else:
        aligned_times.append(times)

# Номер измерения (X) - от 0 до N-1
x = np.arange(max_length)  # от 0 до max_length-1

# ПОСТРОЕНИЕ ГРАФИКА
plt.figure(figsize=(14, 8))

# Цвета и маркеры для разных запусков
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

for i, times in enumerate(aligned_times[:NUM_RUNS]):
    # Используем разные цвета и маркеры для каждого запуска
    color = colors[i % len(colors)]
    marker = markers[i % len(markers)]
    label = f'Run_{i+1}'

    # Строим линию только для не-NaN значений
    valid_mask = ~np.isnan(times)
    plt.plot(x[valid_mask], times[valid_mask],
             marker=marker, linestyle='-', linewidth=1.5,
             markersize=3, color=color, alpha=0.7, label=label)

# Настройки графика
plt.xlabel('Номер измерения', fontsize=14, fontweight='bold')
plt.ylabel('Время предсказания (мс)', fontsize=14, fontweight='bold')
plt.title(f'Сравнение времени выполнения предсказаний (нормализация: {"ON" if USE_NORMALIZATION else "OFF"})',
          fontsize=16, fontweight='bold')
plt.legend(fontsize=10, loc='best')
plt.grid(True, linestyle='--', alpha=0.6)

# Настройка подписей оси X (показываем каждый 20-й индекс, если много точек)
if max_length > 50:
    step = max(1, max_length // 20)
    plt.xticks(x[::step], rotation=45)
else:
    plt.xticks(x, rotation=45)

plt.tight_layout()
plt.savefig('./comparison_times.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("✓ График сохранен: comparison_times.png")

# СОХРАНЯЕМ ВСЕ ДАННЫЕ В CSV
comparison_df = pd.DataFrame()
for i, times in enumerate(aligned_times):
    comparison_df[f'Run_{i+1}_time_ms'] = times

comparison_df.insert(0, 'measurement_number', x)
comparison_df.to_csv('./all_runs_comparison.csv', index=False)
print(f"✓ Данные сохранены в all_runs_comparison.csv")

# СТАТИСТИКА
print("\n" + "="*60)
print("СТАТИСТИКА ПО ЗАПУСКАМ")
print("="*60)

stats_data = []
for i, times in enumerate(aligned_times[:successful_runs]):
    valid_times = times[~np.isnan(times)]
    stats_data.append({
        'Run': i+1,
        'Count': len(valid_times),
        'Mean_ms': np.mean(valid_times),
        'Std_ms': np.std(valid_times),
        'Min_ms': np.min(valid_times),
        'Max_ms': np.max(valid_times),
        'Median_ms': np.median(valid_times),
        'P95_ms': np.percentile(valid_times, 95),
        'P99_ms': np.percentile(valid_times, 99)
    })

stats_df = pd.DataFrame(stats_data)
print(stats_df.to_string(index=False))
stats_df.to_csv('./runs_statistics.csv', index=False)
print("\n✓ Статистика сохранена в runs_statistics.csv")

print("\n" + "="*60)
print("ГОТОВО!")
print("="*60)
