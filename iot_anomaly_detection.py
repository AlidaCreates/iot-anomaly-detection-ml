
# @title 1. УСТАНОВКА И ИМПОРТ БИБЛИОТЕК { display-mode: "form" }

# Установка необходимых библиотек
!pip install -q kagglehub imbalanced-learn tensorflow xgboost scikit-learn pandas numpy matplotlib seaborn joblib

# Импорты
import kagglehub
import pandas as pd
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import json
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, confusion_matrix, roc_curve, auc)
from xgboost import XGBClassifier
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

print("✅ Все библиотеки установлены и импортированы!")

# @title 2. ПОДКЛЮЧЕНИЕ К GOOGLE DRIVE (для сохранения результатов) { display-mode: "form" }

from google.colab import drive
drive.mount('/content/drive')

# Создаем папку для результатов
RESULTS_PATH = '/content/drive/MyDrive/IoT_Anomaly_Detection_Results/'
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(os.path.join(RESULTS_PATH, 'models'), exist_ok=True)
os.makedirs(os.path.join(RESULTS_PATH, 'figures'), exist_ok=True)

print(f"✅ Результаты будут сохранены в: {RESULTS_PATH}")

# @title 3. НАСТРОЙКА KAGGLE

import json

# Вставьте ваш kaggle.json содержимое сюда
kaggle_credentials = {"username":"alidamirmagaliyeva",
                      "key":"6da2e99e664e2b36e335dc7abadc37a2"}

# Сохраняем credentials
os.makedirs('/root/.kaggle', exist_ok=True)
with open('/root/.kaggle/kaggle.json', 'w') as f:
    json.dump(kaggle_credentials, f)

!chmod 600 /root/.kaggle/kaggle.json
print("✅ Kaggle настроен!")


# @title 4. ЗАГРУЗКА ДАТАСЕТА TON_IoT Network Dataset { display-mode: "form" }

print("📥 Загрузка датасета TON_IoT Network Dataset с Google Drive...")

try:
    # Монтируем Google Drive
    from google.colab import drive
    drive.mount('/content/drive')

    # Путь к папке с данными - исправленный
    drive_path = '/content/drive/MyDrive/Train_Test_Network'

    print(f"🔍 Проверяем путь: {drive_path}")

    # Проверяем существование пути
    if os.path.exists(drive_path):
        print(f"✅ Папка найдена: {drive_path}")

        # Создаем локальную папку для данных
        local_data_path = '/content/TON_IoT Network Dataset_data'
        os.makedirs(local_data_path, exist_ok=True)

        # Копируем все CSV файлы
        files_copied = 0
        print("
📋 Найденные файлы:")

        for file in os.listdir(drive_path):
            if file.endswith('.csv'):
                src = os.path.join(drive_path, file)
                dst = os.path.join(local_data_path, file)

                # Копируем файл
                shutil.copy2(src, dst)
                files_copied += 1

                # Получаем размер файла
                size_mb = os.path.getsize(src) / (1024 * 1024)
                print(f"  ✅ {file} ({size_mb:.2f} MB) - скопирован")
            else:
                print(f"  ⏺ {file} (не CSV)")

        print(f"
✅ Скопировано {files_copied} CSV файлов")
        path = local_data_path

        # Просмотр скопированных файлов
        print("
📋 Файлы в локальной папке:")
        files = os.listdir(path)
        for f in sorted(files):
            size = os.path.getsize(os.path.join(path, f)) / (1024 * 1024)
            print(f"  • {f} ({size:.2f} MB)")

    else:
        print(f"❌ Папка {drive_path} не найдена!")

        # Показываем содержимое MyDrive для диагностики
        print("
🔍 Содержимое MyDrive:")
        try:
            mydrive_contents = os.listdir('/content/drive/MyDrive')
            for item in mydrive_contents:
                item_path = f'/content/drive/MyDrive/{item}'
                if os.path.isdir(item_path):
                    print(f"  📁 {item}/")
                    # Показываем первые 5 файлов в каждой папке
                    try:
                        subitems = os.listdir(item_path)[:5]
                        for subitem in subitems:
                            print(f"      📄 {subitem}")
                    except:
                        pass
                else:
                    print(f"  📄 {item}")
        except Exception as e:
            print(f"  Ошибка при чтении: {e}")

        raise Exception("Папка с данными не найдена")

except Exception as e:
    print(f"
❌ Ошибка загрузки: {e}")
    print("
🔄 Создаю синтетический датасет для тестирования...")

    from sklearn.datasets import make_classification
    import pandas as pd
    import numpy as np

    # Создаем синтетические данные, похожие на CICIDS2017
    n_samples = 10000
    n_features = 20

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )

    # Создаем имена признаков как в реальном датасете
    feature_names = [
        ' Destination Port',
        ' Flow Duration',
        ' Total Fwd Packets',
        ' Total Backward Packets',
        'Total Length of Fwd Packets',
        ' Total Length of Bwd Packets',
        ' Fwd Packet Length Max',
        ' Fwd Packet Length Min',
        ' Fwd Packet Length Mean',
        ' Fwd Packet Length Std',
        'Bwd Packet Length Max',
        ' Bwd Packet Length Min',
        ' Bwd Packet Length Mean',
        ' Bwd Packet Length Std',
        'Flow Bytes/s',
        ' Flow Packets/s',
        ' Flow IAT Mean',
        ' Flow IAT Std',
        ' Flow IAT Max',
        ' Flow IAT Min'
    ][:n_features]

    df = pd.DataFrame(X, columns=feature_names)
    df[' Label'] = y  # Добавляем колонку с метками как в оригинале

    # Сохраняем
    synthetic_path = '/content/synthetic_cicids2017.csv'
    df.to_csv(synthetic_path, index=False)

    print(f"✅ Создан синтетический датасет: {synthetic_path}")
    print(f"📊 Размер: {n_samples} строк, {n_features} признаков")
    print(f"📁 Файл сохранен для тестирования")

    path = '/content'

    # @title 2. АНАЛИЗ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ (LABEL) { display-mode: "form" }

print("="*70)
print("🏷️ АНАЛИЗ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ")
print("="*70)

# Проверяем наличие колонки с метками (разные варианты)
possible_label_cols = [' Label', 'label', 'Label', 'LABEL', ' class', 'Class', 'attack', 'Attack']

label_col = None
for col in possible_label_cols:
    if col in df.columns:
        label_col = col
        print(f"✅ Колонка с метками найдена: '{label_col}'")
        break

if label_col is None:
    # Если не нашли по точному совпадению, ищем частичное
    for col in df.columns:
        if 'label' in col.lower() or 'class' in col.lower() or 'attack' in col.lower():
            label_col = col
            print(f"✅ Найдена похожая колонка: '{label_col}'")
            break

if label_col:
    # Анализ распределения
    label_counts = df[label_col].value_counts()
    total = len(df)

    print(f"
📊 РАСПРЕДЕЛЕНИЕ КЛАССОВ:")
    print(f"
{'Класс':25} | {'Количество':12} | {'Процент':8}")
    print(f"{'-'*50}")

    for label, count in label_counts.items():
        percentage = count / total * 100
        print(f"{str(label)[:25]:25} | {count:12,} | {percentage:6.2f}%")

    # Анализ дисбаланса
    print(f"
📈 АНАЛИЗ ДИСБАЛАНСА:")
    if len(label_counts) > 1:
        majority = label_counts.iloc[0]
        minority = label_counts.iloc[-1]
        ratio = majority / minority

        print(f"   • Мажоритарный класс: {label_counts.index[0]} ({majority:,})")
        print(f"   • Миноритарный класс: {label_counts.index[-1]} ({minority:,})")
        print(f"   • Коэффициент дисбаланса: {ratio:.2f}:1")

        if ratio > 100:
            print(f"   ⚠️ КРИТИЧЕСКИЙ ДИСБАЛАНС")
        elif ratio > 20:
            print(f"   ⚠️ СИЛЬНЫЙ ДИСБАЛАНС")
        elif ratio > 5:
            print(f"   ⚠️ УМЕРЕННЫЙ ДИСБАЛАНС")
        else:
            print(f"   ✅ ХОРОШИЙ БАЛАНС")

    # Сохраняем информацию о колонке с метками
    print(f"
💾 Для следующих шагов используем колонку: '{label_col}'")

else:
    print(f"❌ Колонка с метками не найдена!")
    print(f"Доступные колонки: {df.columns.tolist()}")

    # @title 6. ПРЕДВАРИТЕЛЬНЫЙ АНАЛИЗ ДАННЫХ { display-mode: "form" }

print("📊 ПРЕДВАРИТЕЛЬНЫЙ АНАЛИЗ ДАННЫХ")
print("="*50)

# Основная информация
print(f"
📁 РАЗМЕР ДАТАСЕТА:")
print(f"   • Строк: {df.shape[0]:,}")
print(f"   • Колонок: {df.shape[1]}")

# Типы данных
print(f"
📋 ТИПЫ ДАННЫХ:")
dtypes_count = df.dtypes.value_counts()
for dtype, count in dtypes_count.items():
    print(f"   • {dtype}: {count} колонок")

# Информация о памяти
memory_usage = df.memory_usage(deep=True).sum() / (1024**2)  # в MB
print(f"
💾 ИСПОЛЬЗОВАНИЕ ПАМЯТИ:")
print(f"   • Всего: {memory_usage:.2f} MB")
print(f"   • На строку: {memory_usage/len(df)*1000:.2f} KB")

print(f"
📌 КОЛОНКИ:")
print(f"   Полный список ({len(df.columns)} колонок):")
for i, col in enumerate(df.columns):
    dtype = df[col].dtype
    non_null = df[col].count()
    null_pct = (len(df) - non_null) / len(df) * 100
    print(f"   {i+1:3}. {col[:50]:50} | Тип: {str(dtype):8} | Заполнено: {non_null:,} ({100-null_pct:.1f}%)")

# Поиск колонки с метками
print(f"
🏷️ ПОИСК КОЛОНКИ С МЕТКАМИ КЛАССОВ:")
label_col = None
potential_labels = []

for col in df.columns:
    col_lower = col.lower()
    if any(term in col_lower for term in ['label', 'class', 'attack', 'benign', 'malign', 'type']):
        potential_labels.append(col)
        unique_vals = df[col].nunique()
        print(f"   • Потенциальная: '{col}' (уникальных значений: {unique_vals})")

        # Проверяем, похоже ли на бинарную классификацию
        if unique_vals <= 10:  # Обычно классы атак <= 10
            if label_col is None:
                label_col = col

if label_col is None and potential_labels:
    label_col = potential_labels[0]
    print(f"
   ✅ Выбрана первая подходящая: '{label_col}'")
elif label_col is None:
    label_col = df.columns[-1]
    print(f"
   ⚠️ Метки не найдены, используем последнюю колонку: '{label_col}'")
else:
    print(f"
   ✅ Выбрана колонка: '{label_col}'")

# Анализ распределения классов
print(f"
📊 АНАЛИЗ РАСПРЕДЕЛЕНИЯ КЛАССОВ ({label_col}):")

if label_col in df.columns:
    # Получаем распределение
    class_dist = df[label_col].value_counts()
    total_samples = len(df)

    print(f"
   Всего классов: {len(class_dist)}")
    print(f"
   Топ-15 классов по количеству:")
    print(f"   {'№':3} | {'Класс':30} | {'Количество':12} | {'Процент':8} | {'Накоп. %':8}")
    print(f"   {'-'*70}")

    cum_percent = 0
    for i, (cls, count) in enumerate(class_dist.head(15).items()):
        percent = count / total_samples * 100
        cum_percent += percent
        cls_str = str(cls)[:30]  # Обрезаем длинные названия
        print(f"   {i+1:3} | {cls_str:30} | {count:12,} | {percent:6.2f}% | {cum_percent:6.2f}%")

    # Проверка на дисбаланс классов
    if len(class_dist) > 1:
        majority = class_dist.iloc[0]
        minority = class_dist.iloc[-1]
        imbalance_ratio = majority / minority

        print(f"
   📈 АНАЛИЗ БАЛАНСА КЛАССОВ:")
        print(f"      • Самый частый класс: {class_dist.index[0]} ({majority:,} samples)")
        print(f"      • Самый редкий класс: {class_dist.index[-1]} ({minority:,} samples)")
        print(f"      • Коэффициент дисбаланса: {imbalance_ratio:.2f}:1")

        if imbalance_ratio > 10:
            print(f"      ⚠️ Сильный дисбаланс классов! Рекомендуется использовать техники балансировки")
        elif imbalance_ratio > 3:
            print(f"      ⚠️ Умеренный дисбаланс классов")
        else:
            print(f"      ✅ Классы достаточно сбалансированы")

# Анализ пропущенных значений
print(f"
🔍 АНАЛИЗ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ:")
missing_total = df.isnull().sum().sum()
missing_cols = df.isnull().sum()[df.isnull().sum() > 0]

if len(missing_cols) > 0:
    print(f"   • Всего пропущенных значений: {missing_total:,}")
    print(f"   • Колонок с пропусками: {len(missing_cols)}")
    print(f"
   Топ-10 колонок с пропусками:")
    print(f"   {'№':3} | {'Колонка':40} | {'Пропуски':12} | {'Процент':8}")
    print(f"   {'-'*70}")

    for i, (col, missing) in enumerate(missing_cols.sort_values(ascending=False).head(10).items()):
        percent = missing / len(df) * 100
        print(f"   {i+1:3} | {col[:40]:40} | {missing:12,} | {percent:7.2f}%")
else:
    print(f"   ✅ Пропущенные значения отсутствуют!")

# Статистика числовых признаков
print(f"
📈 СТАТИСТИКА ЧИСЛОВЫХ ПРИЗНАКОВ:")
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f"   • Числовых признаков: {len(numeric_cols)}")

if len(numeric_cols) > 0:
    # Базовая статистика
    stats = df[numeric_cols].describe().T
    print(f"
   Статистика первых 5 числовых признаков:")
    print(f"   {'Признак':25} | {'Среднее':12} | {'Стд':10} | {'Мин':10} | {'Макс':10}")
    print(f"   {'-'*75}")

    for col in numeric_cols[:5]:
        mean = stats.loc[col, 'mean'] if col in stats.index else df[col].mean()
        std = stats.loc[col, 'std'] if col in stats.index else df[col].std()
        min_val = stats.loc[col, 'min'] if col in stats.index else df[col].min()
        max_val = stats.loc[col, 'max'] if col in stats.index else df[col].max()

        print(f"   {col[:25]:25} | {mean:11.2f} | {std:9.2f} | {min_val:9.2f} | {max_val:9.2f}")

# Анализ категориальных признаков
print(f"
🏷️ КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ:")
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
print(f"   • Категориальных признаков: {len(categorical_cols)}")

for col in categorical_cols[:5]:  # Первые 5
    unique_vals = df[col].nunique()
    print(f"   • {col}: {unique_vals} уникальных значений")

# Поиск бесконечных значений
print(f"
🔢 ПРОВЕРКА НА БЕСКОНЕЧНЫЕ ЗНАЧЕНИЯ:")
inf_cols = []
for col in numeric_cols:
    inf_count = np.isinf(df[col]).sum()
    if inf_count > 0:
        inf_cols.append((col, inf_count))

if inf_cols:
    print(f"   ⚠️ Найдены бесконечные значения:")
    for col, count in inf_cols[:5]:
        print(f"      • {col}: {count}")
else:
    print(f"   ✅ Бесконечные значения отсутствуют")

# Итоговое резюме
print(f"
{'='*50}")
print(f"📋 ИТОГОВОЕ РЕЗЮМЕ:")
print(f"   • Размер датасета: {df.shape[0]:,} x {df.shape[1]}")
print(f"   • Типы данных: {', '.join([f'{k}: {v}' for k, v in dtypes_count.items()])}")
print(f"   • Метки классов: '{label_col}' ({df[label_col].nunique()} классов)")
print(f"   • Пропущенные значения: {missing_total:,}")
print(f"   • Использование памяти: {memory_usage:.2f} MB")
print(f"   • Дисбаланс классов: {imbalance_ratio if 'imbalance_ratio' in locals() else 'N/A'}:1")

# @title 7. ПРЕДОБРАБОТКА ДАННЫХ { display-mode: "form" }

print("="*70)
print("🧹 ПРЕДОБРАБОТКА ДАННЫХ CICIDS2017")
print("="*70)

from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import numpy as np
import pandas as pd

# Определяем колонку с метками (из предыдущего шага)
if 'label_col' not in locals():
    # Если переменная не определена, ищем колонку с метками
    for col in df.columns:
        if 'label' in col.lower():
            label_col = col
            break
    else:
        label_col = df.columns[-1]

print(f"
🔍 Используем колонку с метками: '{label_col}'")

# Копируем данные
df_clean = df.copy()
print(f"
📊 Исходные данные: {df_clean.shape}")

# 1. УДАЛЕНИЕ ДУБЛИКАТОВ
print("
1️⃣ УДАЛЕНИЕ ДУБЛИКАТОВ")
print("-"*50)
initial_len = len(df_clean)
df_clean = df_clean.drop_duplicates()
duplicates_removed = initial_len - len(df_clean)
print(f"   • Удалено дубликатов: {duplicates_removed}")
print(f"   • Осталось записей: {len(df_clean):,}")

# 2. ВЫДЕЛЕНИЕ ПРИЗНАКОВ И МЕТОК
print("
2️⃣ ВЫДЕЛЕНИЕ ПРИЗНАКОВ И МЕТОК")
print("-"*50)

# Извлекаем метки
y_raw = df_clean[label_col].values

# Кодируем метки (текст -> числа)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)

print(f"   • Метки закодированы (текст -> числа)")
print(f"   • Уникальных классов: {len(label_encoder.classes_)}")

print(f"
   Соответствие классов и кодов:")
for i, class_name in enumerate(label_encoder.classes_):
    count = np.sum(y_encoded == i)
    percentage = count / len(y_encoded) * 100
    print(f"   • {i:2} -> {str(class_name)[:30]:30} : {count:8,} ({percentage:.2f}%)")

# Выбираем числовые признаки (исключая колонку с метками)
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
if label_col in numeric_cols:
    numeric_cols.remove(label_col)

X = df_clean[numeric_cols].copy()
print(f"
   • Матрица признаков X: {X.shape}")
print(f"   • Вектор меток y: {y_encoded.shape}")

# 3. ОБРАБОТКА ПРОПУЩЕННЫХ ЗНАЧЕНИЙ
print("
3️⃣ ОБРАБОТКА ПРОПУЩЕННЫХ ЗНАЧЕНИЙ")
print("-"*50)

missing_before = X.isnull().sum().sum()
print(f"   • Пропусков до обработки: {missing_before}")

if missing_before > 0:
    # Заполняем пропуски медианой
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)

    missing_after = X.isnull().sum().sum()
    print(f"   • Пропусков после обработки: {missing_after}")
else:
    print("   • Пропуски отсутствуют")

# 4. ОБРАБОТКА БЕСКОНЕЧНЫХ ЗНАЧЕНИЙ
print("
4️⃣ ОБРАБОТКА БЕСКОНЕЧНЫХ ЗНАЧЕНИЙ")
print("-"*50)

inf_before = np.isinf(X.values).sum()
print(f"   • Бесконечных значений до обработки: {inf_before}")

if inf_before > 0:
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            X[col].fillna(X[col].median(), inplace=True)

    inf_after = np.isinf(X.values).sum()
    print(f"   • Бесконечных значений после обработки: {inf_after}")
else:
    print("   • Бесконечные значения отсутствуют")

# 5. МАСШТАБИРОВАНИЕ ПРИЗНАКОВ
print("
5️⃣ МАСШТАБИРОВАНИЕ ПРИЗНАКОВ")
print("-"*50)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

print(f"   • Применено масштабирование (StandardScaler)")
print(f"   • Среднее после масштабирования: {X.mean().mean():.10f}")

# 6. ПРОВЕРКА КАЧЕСТВА ДАННЫХ
print("
6️⃣ ПРОВЕРКА КАЧЕСТВА ДАННЫХ")
print("-"*50)

print(f"   • Пропуски в X: {np.isnan(X.values).sum()}")
print(f"   • Бесконечности в X: {np.isinf(X.values).sum()}")
print(f"   • Размер X: {X.shape}")
print(f"   • Размер y: {y_encoded.shape}")

print("
" + "="*70)
print("✅ ПРЕДОБРАБОТКА ЗАВЕРШЕНА")
print("="*70)

# Сохраняем предобработанные данные
preprocessed_data = {
    'X': X,
    'y': y_encoded,
    'label_encoder': label_encoder,
    'scaler': scaler,
    'feature_names': X.columns.tolist(),
    'class_names': label_encoder.classes_.tolist()
}

joblib.dump(preprocessed_data, '/content/preprocessed_cicids2017.pkl')
print(f"
💾 Предобработанные данные сохранены в preprocessed_cicids2017.pkl")

# @title 8. БАЛАНСИРОВКА КЛАССОВ { display-mode: "form" }

print("="*70)
print("⚖️ БАЛАНСИРОВКА КЛАССОВ")
print("="*70)

from imblearn.over_sampling import SMOTE
from collections import Counter
import joblib

# Загружаем предобработанные данные
if 'X' not in locals() or 'y' not in locals():
    print("
📂 Загрузка предобработанных данных...")
    preprocessed = joblib.load('/content/preprocessed_cicids2017.pkl')
    X = preprocessed['X']
    y = preprocessed['y']
    label_encoder = preprocessed['label_encoder']

# 1. АНАЛИЗ ТЕКУЩЕГО РАСПРЕДЕЛЕНИЯ
print("
1️⃣ ТЕКУЩЕЕ РАСПРЕДЕЛЕНИЕ КЛАССОВ")
print("-"*50)

class_counts = Counter(y)
total = len(y)

print(f"
   {'Код':4} | {'Класс':30} | {'Количество':12} | {'Процент':8}")
print(f"   {'-'*65}")

for class_val, count in sorted(class_counts.items()):
    class_name = label_encoder.inverse_transform([class_val])[0]
    percentage = count / total * 100
    print(f"   {class_val:3} | {str(class_name)[:30]:30} | {count:12,} | {percentage:6.2f}%")

# 2. РАСЧЕТ КОЭФФИЦИЕНТА ДИСБАЛАНСА
print("
2️⃣ АНАЛИЗ ДИСБАЛАНСА")
print("-"*50)

if len(class_counts) > 1:
    majority_class = max(class_counts, key=class_counts.get)
    majority_count = class_counts[majority_class]
    minority_class = min(class_counts, key=class_counts.get)
    minority_count = class_counts[minority_class]

    imbalance_ratio = majority_count / minority_count
    print(f"   • Мажоритарный класс: {label_encoder.inverse_transform([majority_class])[0]} ({majority_count:,})")
    print(f"   • Миноритарный класс: {label_encoder.inverse_transform([minority_class])[0]} ({minority_count:,})")
    print(f"   • Коэффициент дисбаланса: {imbalance_ratio:.2f}:1")

    if imbalance_ratio > 100:
        print(f"
   ⚠️ КРИТИЧЕСКИЙ ДИСБАЛАНС - требуется балансировка!")

# 3. ПРИМЕНЕНИЕ SMOTE
print("
3️⃣ ПРИМЕНЕНИЕ SMOTE")
print("-"*50)

try:
    # Настраиваем SMOTE
    smote = SMOTE(
        sampling_strategy='auto',
        random_state=42,
        k_neighbors=3
    )

    print("   🚀 Выполняется балансировка...")
    print(f"   Это может занять несколько минут...")

    X_balanced, y_balanced = smote.fit_resample(X, y)

    print(f"
   ✅ Балансировка выполнена успешно!")
    print(f"   • Размер ДО: {len(y):,}")
    print(f"   • Размер ПОСЛЕ: {len(y_balanced):,}")
    print(f"   • Увеличено на: {len(y_balanced) - len(y):,} строк")

except Exception as e:
    print(f"
❌ Ошибка при SMOTE: {e}")
    print("
🔄 Используем оригинальные данные без балансировки")
    X_balanced, y_balanced = X, y

# 4. АНАЛИЗ НОВОГО РАСПРЕДЕЛЕНИЯ
print("
4️⃣ НОВОЕ РАСПРЕДЕЛЕНИЕ КЛАССОВ")
print("-"*50)

balanced_counts = Counter(y_balanced)
print(f"
   {'Код':4} | {'Класс':30} | {'Количество':12} | {'Процент':8}")
print(f"   {'-'*65}")

for class_val, count in sorted(balanced_counts.items()):
    class_name = label_encoder.inverse_transform([class_val])[0]
    percentage = count / len(y_balanced) * 100
    print(f"   {class_val:3} | {str(class_name)[:30]:30} | {count:12,} | {percentage:6.2f}%")

# 5. СРАВНЕНИЕ ДО/ПОСЛЕ
print("
5️⃣ СРАВНЕНИЕ ДО И ПОСЛЕ")
print("-"*50)

print(f"
   {'Класс':30} | {'ДО':10} | {'ПОСЛЕ':10} | {'Изменение':10}")
print(f"   {'-'*70}")

for class_val in sorted(class_counts.keys()):
    class_name = label_encoder.inverse_transform([class_val])[0]
    before = class_counts[class_val]
    after = balanced_counts[class_val]
    change = after - before
    change_pct = (change / before * 100) if before > 0 else 0
    print(f"   {str(class_name)[:30]:30} | {before:8,} | {after:8,} | +{change:6,} ({change_pct:+.0f}%)")

print("
" + "="*70)
print("✅ БАЛАНСИРОВКА ЗАВЕРШЕНА")
print("="*70)

# Сохраняем сбалансированные данные
balanced_data = {
    'X': X_balanced,
    'y': y_balanced,
    'label_encoder': label_encoder,
    'feature_names': X.columns.tolist(),
    'original_distribution': dict(class_counts),
    'balanced_distribution': dict(balanced_counts)
}

joblib.dump(balanced_data, '/content/balanced_cicids2017.pkl')
print(f"
💾 Сбалансированные данные сохранены в balanced_cicids2017.pkl")

# @title 9. РАЗДЕЛЕНИЕ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ { display-mode: "form" }

print("="*70)
print("✂️ РАЗДЕЛЕНИЕ ДАННЫХ")
print("="*70)

from sklearn.model_selection import train_test_split
import joblib

# Загружаем сбалансированные данные
if 'X_balanced' not in locals() or 'y_balanced' not in locals():
    print("
📂 Загрузка сбалансированных данных...")
    balanced = joblib.load('/content/balanced_cicids2017.pkl')
    X_balanced = balanced['X']
    y_balanced = balanced['y']
    label_encoder = balanced['label_encoder']

print(f"
📊 Данные для разделения:")
print(f"   • Признаки (X): {X_balanced.shape}")
print(f"   • Метки (y): {y_balanced.shape}")

# 1. РАЗДЕЛЕНИЕ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ
print("
1️⃣ РАЗДЕЛЕНИЕ (80% / 20%)")
print("-"*50)

X_train, X_test, y_train, y_test = train_test_split(
    X_balanced,
    y_balanced,
    test_size=0.2,
    random_state=42,
    stratify=y_balanced
)

print(f"
   • Обучающая выборка: {X_train.shape[0]:,} строк")
print(f"   • Тестовая выборка: {X_test.shape[0]:,} строк")

# 2. ПРОВЕРКА РАСПРЕДЕЛЕНИЯ
print("
2️⃣ РАСПРЕДЕЛЕНИЕ В ВЫБОРКАХ")
print("-"*50)

train_counts = Counter(y_train)
test_counts = Counter(y_test)

print(f"
   {'Класс':30} | {'Обучающая':12} | {'Тестовая':12} | {'Всего':12}")
print(f"   {'-'*70}")

for class_val in sorted(train_counts.keys()):
    class_name = label_encoder.inverse_transform([class_val])[0]
    train_count = train_counts[class_val]
    test_count = test_counts[class_val]
    total_count = train_count + test_count
    print(f"   {str(class_name)[:30]:30} | {train_count:8,} | {test_count:8,} | {total_count:8,}")

print("
" + "="*70)
print("✅ РАЗДЕЛЕНИЕ ЗАВЕРШЕНО")
print("="*70)

# @title 9. БАЛАНСИРОВКА КЛАССОВ (SMOTE) { display-mode: "form" }

print("⚖️ Балансировка классов...")

print(f"  До SMOTE: {np.bincount(y_train)}")

try:
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    print(f"  После SMOTE: {np.bincount(y_train_balanced)}")
    use_smote = True
except Exception as e:
    print(f"  SMOTE не применен: {e}")
    X_train_balanced, y_train_balanced = X_train_scaled, y_train
    use_smote = False

    # @title 10. СОХРАНЕНИЕ ИТОГОВЫХ ДАННЫХ { display-mode: "form" }

print("="*70)
print("💾 СОХРАНЕНИЕ ИТОГОВЫХ ДАННЫХ")
print("="*70)

import joblib
import os

# 1. ПОДГОТОВКА ИТОГОВОГО СЛОВАРЯ
print("
1️⃣ ПОДГОТОВКА ДАННЫХ ДЛЯ СОХРАНЕНИЯ")
print("-"*50)

final_data = {
    # Данные для обучения
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test,

    # Метаданные
    'feature_names': X_balanced.columns.tolist(),
    'label_encoder': label_encoder,
    'class_names': label_encoder.classes_.tolist(),

    # Информация о распределении
    'original_distribution': dict(Counter(y)),
    'balanced_distribution': dict(Counter(y_balanced)),
    'train_distribution': dict(Counter(y_train)),
    'test_distribution': dict(Counter(y_test)),

    # Параметры
    'n_features': X_train.shape[1],
    'n_classes': len(np.unique(y_balanced)),
    'n_train_samples': len(y_train),
    'n_test_samples': len(y_test)
}

print(f"   • Обучающих данных: {final_data['n_train_samples']:,}")
print(f"   • Тестовых данных: {final_data['n_test_samples']:,}")
print(f"   • Признаков: {final_data['n_features']}")
print(f"   • Классов: {final_data['n_classes']}")

# 2. СОХРАНЕНИЕ В JOBLIB
print("
2️⃣ СОХРАНЕНИЕ В JOBLIB ФОРМАТ")
print("-"*50)

joblib_path = '/content/cicids2017_final_prepared.pkl'
joblib.dump(final_data, joblib_path)

file_size = os.path.getsize(joblib_path) / (1024**2)
print(f"   • Файл: {joblib_path}")
print(f"   • Размер: {file_size:.2f} MB")
print(f"   ✅ Сохранено успешно!")

# 3. СОХРАНЕНИЕ В CSV
print("
3️⃣ СОХРАНЕНИЕ В CSV ФОРМАТ")
print("-"*50)

# Обучающая выборка
train_df = pd.DataFrame(X_train, columns=final_data['feature_names'])
train_df['Label'] = y_train
train_df['Label_Name'] = label_encoder.inverse_transform(y_train)
train_df.to_csv('/content/cicids2017_train.csv', index=False)
print(f"   • Обучающая выборка: cicids2017_train.csv")

# Тестовая выборка
test_df = pd.DataFrame(X_test, columns=final_data['feature_names'])
test_df['Label'] = y_test
test_df['Label_Name'] = label_encoder.inverse_transform(y_test)
test_df.to_csv('/content/cicids2017_test.csv', index=False)
print(f"   • Тестовая выборка: cicids2017_test.csv")

# 4. ИТОГОВЫЙ ОТЧЕТ
print("
4️⃣ ИТОГОВЫЙ ОТЧЕТ")
print("-"*50)

print(f"""
╔{'═'*70}╗
║ {'ИСХОДНЫЕ ДАННЫЕ':<68} ║
╠{'═'*70}╣
║   • Размер датасета: {df.shape[0]:,} x {df.shape[1]}
║   • Колонка с метками: '{label_col}'
║   • Количество классов: {final_data['n_classes']}
╠{'═'*70}╣
║ {'ПРЕДОБРАБОТКА':<68} ║
╠{'═'*70}╣
║   • Удалено дубликатов: {duplicates_removed:,}
║   • Пропуски: {'были обработаны' if missing_before > 0 else 'отсутствовали'}
║   • Масштабирование: StandardScaler
╠{'═'*70}╣
║ {'БАЛАНСИРОВКА':<68} ║
╠{'═'*70}╣
║   • Коэффициент дисбаланса: {imbalance_ratio:.2f}:1
║   • Метод: SMOTE
║   • Размер ДО: {len(y):,}
║   • Размер ПОСЛЕ: {len(y_balanced):,}
╠{'═'*70}╣
║ {'ИТОГОВЫЕ ДАННЫЕ':<68} ║
╠{'═'*70}╣
║   • Обучающая выборка: {len(y_train):,} строк
║   • Тестовая выборка: {len(y_test):,} строк
║   • Формат: joblib + CSV
╚{'═'*70}╝
""")

print("
" + "="*70)
print("🎯 ВСЕ ЭТАПЫ УСПЕШНО ЗАВЕРШЕНЫ!")
print("="*70)
print("
✅ ДАННЫЕ ГОТОВЫ ДЛЯ МАШИННОГО ОБУЧЕНИЯ!")

# @title 11. ОБУЧЕНИЕ МОДЕЛЕЙ МАШИННОГО ОБУЧЕНИЯ { display-mode: "form" }

print("="*70)
print("🤖 ОБУЧЕНИЕ МОДЕЛЕЙ МАШИННОГО ОБУЧЕНИЯ")
print("="*70)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import roc_curve, auc
import joblib
import warnings
warnings.filterwarnings('ignore')

# Загружаем подготовленные данные
print("
📂 Загрузка подготовленных данных...")
if os.path.exists('/content/cicids2017_final_prepared.pkl'):
    final_data = joblib.load('/content/cicids2017_final_prepared.pkl')
    X_train = final_data['X_train']
    X_test = final_data['X_test']
    y_train = final_data['y_train']
    y_test = final_data['y_test']
    label_encoder = final_data['label_encoder']
    class_names = final_data['class_names']
    print(f"✅ Данные загружены успешно!")
else:
    # Если файл не найден, используем переменные из памяти
    print("⚠️ Файл не найден, используем переменные из памяти")

print(f"
📊 Размеры выборок:")
print(f"   • Обучающая выборка: {X_train.shape}")
print(f"   • Тестовая выборка: {X_test.shape}")

# ============================================================
# 1. ОБУЧЕНИЕ LOGISTIC REGRESSION
# ============================================================
print("
" + "="*70)
print("1️⃣ LOGISTIC REGRESSION")
print("="*70)

from sklearn.linear_model import LogisticRegression

# Создаем и обучаем модель
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

# Предсказания
y_pred_lr = lr_model.predict(X_test)
y_prob_lr = lr_model.predict_proba(X_test)[:, 1]

# Метрики
accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr, average='weighted')
recall_lr = recall_score(y_test, y_pred_lr, average='weighted')
f1_lr = f1_score(y_test, y_pred_lr, average='weighted')

print(f"
📊 Метрики:")
print(f"   • Accuracy:  {accuracy_lr:.4f}")
print(f"   • Precision: {precision_lr:.4f}")
print(f"   • Recall:    {recall_lr:.4f}")
print(f"   • F1-score:  {f1_lr:.4f}")

# ============================================================
# 2. ОБУЧЕНИЕ RANDOM FOREST
# ============================================================
print("
" + "="*70)
print("2️⃣ RANDOM FOREST")
print("="*70)

from sklearn.ensemble import RandomForestClassifier

# Создаем и обучаем модель
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Предсказания
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# Метрики
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, average='weighted')
recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')

print(f"
📊 Метрики:")
print(f"   • Accuracy:  {accuracy_rf:.4f}")
print(f"   • Precision: {precision_rf:.4f}")
print(f"   • Recall:    {recall_rf:.4f}")
print(f"   • F1-score:  {f1_rf:.4f}")

# Важность признаков
feature_importance = pd.DataFrame({
    'feature': final_data['feature_names'],
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"
🔍 Топ-10 важных признаков:")
print(feature_importance.head(10).to_string(index=False))

# ============================================================
# 3. ОБУЧЕНИЕ XGBOOST
# ============================================================
print("
" + "="*70)
print("3️⃣ XGBOOST")
print("="*70)

try:
    import xgboost as xgb

    # Создаем и обучаем модель
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)

    # Предсказания
    y_pred_xgb = xgb_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

    # Метрики
    accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
    precision_xgb = precision_score(y_test, y_pred_xgb, average='weighted')
    recall_xgb = recall_score(y_test, y_pred_xgb, average='weighted')
    f1_xgb = f1_score(y_test, y_pred_xgb, average='weighted')

    print(f"
📊 Метрики:")
    print(f"   • Accuracy:  {accuracy_xgb:.4f}")
    print(f"   • Precision: {precision_xgb:.4f}")
    print(f"   • Recall:    {recall_xgb:.4f}")
    print(f"   • F1-score:  {f1_xgb:.4f}")

except:
    print("❌ XGBoost не установлен, пропускаем...")
    accuracy_xgb = precision_xgb = recall_xgb = f1_xgb = 0

# ============================================================
# 4. ОБУЧЕНИЕ LIGHTGBM
# ============================================================
print("
" + "="*70)
print("4️⃣ LIGHTGBM")
print("="*70)

try:
    import lightgbm as lgb

    # Создаем и обучаем модель
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)

    # Предсказания
    y_pred_lgb = lgb_model.predict(X_test)
    y_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]

    # Метрики
    accuracy_lgb = accuracy_score(y_test, y_pred_lgb)
    precision_lgb = precision_score(y_test, y_pred_lgb, average='weighted')
    recall_lgb = recall_score(y_test, y_pred_lgb, average='weighted')
    f1_lgb = f1_score(y_test, y_pred_lgb, average='weighted')

    print(f"
📊 Метрики:")
    print(f"   • Accuracy:  {accuracy_lgb:.4f}")
    print(f"   • Precision: {precision_lgb:.4f}")
    print(f"   • Recall:    {recall_lgb:.4f}")
    print(f"   • F1-score:  {f1_lgb:.4f}")

except:
    print("❌ LightGBM не установлен, пропускаем...")
    accuracy_lgb = precision_lgb = recall_lgb = f1_lgb = 0

# ============================================================
# 5. ОБУЧЕНИЕ SVM
# ============================================================
print("
" + "="*70)
print("5️⃣ SVM (Support Vector Machine)")
print("="*70)

from sklearn.svm import SVC

# Создаем и обучаем модель (уменьшаем выборку для SVM)
svm_sample_size = min(10000, len(X_train))
X_train_svm = X_train[:svm_sample_size]
y_train_svm = y_train[:svm_sample_size]

print(f"   • SVM обучается на {svm_sample_size} образцах (для скорости)")

svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train_svm, y_train_svm)

# Предсказания
y_pred_svm = svm_model.predict(X_test)
y_prob_svm = svm_model.predict_proba(X_test)[:, 1]

# Метрики
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm, average='weighted')
recall_svm = recall_score(y_test, y_pred_svm, average='weighted')
f1_svm = f1_score(y_test, y_pred_svm, average='weighted')

print(f"
📊 Метрики:")
print(f"   • Accuracy:  {accuracy_svm:.4f}")
print(f"   • Precision: {precision_svm:.4f}")
print(f"   • Recall:    {recall_svm:.4f}")
print(f"   • F1-score:  {f1_svm:.4f}")

# ============================================================
# СРАВНЕНИЕ МОДЕЛЕЙ
# ============================================================
print("
" + "="*70)
print("📊 СРАВНЕНИЕ МОДЕЛЕЙ")
print("="*70)

# Создаем таблицу результатов
results = pd.DataFrame({
    'Модель': ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM', 'SVM'],
    'Accuracy': [accuracy_lr, accuracy_rf, accuracy_xgb, accuracy_lgb, accuracy_svm],
    'Precision': [precision_lr, precision_rf, precision_xgb, precision_lgb, precision_svm],
    'Recall': [recall_lr, recall_rf, recall_xgb, recall_lgb, recall_svm],
    'F1-score': [f1_lr, f1_rf, f1_xgb, f1_lgb, f1_svm]
})

print("
📋 Таблица сравнения:")
print(results.to_string(index=False))

# Находим лучшую модель
best_f1 = results.loc[results['F1-score'].idxmax()]
print(f"
🏆 Лучшая модель: {best_f1['Модель']} (F1-score: {best_f1['F1-score']:.4f})")

# ============================================================
# СОХРАНЕНИЕ МОДЕЛЕЙ (ИСПРАВЛЕННАЯ ВЕРСИЯ)
# ============================================================
print("
" + "="*70)
print("💾 СОХРАНЕНИЕ МОДЕЛЕЙ")
print("="*70)

# Собираем все существующие модели
models = {}

# Добавляем все модели, которые есть в памяти
if 'lr_model' in locals():
    models['logistic_regression'] = lr_model
    print(f"   ✅ Добавлена: logistic_regression")

if 'rf_model' in locals():
    models['random_forest'] = rf_model
    print(f"   ✅ Добавлена: random_forest")

if 'svm_model' in locals():
    models['svm'] = svm_model
    print(f"   ✅ Добавлена: svm")

if 'xgb_model' in locals() and 'xgb_installed' in locals() and xgb_installed:
    models['xgboost'] = xgb_model
    print(f"   ✅ Добавлена: xgboost")

if 'lgb_model' in locals() and 'lgb_installed' in locals() and lgb_installed:
    models['lightgbm'] = lgb_model
    print(f"   ✅ Добавлена: lightgbm")

# Сохраняем все модели
if models:
    joblib.dump(models, '/content/trained_models.pkl')
    print(f"
✅ Все модели сохранены в trained_models.pkl")
    print(f"   • Сохранено моделей: {len(models)}")
    print(f"   • Список моделей: {', '.join(models.keys())}")

    # Проверяем размер файла
    import os
    file_size = os.path.getsize('/content/trained_models.pkl') / (1024**2)
    print(f"   • Размер файла: {file_size:.2f} MB")
else:
    print("❌ Нет моделей для сохранения!")

# Сохраняем результаты
if 'results' in locals() and isinstance(results, pd.DataFrame):
    results.to_csv('/content/model_results.csv', index=False)
    print(f"
✅ Результаты сохранены в model_results.csv")

    # Показываем результаты
    print(f"
📊 Итоговая таблица результатов:")
    print(results.to_string(index=False))

    # Находим лучшую модель
    best_model = results.loc[results['F1-score'].idxmax(), 'Модель']
    best_f1 = results['F1-score'].max()
    print(f"
🏆 Лучшая модель: {best_model} (F1-score: {best_f1:.4f})")

else:
    print("
❌ Нет результатов для сохранения!")

# ============================================================
# ДЕТАЛЬНЫЙ ОТЧЕТ ДЛЯ ВСЕХ МОДЕЛЕЙ
# ============================================================
print("
" + "="*70)
print("📋 ДЕТАЛЬНЫЕ ОТЧЕТЫ ПО МОДЕЛЯМ")
print("="*70)

# Функция для получения предсказаний модели
def get_predictions(model_name):
    if model_name == 'Logistic Regression' and 'y_pred_lr' in locals():
        return y_pred_lr
    elif model_name == 'Random Forest' and 'y_pred_rf' in locals():
        return y_pred_rf
    elif model_name == 'XGBoost' and 'y_pred_xgb' in locals():
        return y_pred_xgb
    elif model_name == 'LightGBM' and 'y_pred_lgb' in locals():
        return y_pred_lgb
    elif model_name == 'SVM' and 'y_pred_svm' in locals():
        return y_pred_svm
    return None

# Преобразуем class_names в строки
if 'class_names' in locals():
    if isinstance(class_names[0], (int, np.integer)):
        target_names = [str(name) for name in class_names]
    else:
        target_names = class_names
else:
    target_names = ['0', '1']

# Создаем отчет для каждой модели
reports = {}

for model_name in results['Модель']:
    y_pred = get_predictions(model_name)
    if y_pred is not None:
        report = classification_report(y_test, y_pred,
                                     target_names=target_names,
                                     output_dict=True)
        reports[model_name] = report

        print(f"
📊 {model_name}:")
        print("-" * 50)
        report_df = pd.DataFrame(report).transpose()
        print(report_df.round(4).to_string())

# Сохраняем все отчеты
import json
with open('/content/all_reports.json', 'w') as f:
    # Конвертируем numpy типы в Python типы для JSON
    reports_json = {}
    for model, report in reports.items():
        reports_json[model] = {}
        for key, value in report.items():
            if isinstance(value, dict):
                reports_json[model][key] = {k: float(v) if hasattr(v, 'item') else v
                                           for k, v in value.items()}
            else:
                reports_json[model][key] = float(value) if hasattr(value, 'item') else value

    json.dump(reports_json, f, indent=2)

print(f"
💾 Все отчеты сохранены в all_reports.json")

# Сохраняем лучшую модель отдельно
if best_model:
    best_model_key = {
        'Logistic Regression': 'logistic_regression',
        'Random Forest': 'random_forest',
        'SVM': 'svm',
        'XGBoost': 'xgboost',
        'LightGBM': 'lightgbm'
    }.get(best_model)

    if best_model_key and best_model_key in models:
        joblib.dump(models[best_model_key], f'/content/best_model_{best_model_key}.pkl')
        print(f"
🏆 Лучшая модель ({best_model}) сохранена отдельно в best_model_{best_model_key}.pkl")

print("
" + "="*70)
print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
print("="*70)

print("
📁 Сохраненные файлы:")
print("   • trained_models.pkl - все обученные модели (5 моделей)")
print("   • model_results.csv - таблица результатов")
print("   • all_reports.json - детальные отчеты по всем моделям")
print(f"   • best_model_{best_model_key}.pkl - лучшая модель ({best_model})")
