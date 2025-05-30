import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay

# Устанавливаем бэкенд для сохранения графиков
import matplotlib
matplotlib.use("Agg")
# Создаём папку для графиков
os.makedirs("plots", exist_ok=True)

def analysis_and_model_page():
    st.title("1. Анализ данных и обучение модели")

    # 1) Загрузка данных
    df = fetch_ucirepo(id=601).data.original
    st.write("**Первый взгляд на данные:**")
    st.dataframe(df.head())

    # 2) Предобработка и EDA
    if st.expander("Предобработка и EDA"):
        st.write("— Удаление идентификаторов и лишних столбцов")
        df = df.drop(columns=["UID", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"])
        st.write("— Типы столбцов и пропуски")
        st.write(df.dtypes.to_frame("dtype"))
        st.write(df.isnull().sum().to_frame("nulls"))

        st.write("— Гистограммы числовых признаков")
        numeric_cols = df.select_dtypes(["int64","float64"]).drop("Machine failure", axis=1).columns
        for col in numeric_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            st.pyplot(fig)
            plt.close(fig)

        st.write("— Корреляции")
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        plt.close(fig)

        st.write("— Баланс классов")
        counts = df["Machine failure"].value_counts().sort_index()
        fig, ax = plt.subplots()
        sns.barplot(x=counts.index, y=counts.values, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    # 3) Feature engineering & split
    st.header("2. Feature Engineering и Train/Test Split")
    X_num = df[["Air temperature","Process temperature","Rotational speed","Torque","Tool wear"]]
    X_cat = df[["Type"]]
    ohe = OneHotEncoder(sparse_output=False, drop="if_binary")
    X_cat_enc = pd.DataFrame(ohe.fit_transform(X_cat),
                              columns=ohe.get_feature_names_out(["Type"]),
                              index=df.index)
    scaler = StandardScaler()
    X_num_scaled = pd.DataFrame(scaler.fit_transform(X_num),
                                columns=X_num.columns,index=df.index)
    X = pd.concat([X_num_scaled, X_cat_enc], axis=1)
    y = df["Machine failure"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    st.write(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # 4) Обучение и оценка логрегрессии
    st.header("3. Обучение и оценка Logistic Regression")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    report = classification_report(y_test, y_pred, output_dict=True)
    st.subheader("Classification Report")
    st.dataframe(pd.DataFrame(report).T)
    auc = roc_auc_score(y_test, y_proba)
    st.write(f"ROC AUC: **{auc:.4f}**")
    # ROC-кривая
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

def load_data(use_local=False):
    """
    Загружает датасет:
      - локально (если use_local=True) из data/predictive_maintenance.csv,
      - из UCIMLRepo (repo['data']['original']) иначе.
    """
    if use_local:
        df = pd.read_csv("data/predictive_maintenance.csv")
    else:
        repo = fetch_ucirepo(id=601)
        # Берём полный исходный фрейм из ключа 'original'
        df = repo['data']['original']
    return df

def preprocess(df):
    """
    Первичная предобработка: 
    - проверка пропущенных значений
    - приведение типов
    - базовая статистика
    """
    print("Shape:", df.shape)
    print(df.dtypes)
    print("Nulls:\n", df.isnull().sum())
    # Здесь можно добавить fillna/удаление строк/кодирование
    return df
    
def plot_histograms(df, cols, bins=30):
    for col in cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], bins=bins, kde=True)
        plt.title(f"Распределение {col}")
        plt.xlabel(col)
        plt.ylabel("Частота")
        plt.tight_layout()
        # Сохраняем и закрываем
        plt.savefig(f"plots/{col}_hist.png")
        plt.close()

def plot_correlation_matrix(df, cols):
    corr = df[cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Матрица корреляций")
    plt.tight_layout()
    plt.savefig("plots/correlation_matrix.png")
    plt.close()

def plot_class_balance(df, target_col):
    counts = df[target_col].value_counts().sort_index()
    plt.figure(figsize=(4, 4))
    sns.barplot(x=counts.index, y=counts.values)
    plt.title(f"Баланс классов: {target_col}")
    plt.xlabel(target_col)
    plt.ylabel("Количество")
    plt.tight_layout()
    plt.savefig(f"plots/{target_col}_balance.png")
    plt.close()

def feature_engineering(df):
    """
    - Кодируем категориальный признак 'Type' через OneHot.
    - Масштабируем числовые признаки.
    - Возвращаем X (матрицу признаков) и y (целевую).
    """
    # Отделяем таргет
    y = df["Machine failure"]
    # Выбираем признаки
    X_num = df[["Air temperature", "Process temperature",
                 "Rotational speed", "Torque", "Tool wear"]]
    X_cat = df[["Type"]]

    # One-hot кодирование
    ohe = OneHotEncoder(sparse_output=False, drop="if_binary")
    X_cat_enc = pd.DataFrame(
        ohe.fit_transform(X_cat),
        columns=ohe.get_feature_names_out(["Type"]),
        index=df.index
    )

    # Масштабирование чисел
    scaler = StandardScaler()
    X_num_scaled = pd.DataFrame(
        scaler.fit_transform(X_num),
        columns=X_num.columns,
        index=df.index
    )

    # Составляем финальный датафрейм
    X = pd.concat([X_num_scaled, X_cat_enc], axis=1)
    return X, y
    
def main():
    df = load_data(use_local=False)
    df = preprocess(df)

    # Выделяем числовые признаки (кроме UID)
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.drop(["UID", "Machine failure"])
    plot_histograms(df, numeric_cols)

    # Корреляции
    plot_correlation_matrix(df, numeric_cols)

    # Баланс целевой переменной
    plot_class_balance(df, "Machine failure")
    X, y = feature_engineering(df)
    # Разбиваем на тренировочную и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)