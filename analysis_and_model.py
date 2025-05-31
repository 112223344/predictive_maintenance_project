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


import matplotlib
matplotlib.use("Agg")
# Создаём папку для графиков
os.makedirs("plots", exist_ok=True)
import joblib   

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
    
     # === 4.1) Обучение Random Forest ===
    st.header("4. Обучение и оценка Random Forest")
    from sklearn.ensemble import RandomForestClassifier

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Предсказания и оценки для Random Forest
    y_pred_rf = rf_model.predict(X_test)
    y_proba_rf = rf_model.predict_proba(X_test)[:, 1]
    report_rf = classification_report(y_test, y_pred_rf, output_dict=True)
    auc_rf = roc_auc_score(y_test, y_proba_rf)

    st.subheader("Random Forest: Classification Report")
    st.dataframe(pd.DataFrame(report_rf).T)
    st.write(f"Random Forest ROC AUC: **{auc_rf:.4f}**")

    # ROC-кривая для Random Forest
    fig_rf, ax_rf = plt.subplots()
    RocCurveDisplay.from_predictions(y_test, y_proba_rf, ax=ax_rf, name="RandomForest")
    st.pyplot(fig_rf)
    plt.close(fig_rf)

    # === 4.2) Сохранение модели и энкодеров ===
    # Сохраняем RandomForest
    joblib.dump(rf_model, "best_model.joblib")
    # ohe и scaler были определены ранее в функции
    joblib.dump(ohe, "ohe_encoder.joblib")
    joblib.dump(scaler, "scaler.joblib")
    st.success("✅ Модель и энкодеры сохранены (best_model.joblib, ohe_encoder.joblib, scaler.joblib)")
    
def prediction_page():
    st.title("Прогноз отказа оборудования (онлайн)")

    # 1) Загружаем сохранённые модель и энкодеры
    try:
        model = joblib.load("best_model.joblib")
        ohe   = joblib.load("ohe_encoder.joblib")
        scaler = joblib.load("scaler.joblib")
    except FileNotFoundError:
        st.error("Модель и/или энкодеры не найдены. Сначала перейдите в 'Анализ и модель' для обучения.")
        return

    # 2) Форма для ввода новых данных
    with st.form(key="predict_form"):
        st.subheader("Введите значения параметров оборудования:")
        type_input = st.selectbox("Type", options=["L", "M", "H"])
        air_temp   = st.number_input("Air temperature (K)", min_value=200.0, max_value=400.0, value=300.0, step=0.1)
        proc_temp  = st.number_input("Process temperature (K)", min_value=200.0, max_value=400.0, value=300.0, step=0.1)
        rot_spd    = st.number_input("Rotational speed (rpm)", min_value=0, max_value=5000, value=1500, step=1)
        torque     = st.number_input("Torque (Nm)", min_value=0.0, max_value=100.0, value=40.0, step=0.1)
        tool_wear  = st.number_input("Tool wear (minutes)", min_value=0, max_value=100, value=0, step=1)

        submit_button = st.form_submit_button(label="Сделать прогноз")

    if submit_button:
        # Составляем DataFrame из введённых значений
        df_in = pd.DataFrame({
            "Air temperature":      [air_temp],
            "Process temperature":  [proc_temp],
            "Rotational speed":     [rot_spd],
            "Torque":               [torque],
            "Tool wear":            [tool_wear],
            "Type":                 [type_input]
        })

        # One-Hot Encoding категориального признака
        X_cat_in = pd.DataFrame(
            ohe.transform(df_in[["Type"]]),
            columns=ohe.get_feature_names_out(["Type"])
        )

        # Масштабирование числовых признаков
        X_num_in = pd.DataFrame(
            scaler.transform(df_in[["Air temperature", "Process temperature", "Rotational speed", "Torque", "Tool wear"]]),
            columns=["Air temperature", "Process temperature", "Rotational speed", "Torque", "Tool wear"]
        )

        # Итоговый набор признаков
        X_in = pd.concat([X_num_in, X_cat_in], axis=1)

        # Предсказание
        proba = model.predict_proba(X_in)[0, 1]
        pred  = model.predict(X_in)[0]

        st.write(f"**Вероятность отказа:** {proba:.2f}")
        if pred == 1:
            st.error("⚠️ Модель предсказывает: возможен отказ оборудования!")
        else:
            st.success("✅ Модель предсказывает: отказа не будет.")

def load_data(use_local=False):

    if use_local:
        df = pd.read_csv("data/predictive_maintenance.csv")
    else:
        repo = fetch_ucirepo(id=601)
        # Берём полный исходный фрейм из ключа 'original'
        df = repo['data']['original']
    return df

def preprocess(df):

    print("Shape:", df.shape)
    print(df.dtypes)
    print("Nulls:\n", df.isnull().sum())
    return df
    
def plot_histograms(df, cols, bins=30):
    for col in cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], bins=bins, kde=True)
        plt.title(f"Распределение {col}")
        plt.xlabel(col)
        plt.ylabel("Частота")
        plt.tight_layout()
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

    #Отделяем таргет
    y = df["Machine failure"]
    # Выбираем признаки
    X_num = df[["Air temperature", "Process temperature",
                 "Rotational speed", "Torque", "Tool wear"]]
    X_cat = df[["Type"]]

    # One кодирование
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

    # Корреляци
    plot_correlation_matrix(df, numeric_cols)

    # Баланс целевой переменной
    plot_class_balance(df, "Machine failure")
    X, y = feature_engineering(df)
    # Разбиваем на тренировочную и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)