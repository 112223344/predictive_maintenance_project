import streamlit as st
import reveal_slides as rs

def presentation_page():
    st.title("Презентация проекта")
    md = """
# Предиктивное обслуживание оборудования
---
## 1. Введение
- Цель: бинарная классификация отказов  
- Датасет: AI4I 2020, 10 000 записей, 14 признаков  
---
## 2. Этапы работы
1. Загрузка и предобработка данных  
2. EDA и анализ распределений  
3. Feature engineering и split  
4. Обучение и оценка моделей  
5. Streamlit-приложение  
---
## 3. Результаты
- LogisticRegression: ROC AUC ~0.XX  
- Баланс классов: 1→~4%  
- Важные признаки: Tool wear, Rotational speed  
---
## 4. Дальнейшие шаги
- Сравнение RandomForest, XGBoost  
- Сохранение лучшей модели (pickle/joblib)  
- Видео-демонстрация  
    """
    # Опции слайдов
    with st.sidebar:
        st.header("Параметры слайдов")
        theme = st.selectbox("Тема", ["black","white","league","beige","sky","night"])
        height = st.number_input("Высота", 400, 800, 600)
        transition = st.selectbox("Переход", ["slide","zoom","none"])
    rs.slides(md, height=height, theme=theme, config={"transition": transition})