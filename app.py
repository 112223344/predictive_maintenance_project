import streamlit as st

# Опишите страницы приложения
PAGES = {
    "Анализ и модель": {
        "module": "analysis_and_model",
        "func": "analysis_and_model_page"
    },
    "Презентация": {
        "module": "presentation",
        "func": "presentation_page"
    }
}

def main():
    st.sidebar.title("Навигация")
    page = st.sidebar.radio("Перейдите на страницу:", list(PAGES.keys()))

    page_info = PAGES[page]
    # Динамически импортируем нужную функцию
    module = __import__(page_info["module"])
    getattr(module, page_info["func"])()

if __name__ == "__main__":
    main()