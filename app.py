import streamlit as st

PAGES = {
    "Анализ и модель": {
        "module": "analysis_and_model",
        "func": "analysis_and_model_page"
    },
    "Прогноз онлайн": {
        "module": "analysis_and_model",
        "func": "prediction_page"
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
    module = __import__(page_info["module"])
    getattr(module, page_info["func"])()

if __name__ == "__main__":
    main()