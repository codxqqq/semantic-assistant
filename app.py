import streamlit as st
from utils import load_all_excels, semantic_search, keyword_search

st.set_page_config(page_title="Semantic Assistant", layout="centered")
st.title("🤖 Semantic Assistant")

query = st.text_input("Введите ваш запрос:")

# Выбор метода поиска
search_mode = st.radio("Выберите метод поиска:", ["Умный (семантический)", "Точный (по словам)"], horizontal=True)

# Только для семантического поиска — включение подзапросов
enable_split = False
if search_mode == "Умный (семантический)":
    enable_split = st.checkbox("🔍 Разбивать длинный запрос на подзапросы", value=True)

if query:
    try:
        df = load_all_excels()

        if search_mode == "Умный (семантический)":
            results = semantic_search(query, df, enable_split=enable_split)
            if results:
                st.markdown("### 🔍 Результаты умного поиска:")
                for score, phrase_full, topics in results:
                    st.markdown(f"- **{phrase_full}** → {', '.join(topics)} (_{score:.2f}_)")
            else:
                st.warning("Совпадений не найдено.")
        else:
            exact_results = keyword_search(query, df)
            if exact_results:
                st.markdown("### 🧷 Точный поиск:")
                for phrase, topics in exact_results:
                    st.markdown(f"- **{phrase}** → {', '.join(topics)}")
            else:
                st.info("Ничего не найдено.")

    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {e}")
