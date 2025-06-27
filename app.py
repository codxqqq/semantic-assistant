import streamlit as st
from utils import load_all_excels, semantic_search, keyword_search

st.set_page_config(page_title="Semantic Assistant", layout="centered")
st.title("🤖 Semantic Assistant")

query = st.text_input("Введите ваш запрос:")

if query:
    try:
        df = load_all_excels()
        results = semantic_search(query, df)

        if results:
            st.markdown("### 🔍 Результаты умного поиска:")
            for score, phrase_full, topics in results:
                st.markdown(f"- **{phrase_full}** → {', '.join(topics)} (_{score:.2f}_)")
        else:
            st.warning("Совпадений не найдено в умном поиске.")

        # Точный поиск всегда показывается снизу
        exact_results = keyword_search(query, df)
        if exact_results:
            st.markdown("### 🧷 Точный поиск:")
            for phrase, topics in exact_results:
                st.markdown(f"- **{phrase}** → {', '.join(topics)}")
        else:
            st.info("Ничего не найдено в точном поиске.")

    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {e}")
