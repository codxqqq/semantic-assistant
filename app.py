import streamlit as st
from utils import load_all_excels, semantic_search, keyword_search

st.set_page_config(page_title="Semantic Assistant", layout="centered")
st.title("🤖 Semantic Assistant")

# Кэшируем загрузку
@st.cache_data
def get_data():
    return load_all_excels()

df = get_data()

# 🔎 ФИЛЬТР ПО ТЕМАТИКАМ (НЕЗАВИСИМЫЙ)
all_topics = sorted({t for topics in df['topics'] for t in topics})
selected_topics = st.multiselect("Фильтр по тематикам:", all_topics)

if selected_topics:
    st.markdown("### 🗂️ Фразы по выбранным тематикам:")
    for _, row in df.iterrows():
        phrase = row["phrase_full"]
        topics = row["topics"]
        if any(t in selected_topics for t in topics):
            st.markdown(f"- **{phrase}** → {', '.join(topics)}")

# 🔍 ПОИСК ПО ЗАПРОСУ (НЕ ЗАВИСИТ ОТ ФИЛЬТРА)
query = st.text_input("Введите ваш запрос:")

if query:
    try:
        results = semantic_search(query, df)
        if results:
            st.markdown("### 🤖 Умный поиск:")
            for score, phrase_full, topics in results:
                st.markdown(f"- **{phrase_full}** → {', '.join(topics)} (_{score:.2f}_)")
        else:
            st.warning("Совпадений не найдено в умном поиске.")

        exact_results = keyword_search(query, df)
        if exact_results:
            st.markdown("### 🧷 Точный поиск:")
            for phrase, topics in exact_results:
                st.markdown(f"- **{phrase}** → {', '.join(topics)}")
        else:
            st.info("Ничего не найдено в точном поиске.")
    except Exception as e:
        st.error(f"Ошибка при поиске: {e}")
