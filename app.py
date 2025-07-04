import streamlit as st
from utils import load_all_excels, semantic_search, keyword_search

st.set_page_config(page_title="Semantic Assistant", layout="centered")
st.title("🤖 Semantic Assistant")

# Загружаем данные один раз
@st.cache_data
def get_data():
    return load_all_excels()

df = get_data()

# Получаем все уникальные тематики
all_topics = sorted({topic for topics in df['topics'] for topic in topics})
selected_topics = st.multiselect("🔎 Фильтр по тематикам:", all_topics)

query = st.text_input("Введите ваш запрос:")

def topic_match_any(topics, selected):
    return any(t in topics for t in selected)

if query:
    try:
        results = semantic_search(query, df)

        if results:
            st.markdown("### 🔍 Результаты умного поиска:")
            for score, phrase_full, topics in results:
                if not selected_topics or topic_match_any(topics, selected_topics):
                    topic_tags = ", ".join(topics)
                    st.markdown(f"- **{phrase_full}** → {topic_tags} (_{score:.2f}_)")
        else:
            st.warning("Совпадений не найдено в умном поиске.")

        exact_results = keyword_search(query, df)
        if exact_results:
            st.markdown("### 🧷 Точный поиск:")
            for phrase, topics in exact_results:
                if not selected_topics or topic_match_any(topics, selected_topics):
                    topic_tags = ", ".join(topics)
                    st.markdown(f"- **{phrase}** → {topic_tags}")
        else:
            st.info("Ничего не найдено в точном поиске.")

    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {e}")
