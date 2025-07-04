import streamlit as st
from utils import load_all_excels, semantic_search, keyword_search, filter_by_topics

st.set_page_config(page_title="Semantic Assistant", layout="centered")
st.title("🤖 Semantic Assistant")

@st.cache_data
def get_data():
    return load_all_excels()

df = get_data()

# 🔘 Извлекаем все уникальные тематики из датафрейма
all_topics = sorted({topic for topics in df['topics'] for topic in topics})

# 📍 Меню выбора тем (мультиселект)
selected_topics = st.multiselect("Фильтр по тематикам (необязательно):", all_topics)

query = st.text_input("Введите ваш запрос:")

if query:
    try:
        # 🔍 Умный поиск
        results = semantic_search(query, df)
        filtered_results = filter_by_topics(results, selected_topics)

        if filtered_results:
            st.markdown("### 🔍 Результаты умного поиска:")
            for score, phrase_full, topics in filtered_results:
                st.markdown(f"- **{phrase_full}** → {', '.join(topics)} (_{score:.2f}_)")
        else:
            st.warning("Совпадений не найдено в умном поиске с выбранными темами.")

        # 🧷 Точный поиск
        exact_results = keyword_search(query, df)
        filtered_exact = filter_by_topics(exact_results, selected_topics)

        if filtered_exact:
            st.markdown("### 🧷 Точный поиск:")
            for phrase, topics in filtered_exact:
                st.markdown(f"- **{phrase}** → {', '.join(topics)}")
        else:
            st.info("Ничего не найдено в точном поиске с выбранными темами.")

    except Exception as e:
        st.error(f"Ошибка при обработке запроса: {e}")
