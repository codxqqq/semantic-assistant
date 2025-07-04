import streamlit as st
from utils import load_all_excels, semantic_search, keyword_search, filter_by_topics

st.set_page_config(page_title="Semantic Assistant", layout="centered")
st.title("🤖 Semantic Assistant")

@st.cache_data
def get_data():
    return load_all_excels()

df = get_data()

# 📍 Блок 1. Фильтрация по тематикам
all_topics = sorted({topic for topics in df['topics'] for topic in topics})
selected_topics = st.multiselect("📂 Показать фразы по тематикам:", all_topics)

if selected_topics:
    filtered_df = df[df['topics'].apply(lambda topics: any(t in topics for t in selected_topics))]
    st.markdown("### 📋 Фразы с выбранными тематиками:")
    for row in filtered_df.itertuples():
        st.markdown(f"- **{row.phrase_full}** → {', '.join(row.topics)}")

st.divider()

# 📍 Блок 2. Умный и точный поиск
query = st.text_input("🔍 Введите ваш запрос для поиска:")

if query:
    try:
        results = semantic_search(query, df)
        if results:
            st.markdown("### 🤖 Умный поиск:")
            for score, phrase_full, topics in results:
                st.markdown(f"- **{phrase_full}** → {', '.join(topics)} (_{score:.2f}_)")
        else:
            st.warning("Умный поиск не нашёл совпадений.")

        exact_results = keyword_search(query, df)
        if exact_results:
            st.markdown("### 🧷 Точный поиск:")
            for phrase, topics in exact_results:
                st.markdown(f"- **{phrase}** → {', '.join(topics)}")
        else:
            st.info("Точный поиск не дал результатов.")

    except Exception as e:
        st.error(f"Ошибка: {e}")
