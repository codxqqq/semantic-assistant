import streamlit as st
from utils import load_all_excels, semantic_search, keyword_search

st.set_page_config(page_title="Semantic Assistant", layout="centered")
st.title("🤖 Semantic Assistant")

# Кэшируем данные
@st.cache_data
def get_data():
    return load_all_excels()

# Загружаем данные
df = get_data()

# Тематики из всех строк
all_topics = sorted({topic for topics in df["topics"] for topic in topics})
selected_topics = st.multiselect("Фильтр по тематикам (необязательно):", all_topics)

# Функция фильтрации (маленькая версия)
def filter_by_topics(df, selected_topics):
    if not selected_topics:
        return df
    return df[df['topics'].apply(lambda t: any(topic in t for topic in selected_topics))]

# Показываем результаты фильтра, если выбраны тематики
if selected_topics:
    filtered_df = filter_by_topics(df, selected_topics)
    if not filtered_df.empty:
        st.markdown("### 📂 Фразы по выбранным тематикам:")
        for _, row in filtered_df.iterrows():
            st.markdown(f"- **{row['phrase']}** → {', '.join(row['topics'])}")
    else:
        st.warning("Нет фраз по выбранным тематикам.")

# Поле для ввода запроса
query = st.text_input("Введите ваш запрос:")

# Обработка запроса
if query:
    try:
        results = semantic_search(query, df)

        if results:
            st.markdown("### 🔍 Результаты умного поиска:")
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
        st.error(f"Ошибка при загрузке данных: {e}")
