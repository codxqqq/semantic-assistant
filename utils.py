import pandas as pd
import requests
import re
from io import BytesIO
from sentence_transformers import SentenceTransformer, util
import pymorphy2
import functools

try:
    import streamlit as st
    cache_decorator = st.cache_data
except ImportError:
    cache_decorator = lambda func: func  # если не используешь Streamlit

# ⚡ Ленивое создание модели
@functools.lru_cache(maxsize=1)
def get_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# ⚡ Ленивый лемматизатор
@functools.lru_cache(maxsize=1)
def get_morph():
    return pymorphy2.MorphAnalyzer()

# Нормализация строки
def preprocess(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

# Лемматизация
def lemmatize(word):
    return get_morph().parse(word)[0].normal_form

# ✅ Кэшируемая лемматизация
@functools.lru_cache(maxsize=10000)
def lemmatize_cached(word):
    return lemmatize(word)

# ✅ Предсобранный словарь синонимов
SYNONYM_DICT = {
    'сим': {'сим', 'симка', 'симкарта'},
    'кредитка': {'кредитка', 'карта'},
    'наличные': {'наличные', 'наличка'}
}

# Ссылки на CSV-файлы
GITHUB_CSV_URLS = [
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data1.csv",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data2.csv",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data3.csv"
]

# Разделение фраз по /
def split_by_slash(phrase):
    parts = [p.strip() for p in str(phrase).split("/") if p.strip()]
    return parts if parts else [phrase]

# ✅ Загрузка CSV-файла
def load_csv(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Ошибка загрузки {url}")
    df = pd.read_csv(BytesIO(response.content), encoding='utf-8')

    topic_cols = [col for col in df.columns if col.lower().startswith("topics")]
    if not topic_cols:
        raise KeyError("Не найдены колонки topics")

    df['topics'] = df[topic_cols].astype(str).agg(lambda x: [v for v in x if v and v != 'nan'], axis=1)
    df['phrase_full'] = df['phrase']
    df['phrase_list'] = df['phrase'].apply(split_by_slash)
    df = df.explode('phrase_list', ignore_index=True)
    df['phrase'] = df['phrase_list']
    df['phrase_proc'] = df['phrase'].apply(preprocess)

    # ✅ Предвычисляем леммы фразы один раз
    df['phrase_lemmas'] = df['phrase_proc'].apply(
        lambda text: {lemmatize_cached(w) for w in re.findall(r"\w+", text)}
    )

    return df[['phrase', 'phrase_proc', 'phrase_full', 'phrase_lemmas', 'topics']]

# ✅ Кэшируемая загрузка всех CSV-файлов
@cache_decorator
def load_all_excels():
    dfs = []
    for url in GITHUB_CSV_URLS:
        try:
            dfs.append(load_csv(url))
        except Exception as e:
            print(f"⚠️ Ошибка с {url}: {e}")
    if not dfs:
        raise ValueError("Не удалось загрузить ни одного файла")
    return pd.concat(dfs, ignore_index=True)

# Семантический поиск
def semantic_search(query, df, top_k=5, threshold=0.5):
    model = get_model()
    query_proc = preprocess(query)
    query_emb = model.encode(query_proc, convert_to_tensor=True)
    phrase_embs = model.encode(df['phrase_proc'].tolist(), convert_to_tensor=True)

    sims = util.pytorch_cos_sim(query_emb, phrase_embs)[0]
    results = [
        (float(score), df.iloc[idx]['phrase_full'], df.iloc[idx]['topics'])
        for idx, score in enumerate(sims) if float(score) >= threshold
    ]
    return sorted(results, key=lambda x: x[0], reverse=True)[:top_k]

# ✅ Точный поиск (оптимизированный)
def keyword_search(query, df):
    query_proc = preprocess(query)
    query_words = re.findall(r"\w+", query_proc)
    query_lemmas = set(map(lemmatize_cached, query_words))  # 🔥 Быстрее, чем list

    matched = []
    for row in df.itertuples():
        phrase_lemmas = row.phrase_lemmas

        if all(
            any(ql in SYNONYM_DICT.get(pl, {pl}) for pl in phrase_lemmas)
            for ql in query_lemmas
        ):
            matched.append((row.phrase, row.topics))
    return matched
