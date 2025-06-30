import pandas as pd
import requests
import re
from io import BytesIO
from sentence_transformers import SentenceTransformer, util
import pymorphy2
import functools
import torch

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

# ✅ Кэшируемая лемматизация для ускорения точного поиска
@functools.lru_cache(maxsize=10000)
def lemmatize_cached(word):
    return lemmatize(word)

# Синонимические группы
SYNONYM_GROUPS = [
    ["сим", "симка", "симкарта", "сим-карта", "сим-карте", "симке", "симку", "симки"],
    ["кредитка", "кредитная карта", "кредитной картой", "картой"],
    ["наличные", "наличка", "наличными"]
]

# Построение словаря синонимов
SYNONYM_DICT = {}
for group in SYNONYM_GROUPS:
    lemmas = {lemmatize(w.lower()) for w in group}
    for lemma in lemmas:
        SYNONYM_DICT[lemma] = lemmas

# Ссылки на Excel-файлы
GITHUB_CSV_URLS = [
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data1.xlsx",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data2.xlsx",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data3.xlsx"
]

# Разделение фраз по /
def split_by_slash(phrase):
    parts = [p.strip() for p in str(phrase).split("/") if p.strip()]
    return parts if parts else [phrase]

# ✅ Векторизованная загрузка Excel-файла
def load_excel(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Ошибка загрузки {url}")
    df = pd.read_excel(BytesIO(response.content))

    topic_cols = [col for col in df.columns if col.lower().startswith("topics")]
    if not topic_cols:
        raise KeyError("Не найдены колонки topics")

    df['topics'] = df[topic_cols].astype(str).agg(lambda x: [v for v in x if v and v != 'nan'], axis=1)
    df['phrase_full'] = df['phrase']
    df['phrase_list'] = df['phrase'].apply(split_by_slash)
    df = df.explode('phrase_list', ignore_index=True)
    df['phrase'] = df['phrase_list']
    df['phrase_proc'] = df['phrase'].apply(preprocess)
    return df[['phrase', 'phrase_proc', 'phrase_full', 'topics']]

# 🔁 Вычисление и добавление эмбеддингов
def compute_embeddings(df):
    model = get_model()
    df['embedding'] = model.encode(df['phrase_proc'].tolist(), convert_to_tensor=True)
    return df

# Загрузка всех Excel-файлов
def load_all_excels():
    dfs = []
    for url in GITHUB_CSV_URLS:
        try:
            dfs.append(load_excel(url))
        except Exception as e:
            print(f"⚠️ Ошибка с {url}: {e}")
    if not dfs:
        raise ValueError("Не удалось загрузить ни одного файла")
    df = pd.concat(dfs, ignore_index=True)
    return compute_embeddings(df)

# ✅ Семантический поиск (с использованием кэшированных эмбеддингов)
def semantic_search(query, df, top_k=5, threshold=0.5):
    model = get_model()
    query_proc = preprocess(query)
    query_emb = model.encode(query_proc, convert_to_tensor=True)

    sims = util.pytorch_cos_sim(query_emb, torch.stack(df['embedding'].tolist()))[0]
    results = [
        (float(score), df.iloc[idx]['phrase_full'], df.iloc[idx]['topics'])
        for idx, score in enumerate(sims) if float(score) >= threshold
    ]
    return sorted(results, key=lambda x: x[0], reverse=True)[:top_k]

# ✅ Точный поиск (оптимизированный через apply)
def keyword_search(query, df):
    query_proc = preprocess(query)
    query_words = re.findall(r"\w+", query_proc)
    query_lemmas = [lemmatize_cached(word) for word in query_words]

    def match_row(row):
        phrase_words = re.findall(r"\w+", row.phrase_proc)
        phrase_lemmas = {lemmatize_cached(word) for word in phrase_words}
        return all(
            any(ql in SYNONYM_DICT.get(pl, {pl}) for pl in phrase_lemmas)
            for ql in query_lemmas
        )

    matches = df[df.apply(match_row, axis=1)]
    return list(zip(matches['phrase'], matches['topics']))
