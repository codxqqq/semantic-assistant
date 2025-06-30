import pandas as pd
import requests
import re
from io import BytesIO
from sentence_transformers import SentenceTransformer, util
import pymorphy2
import functools

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

@functools.lru_cache(maxsize=10000)
def lemmatize_cached(word):
    return lemmatize(word)

# Синонимические группы
SYNONYM_GROUPS = [
    ["сим", "симка", "симкарта", "сим-карта", "сим-карте", "симке", "симку", "симки"],
    ["кредитка", "кредитная карта", "кредитной картой", "картой"],
    ["наличные", "наличка", "наличными"]
]

# Построение отображения леммы на каноническую форму
SYNONYM_MAP = {}
for group in SYNONYM_GROUPS:
    lemmas = sorted({lemmatize(w.lower()) for w in group})
    base = lemmas[0]
    for lemma in lemmas:
        SYNONYM_MAP[lemma] = base

def normalize_lemma(lemma):
    return SYNONYM_MAP.get(lemma, lemma)

# Ссылки на Excel-файлы
GITHUB_CSV_URLS = [
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data1.xlsx",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data2.xlsx",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data3.xlsx"
]

def lemmatize_phrase(phrase_proc):
    words = re.findall(r"\w+", phrase_proc)
    return {normalize_lemma(lemmatize_cached(w)) for w in words}

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
    df = df.assign(phrase_list=df['phrase'].str.split('/')).explode('phrase_list')
    df['phrase'] = df['phrase_list'].str.strip()
    df['phrase_proc'] = df['phrase'].apply(preprocess)
    df['phrase_lemmas'] = df['phrase_proc'].apply(lemmatize_phrase)

    # ⚡ Предварительное кодирование эмбеддингов
    model = get_model()
    df['embedding'] = model.encode(df['phrase_proc'].tolist(), convert_to_tensor=True, show_progress_bar=False)

    return df[['phrase', 'phrase_proc', 'phrase_full', 'topics', 'phrase_lemmas', 'embedding']]

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
    return pd.concat(dfs, ignore_index=True)

# Семантический поиск
def semantic_search(query, df, top_k=5, threshold=0.5):
    model = get_model()
    query_proc = preprocess(query)
    query_emb = model.encode(query_proc, convert_to_tensor=True)
    sims = util.pytorch_cos_sim(query_emb, list(df['embedding']))[0]

    results = [
        (float(score), df.iloc[idx]['phrase_full'], df.iloc[idx]['topics'])
        for idx, score in enumerate(sims) if float(score) >= threshold
    ]
    return sorted(results, key=lambda x: x[0], reverse=True)[:top_k]

# ✅ Точный поиск (оптимизированный)
def keyword_search(query, df):
    query_proc = preprocess(query)
    query_words = re.findall(r"\w+", query_proc)
    query_lemmas = [normalize_lemma(lemmatize_cached(word)) for word in query_words]

    matched = []
    for row in df.itertuples():
        if all(
            any(ql == pl for pl in row.phrase_lemmas)
            for ql in query_lemmas
        ):
            matched.append((row.phrase, row.topics))
    return matched
