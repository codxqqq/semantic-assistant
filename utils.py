import pandas as pd
import requests
import re
from io import BytesIO
from sentence_transformers import SentenceTransformer, util
import pymorphy2
import functools

@functools.lru_cache(maxsize=1)
def get_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

@functools.lru_cache(maxsize=1)
def get_morph():
    return pymorphy2.MorphAnalyzer()

def preprocess(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

def lemmatize(word):
    return get_morph().parse(word)[0].normal_form

@functools.lru_cache(maxsize=10000)
def lemmatize_cached(word):
    return lemmatize(word)

SYNONYM_GROUPS = [
    ["сим", "симка", "симкарта", "сим-карта", "сим-карте", "симке", "симку", "симки"],
    ["кредитка", "кредитная карта", "кредитная карточка", "кредитной картой"],
    ["дебетовка", "дебетовая карта", "дебетовая карточка", "дебетовой картой"],
    ["карта", "карточка"],
    ["наличные", "наличка", "наличными"]
]

SYNONYM_DICT = {}
for group in SYNONYM_GROUPS:
    lemmas = {lemmatize_cached(w) for phrase in group for w in re.findall(r"\w+", phrase)}
    for lemma in lemmas:
        SYNONYM_DICT[lemma] = lemmas

GITHUB_CSV_URLS = [
    "https://raw.githubusercontent.com/codxqqq/semantic-assistant/main/data4.xlsx",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data21.xlsx",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data31.xlsx"
]

def split_by_slash(phrase):
    parts = [p.strip() for p in str(phrase).split("/") if p.strip()]
    return parts if parts else [phrase]

def load_excel(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Ошибка загрузки {url}")
    df = pd.read_excel(BytesIO(response.content))

    topic_cols = [col for col in df.columns if col.lower().startswith("topics")]
    if not topic_cols:
        raise KeyError("Не найдены колонки topics")

    df['topics'] = df[topic_cols].astype(str).agg(lambda x: [v for v in x if v and v != 'nan'], axis=1)
    df['phrase_full'] = df['phrase'].apply(preprocess)  # нормализация

    df['phrase_list'] = df['phrase'].apply(split_by_slash)
    df = df.explode('phrase_list', ignore_index=True)
    df['phrase'] = df['phrase_list']
    df['phrase_proc'] = df['phrase'].apply(preprocess)
    df['phrase_lemmas'] = df['phrase_proc'].apply(
        lambda text: {lemmatize_cached(w) for w in re.findall(r"\w+", text)}
    )
    return df[['phrase', 'phrase_proc', 'phrase_full', 'phrase_lemmas', 'topics']]

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

def semantic_search(query, df, top_k=5, threshold=0.5):
    model = get_model()
    query_proc = preprocess(query)
    query_emb = model.encode(query_proc, convert_to_tensor=True)

    if 'phrase_embs' not in df.attrs:
        df.attrs['phrase_embs'] = model.encode(df['phrase_proc'].tolist(), convert_to_tensor=True)

    phrase_embs = df.attrs['phrase_embs']
    sims = util.pytorch_cos_sim(query_emb, phrase_embs)[0]

    seen = set()
    results = []
    for idx, score in enumerate(sims):
        if float(score) >= threshold:
            phrase_full = df.iloc[idx]['phrase_full']
            topics = tuple(sorted(df.iloc[idx]['topics']))
            key = (phrase_full, topics)
            if key not in seen:
                results.append((float(score), phrase_full, list(topics)))
                seen.add(key)
    return sorted(results, key=lambda x: x[0], reverse=True)[:top_k]

def keyword_search(query, df):
    query_proc = preprocess(query)
    query_words = re.findall(r"\w+", query_proc)
    query_lemmas = {lemmatize_cached(word) for word in query_words}

    matched = []
    seen_phrases = set()

    for row in df.itertuples():
        phrase_lemmas = row.phrase_lemmas
        expanded = set()
        for pl in phrase_lemmas:
            expanded.update(SYNONYM_DICT.get(pl, {pl}))
        if query_lemmas.issubset(expanded):
            if row.phrase_full not in seen_phrases:
                matched.append((row.phrase_full, row.topics))
                seen_phrases.add(row.phrase_full)
    return matched

def filter_by_topics(results, selected_topics):
    if not selected_topics:
        return results

    filtered = []
    seen = set()
    for item in results:
        if isinstance(item, tuple) and len(item) == 3:
            score, phrase, topics = item
            key = (phrase, tuple(sorted(topics)))
            if key not in seen and set(topics) & set(selected_topics):
                filtered.append((score, phrase, topics))
                seen.add(key)
        elif isinstance(item, tuple) and len(item) == 2:
            phrase, topics = item
            key = (phrase, tuple(sorted(topics)))
            if key not in seen and set(topics) & set(selected_topics):
                filtered.append((phrase, topics))
                seen.add(key)
    return filtered
