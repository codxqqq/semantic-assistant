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
    ["кредитная карта", "кредитка"],
    ["наличные", "наличка", "наличными"],
    ["дебетовая карта", "дебетовка", "дебетовая"],
    ["карточка", "карта"],
    ["потерял", "утеря", "потеря", "утерял"]
]

# Создаем словарь для замены фраз по приоритету (длина убыв.)
PHRASE_SYNONYMS = {}
for group in SYNONYM_GROUPS:
    group_sorted = sorted(group, key=len, reverse=True)
    canonical = group_sorted[0]
    for variant in group_sorted:
        PHRASE_SYNONYMS[variant] = canonical

SYNONYM_DICT = {}
for group in SYNONYM_GROUPS:
    lemmas = {lemmatize(w.lower()) for w in group}
    for lemma in lemmas:
        SYNONYM_DICT[lemma] = lemmas

def replace_synonyms_in_text(text):
    """Заменяет многословные и односоставные синонимы в тексте"""
    text_proc = text.lower()
    for phrase, canonical in sorted(PHRASE_SYNONYMS.items(), key=lambda x: -len(x[0])):
        pattern = rf"\b{re.escape(phrase)}\b"
        text_proc = re.sub(pattern, canonical, text_proc)
    return preprocess(text_proc)

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

    df['phrase_full'] = df['phrase']
    df['phrase_list'] = df['phrase'].apply(split_by_slash)

    # Удаляем дубликаты подфраз
    df = df.explode('phrase_list', ignore_index=True)
    df['phrase_list'] = df['phrase_list'].apply(str.strip)
    df.drop_duplicates(subset=['phrase_list'], inplace=True)

    df['phrase'] = df['phrase_list']
    df['phrase_proc'] = df['phrase'].apply(replace_synonyms_in_text)
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
    query_proc = replace_synonyms_in_text(query)
    query_emb = model.encode(query_proc, convert_to_tensor=True)

    if 'phrase_embs' not in df.attrs:
        df.attrs['phrase_embs'] = model.encode(df['phrase_proc'].tolist(), convert_to_tensor=True)

    phrase_embs = df.attrs['phrase_embs']
    sims = util.pytorch_cos_sim(query_emb, phrase_embs)[0]
    results = [
        (float(score), df.iloc[idx]['phrase_full'], df.iloc[idx]['topics'])
        for idx, score in enumerate(sims) if float(score) >= threshold
    ]
    return sorted(results, key=lambda x: x[0], reverse=True)[:top_k]

def keyword_search(query, df):
    query_proc = replace_synonyms_in_text(query)
    query_words = re.findall(r"\w+", query_proc)
    query_lemmas = [lemmatize_cached(word) for word in query_words]

    matched = []
    for row in df.itertuples():
        phrase_lemmas = row.phrase_lemmas
        if all(
            any(ql in SYNONYM_DICT.get(pl, {pl}) for pl in phrase_lemmas)
            for ql in query_lemmas
        ):
            matched.append((row.phrase_full, row.topics))
    return matched

def filter_by_topics(results, selected_topics):
    if not selected_topics:
        return results

    filtered = []
    for item in results:
        if isinstance(item, tuple) and len(item) == 3:
            score, phrase, topics = item
            if set(topics) & set(selected_topics):
                filtered.append((score, phrase, topics))
        elif isinstance(item, tuple) and len(item) == 2:
            phrase, topics = item
            if set(topics) & set(selected_topics):
                filtered.append((phrase, topics))
    return filtered
