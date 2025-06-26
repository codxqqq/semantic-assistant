# utils.py
import pandas as pd
import requests
import re
from io import BytesIO
from sentence_transformers import SentenceTransformer, util
from nltk.stem.snowball import SnowballStemmer

# Модель для семантического поиска
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Стеммер для русского языка
stemmer = SnowballStemmer("russian")

# Глобальный словарь синонимов
SYNONYM_GROUPS = [
    ["сим", "симка", "симкарта", "сим-карта", "сим-карте", "симке", "симку", "симки"],
    ["кредитка", "кредитная карта", "кредитной картой", "картой"],
    ["наличные", "наличка", "наличными"]
]

# Построение взаимного словаря синонимов (быстрый доступ)
SYNONYM_DICT = {}
for group in SYNONYM_GROUPS:
    for word in group:
        stem = stemmer.stem(word.lower())
        SYNONYM_DICT[stem] = {stemmer.stem(w) for w in group}

# Ссылки на Excel-файлы
GITHUB_CSV_URLS = [
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data1.xlsx",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data2.xlsx",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data3.xlsx"
]

# Нормализация строки
def preprocess(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

# Расширение строки на подфразы по /
def split_by_slash(phrase):
    parts = [p.strip() for p in str(phrase).split("/") if p.strip()]
    return parts if len(parts) > 1 else [phrase]

# Загрузка одного Excel-файла с разделением по /
def load_excel(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Ошибка загрузки {url}")
    df = pd.read_excel(BytesIO(response.content))

    topic_cols = [col for col in df.columns if col.lower().startswith("topics")]
    if not topic_cols:
        raise KeyError("Не найдены колонки topics")

    rows = []
    for _, row in df.iterrows():
        phrase = row['phrase']
        topics = [t for t in row[topic_cols].fillna('').tolist() if t]
        for sub_phrase in split_by_slash(phrase):
            rows.append({
                'phrase': sub_phrase,
                'phrase_proc': preprocess(sub_phrase),
                'topics': topics
            })

    return pd.DataFrame(rows)

# Загрузка всех Excel-файлов
def load_all_excels():
    dfs = []
    for url in GITHUB_CSV_URLS:
        try:
            df = load_excel(url)
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Ошибка с {url}: {e}")
    if not dfs:
        raise ValueError("Не удалось загрузить ни одного файла")
    return pd.concat(dfs, ignore_index=True)

# Семантический поиск
def semantic_search(query, df, top_k=5, threshold=0.5):
    query_proc = preprocess(query)
    query_emb = model.encode(query_proc, convert_to_tensor=True)
    phrase_embs = model.encode(df['phrase_proc'].tolist(), convert_to_tensor=True)

    sims = util.pytorch_cos_sim(query_emb, phrase_embs)[0]
    results = []

    for idx, score in enumerate(sims):
        score = float(score)
        if score >= threshold:
            phrase = df.iloc[idx]['phrase']
            topics = df.iloc[idx]['topics']
            results.append((score, phrase, topics))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]

# Точный поиск
def keyword_search(query, df):
    query_proc = preprocess(query)
    query_stem = stemmer.stem(query_proc)

    # Получаем группу синонимов по стему запроса
    synonyms = SYNONYM_DICT.get(query_stem, {query_stem})

    matched = []
    for _, row in df.iterrows():
        words = re.findall(r"\w+", row['phrase_proc'])
        stems = [stemmer.stem(word) for word in words]

        # Совпадение по любому синониму
        if any(stem in synonyms for stem in stems):
            matched.append((row['phrase'], row['topics']))

    return matched
