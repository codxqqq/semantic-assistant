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
                'phrase_full': phrase,  # новая колонка для отображения
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
            phrase_full = df.iloc[idx]['phrase_full']
            topics = df.iloc[idx]['topics']
            results.append((score, phrase_full, topics))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]

# Точный поиск с учетом всех слов и синонимов
def keyword_search(query, df):
    query_proc = preprocess(query)
    query_words = re.findall(r"\w+", query_proc)
    query_stems = [stemmer.stem(word) for word in query_words]

    matched = []
    for _, row in df.iterrows():
        phrase_words = re.findall(r"\w+", row['phrase_proc'])
        phrase_stems = {stemmer.stem(word) for word in phrase_words}

        # Все слова запроса (или их синонимы) должны присутствовать в фразе
        if all(
            any(
                qs in SYNONYM_DICT.get(ps, {ps})
                for ps in phrase_stems
            )
            for qs in query_stems
        ):
            matched.append((row['phrase'], row['topics']))

    return matched


