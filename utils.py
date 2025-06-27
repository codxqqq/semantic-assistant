import pandas as pd
import requests
import re
from io import BytesIO
from sentence_transformers import SentenceTransformer, util
import pymorphy2

# Модель для семантического поиска
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Лемматизатор pymorphy2
morph = pymorphy2.MorphAnalyzer()

# Синонимические группы
SYNONYM_GROUPS = [
    ["сим", "симка", "симкарта", "сим-карта", "сим-карте", "симке", "симку", "симки"],
    ["кредитка", "кредитная карта", "кредитной картой", "картой"],
    ["наличные", "наличка", "наличными"]
]

# Преобразование слов в леммы
def lemmatize(word):
    return morph.parse(word)[0].normal_form

# Построение словаря синонимов на основе лемм
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

# Нормализация строки
def preprocess(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

# Расширение строки на подфразы по /
def split_by_slash(phrase):
    parts = [p.strip() for p in str(phrase).split("/") if p.strip()]
    return parts if len(parts) > 1 else [phrase]

# Разбиение длинного запроса на подзапросы
def split_into_subqueries(text):
    text = preprocess(text)
    parts = re.split(r"\b(?:и|а|но|если|когда|после того как|потому что|,|\.|\n)\b", text)
    subqueries = [p.strip() for p in parts if len(p.strip()) >= 5]
    return subqueries if len(subqueries) > 1 else [text]

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
                'phrase_full': phrase,
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

# Семантический поиск с подзапросами
def semantic_search(query, df, top_k=5, threshold=0.5):
    subqueries = split_into_subqueries(query)
    phrase_embs = model.encode(df['phrase_proc'].tolist(), convert_to_tensor=True)

    all_results = []

    for subq in subqueries:
        subq_emb = model.encode(subq, convert_to_tensor=True)
        sims = util.pytorch_cos_sim(subq_emb, phrase_embs)[0]

        for idx, score in enumerate(sims):
            score = float(score)
            if score >= threshold:
                phrase_full = df.iloc[idx]['phrase_full']
                topics = df.iloc[idx]['topics']
                all_results.append((score, phrase_full, topics))

    # Удалим дубликаты и оставим top_k лучших
    seen = set()
    unique_results = []
    for r in sorted(all_results, key=lambda x: x[0], reverse=True):
        key = (r[1], tuple(r[2]))
        if key not in seen:
            unique_results.append(r)
            seen.add(key)
        if len(unique_results) >= top_k:
            break

    return unique_results

# Точный поиск с использованием лемм и синонимов
def keyword_search(query, df):
    query_proc = preprocess(query)
    query_words = re.findall(r"\w+", query_proc)
    query_lemmas = [lemmatize(word) for word in query_words]

    matched = []
    for _, row in df.iterrows():
        phrase_words = re.findall(r"\w+", row['phrase_proc'])
        phrase_lemmas = {lemmatize(word) for word in phrase_words}

        if all(
            any(
                ql in SYNONYM_DICT.get(pl, {pl})
                for pl in phrase_lemmas
            )
            for ql in query_lemmas
        ):
            matched.append((row['phrase'], row['topics']))

    return matched
