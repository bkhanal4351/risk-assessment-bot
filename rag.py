import os
import csv
import hashlib
from datetime import datetime
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from groq import Groq
from dotenv import load_dotenv

from config import (
    CSV_PATH, EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE, TOP_K, MAX_RECORDS,
    HIGH_CONFIDENCE_THRESHOLD, MODERATE_CONFIDENCE_THRESHOLD,
    LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, CHAT_HISTORY_WINDOW,
    SUMMARIZATION_THRESHOLD, AGGREGATE_KEYWORDS, SYSTEM_PROMPT,
    FEEDBACK_CSV_PATH, FEEDBACK_LOOKBACK,
)

load_dotenv()


# =============================================================================
# PURE HELPER FUNCTIONS
# =============================================================================

def row_to_text(row):
    """Converts a DataFrame row into a plain-text sentence for embedding and
    LLM context."""
    return (
        f"Risk: {row['risk_title']}. "
        f"Description: {row['risk_description']}. "
        f"Risk Level: {row['risk_level']}. "
        f"Primary Control ({row['primary_control_type']}): {row['primary_control_description']}. "
        f"Secondary Control ({row['secondary_control_type']}): {row['secondary_control_description']}."
    )


def confidence_label(score):
    """Converts cosine similarity score to (label, color) tuple."""
    if score >= HIGH_CONFIDENCE_THRESHOLD:
        return "High", "green"
    elif score >= MODERATE_CONFIDENCE_THRESHOLD:
        return "Moderate", "orange"
    else:
        return "Low", "red"


def _damerau_levenshtein(s1, s2):
    """Computes Damerau-Levenshtein distance (counts transpositions as 1 edit)."""
    len1, len2 = len(s1), len(s2)
    d = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        d[i][0] = i
    for j in range(len2 + 1):
        d[0][j] = j
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + cost,
            )
            if i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[len1][len2]


def _stem_match(word1, word2, min_chars=4, ratio=0.75):
    """Check if two words match via prefix overlap or edit distance.
    First tries prefix matching for inflected forms (spill <-> spilling).
    Falls back to Damerau-Levenshtein distance for transposition typos
    (farud <-> fraud). Requires words to be at least min_chars long."""
    shorter_len = min(len(word1), len(word2))
    if shorter_len < min_chars:
        return word1 == word2
    # Prefix match for inflected forms
    shared = 0
    for c1, c2 in zip(word1, word2):
        if c1 != c2:
            break
        shared += 1
    if shared >= max(min_chars, shorter_len * ratio):
        return True
    # Damerau-Levenshtein fallback for transposition/substitution typos
    # Only for words of similar length (avoids matching unrelated words)
    if abs(len(word1) - len(word2)) > 2:
        return False
    max_dist = 1 if shorter_len <= 5 else 2
    return _damerau_levenshtein(word1, word2) <= max_dist


def summarize_records(records_text, max_individual=SUMMARIZATION_THRESHOLD,
                      force_summary=False):
    """Groups large record sets by risk title instead of listing every record.
    Prevents token overflow when sending context to the LLM.
    When force_summary=True, always includes title counts even for small sets."""
    if len(records_text) <= max_individual and not force_summary:
        return "\n".join(records_text)

    title_counts = {}
    for record in records_text:
        if record.startswith("Risk: "):
            title = record.split(". Description:")[0].replace("Risk: ", "")
            title_counts[title] = title_counts.get(title, 0) + 1

    summary_lines = [f"Total matching records: {len(records_text)}"]
    summary_lines.append("\nRecords by risk title:")
    for title, count in sorted(title_counts.items(), key=lambda x: -x[1]):
        summary_lines.append(f"  {title}: {count} records")

    show_count = min(max_individual, len(records_text))
    summary_lines.append(f"\nSample records (showing {show_count} of {len(records_text)}):")
    for record in records_text[:show_count]:
        summary_lines.append(record)

    return "\n".join(summary_lines)


# =============================================================================
# CACHED DATA LOADING
# =============================================================================

def _get_csv_hash():
    """Computes SHA-256 of the CSV file. Called on every app rerun (cheap I/O
    operation) so that downstream caches detect data changes immediately."""
    with open(CSV_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


csv_hash = _get_csv_hash()


@st.cache_data
def load_data(csv_hash):
    """Loads the CSV, standardizes columns, pre-computes lowercase columns
    for fast keyword lookups, and converts all rows to text sentences."""
    data = pd.read_csv(CSV_PATH)
    data.rename(columns={
        "Risk Title": "risk_title",
        "Risk Statement/Description": "risk_description",
        "Risk Level (Inherent)": "risk_level",
        "Primary Control Type": "primary_control_type",
        "Primary Control Description": "primary_control_description",
        "Secondary Control Type": "secondary_control_type",
        "Secondary Control Description": "secondary_control_description"
    }, inplace=True)
    data["secondary_control_type"] = data["secondary_control_type"].fillna("N/A")
    data["secondary_control_description"] = data["secondary_control_description"].fillna("N/A")

    data["_risk_level_lower"] = data["risk_level"].str.lower()
    data["_primary_control_type_lower"] = data["primary_control_type"].str.lower()
    data["_risk_title_lower"] = data["risk_title"].str.lower()
    data["_risk_description_lower"] = data["risk_description"].astype(str).str.lower()

    sentences = [row_to_text(row) for _, row in data.iterrows()]
    return data, sentences


df, row_sentences = load_data(csv_hash)


@st.cache_resource
def load_model_and_embeddings(csv_hash):
    """Loads the sentence transformer model and encodes all row sentences."""
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE)
    embeddings = model.encode(row_sentences, convert_to_tensor=True)
    return model, embeddings


embedding_model, row_embeddings = load_model_and_embeddings(csv_hash)


@st.cache_data
def build_summary(csv_hash):
    """Pre-computes aggregate statistics for the full dataset."""
    lines = []
    lines.append(f"Total risks in registry: {len(df)}")

    level_counts = df['risk_level'].value_counts()
    lines.append("\nRisks per risk level:")
    for level, count in level_counts.items():
        lines.append(f"  {level}: {count}")

    primary_type_counts = df['primary_control_type'].value_counts()
    lines.append(f"\nTotal unique primary control types: {len(primary_type_counts)}")
    lines.append("\nRisks per primary control type:")
    for ctrl, count in primary_type_counts.items():
        lines.append(f"  {ctrl}: {count}")

    secondary_type_counts = df['secondary_control_type'].value_counts()
    lines.append("\nRisks per secondary control type:")
    for ctrl, count in secondary_type_counts.items():
        lines.append(f"  {ctrl}: {count}")

    title_counts = df['risk_title'].value_counts()
    if title_counts.max() > 1:
        lines.append("\nMost frequent risk titles:")
        for title, count in title_counts.head(10).items():
            lines.append(f"  {title}: {count}")

    return "\n".join(lines)


data_summary = build_summary(csv_hash)


# =============================================================================
# GROQ CLIENT
# =============================================================================

def _init_groq_client():
    """Initializes the Groq API client. Shows an error in the Streamlit UI
    if the API key is missing."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found. Please set it in your .env file.")
        st.stop()
    return Groq(api_key=api_key)


client = _init_groq_client()


# =============================================================================
# CORE LOGIC
# =============================================================================

def is_aggregate_question(question):
    """Two-pass aggregate detection. Returns True only if no specific risk
    title is mentioned AND aggregate keywords are present."""
    q_lower = question.lower()

    # Pass 1: Check if a specific risk title is mentioned
    q_words = [w for w in q_lower.split() if len(w) > 3]
    for title in df['_risk_title_lower'].dropna().unique():
        title_words = [w for w in title.split() if len(w) > 3]
        matches = sum(1 for tw in title_words if any(_stem_match(tw, qw) for qw in q_words))
        if len(title_words) >= 2 and matches >= 2:
            return False
        if len(title_words) == 1 and matches == 1:
            return False

    # Pass 2: Check for aggregate keywords
    return any(kw in q_lower for kw in AGGREGATE_KEYWORDS)


def lookup_records(question):
    """Keyword-based record search using pre-computed lowercase columns.
    Uses prefix-based stem matching so inflected forms like 'spilling'
    match 'spill', 'controls' match 'control', etc."""
    q_lower = question.lower()
    q_words = [w for w in q_lower.split() if len(w) > 3]
    matched_indices = set()

    for level in df['_risk_level_lower'].dropna().unique():
        if level in q_lower:
            matched_indices.update(df.index[df['_risk_level_lower'] == level])

    for ctrl in df['_primary_control_type_lower'].dropna().unique():
        if ctrl in q_lower:
            matched_indices.update(df.index[df['_primary_control_type_lower'] == ctrl])

    # Common words that appear in many titles — not distinctive enough for 1-word match
    _common_title_words = {
        "risk", "control", "management", "system", "process", "data",
        "failure", "response", "delay", "lack", "loss", "review",
        "monitoring", "compliance", "program", "plan", "report",
    }

    for title in df['_risk_title_lower'].dropna().unique():
        title_words = [w for w in title.split() if len(w) > 3]
        matching_count = sum(1 for tw in title_words if any(_stem_match(tw, qw) for qw in q_words))
        if len(title_words) >= 2 and matching_count >= 2:
            matched_indices.update(df.index[df['_risk_title_lower'] == title])
        elif len(title_words) == 1 and matching_count == 1:
            matched_indices.update(df.index[df['_risk_title_lower'] == title])
        elif len(title_words) >= 2 and matching_count == 1:
            # Allow 1-word match if the matched word is distinctive (not generic)
            matched_tw = [tw for tw in title_words if any(_stem_match(tw, qw) for qw in q_words)]
            if matched_tw and matched_tw[0] not in _common_title_words:
                matched_indices.update(df.index[df['_risk_title_lower'] == title])

    if not matched_indices:
        for idx, row in df.iterrows():
            desc_words = [w for w in row['_risk_description_lower'].split() if len(w) > 3]
            if sum(1 for qw in q_words if any(_stem_match(qw, dw) for dw in desc_words)) >= 2:
                matched_indices.add(idx)

    return df.loc[list(matched_indices)]


def ask_llm_stream(question, context, chat_history=None, feedback_context=""):
    """Streams the LLM response via Groq API. Includes recent chat history
    and user feedback for improved answers."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if chat_history:
        for msg in chat_history[-CHAT_HISTORY_WINDOW:]:
            messages.append(msg)

    user_content = f"{context}\n\nQuestion: {question}"
    if feedback_context:
        user_content = f"{feedback_context}\n\n{user_content}"

    messages.append({
        "role": "user",
        "content": user_content
    })

    try:
        stream = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"Error communicating with Groq API: {e}"


def _score_record(record_text, q_embedding):
    """Returns cosine similarity between a record and the query embedding."""
    rec_embedding = embedding_model.encode(record_text, convert_to_tensor=True)
    return float(util.cos_sim(q_embedding, rec_embedding)[0][0])


def _run_retrieval(query):
    """Inner retrieval: keyword lookup + embedding search. Returns
    (lookup_texts, embedding_records, top_score, query, sources, scores_map)."""
    lookup_matches = lookup_records(query)
    lookup_texts = [row_to_text(row) for _, row in lookup_matches.iterrows()]

    q_embedding = embedding_model.encode(query, convert_to_tensor=True)
    all_scores = util.cos_sim(q_embedding, row_embeddings)[0]
    top_results = all_scores.topk(k=min(TOP_K, len(row_sentences)))
    top_indices = top_results.indices.tolist()
    top_scores = top_results.values.tolist()
    top_score = top_scores[0]
    embedding_records = [row_sentences[i] for i in top_indices]

    # Build a map of record text -> cosine score for embedding results
    scores_map = {}
    for idx, sim_score in zip(top_indices, top_scores):
        scores_map[row_sentences[idx]] = round(sim_score, 2)

    # Score keyword-only matches against the query embedding
    for text in lookup_texts:
        if text not in scores_map:
            scores_map[text] = round(_score_record(text, q_embedding), 2)

    sources = []
    for idx, sim_score in zip(top_indices[:5], top_scores[:5]):
        row_num = idx + 1
        title = df.iloc[idx]['risk_title']
        sources.append((row_num, title, round(sim_score, 2)))

    return lookup_texts, embedding_records, top_score, query, sources, scores_map


def retrieve_context(question):
    """Main RAG pipeline: runs keyword + embedding search and builds LLM context."""
    lookup_texts, embedding_records, top_score, query, sources, scores_map = _run_retrieval(question)

    is_agg = is_aggregate_question(query)

    all_records = lookup_texts + [r for r in embedding_records if r not in lookup_texts]

    record_count = len(all_records)
    if record_count > MAX_RECORDS:
        all_records = all_records[:MAX_RECORDS]

    if is_agg:
        context_records = summarize_records(all_records, force_summary=True)
        context = (
            f"THIS IS AN AGGREGATE QUESTION. Present counts, totals, and "
            f"breakdowns — do NOT use the Best Match / Other Controls format.\n\n"
            f"Dataset Summary:\n{data_summary}\n\n"
            f"Matching Records ({record_count} found):\n{context_records}"
        )
    else:
        best_match = all_records[0] if all_records else ""
        best_score = scores_map.get(best_match, 0)
        other_records = all_records[1:] if len(all_records) > 1 else []
        context = f"BEST MATCH (Relevance: {best_score}):\n{best_match}"
        if other_records:
            scored_others = []
            for rec in other_records:
                rec_score = scores_map.get(rec, 0)
                scored_others.append(f"[Relevance: {rec_score}] {rec}")
            other_text = summarize_records(scored_others)
            context += f"\n\nOTHER RELATED RECORDS ({len(other_records)}):\n{other_text}"

    return context, top_score, is_agg, sources


# =============================================================================
# FEEDBACK
# =============================================================================

def save_feedback(question, response_summary, rating, comment=""):
    """Appends a feedback entry to the feedback CSV file."""
    file_exists = os.path.exists(FEEDBACK_CSV_PATH)
    with open(FEEDBACK_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "question", "response_summary", "rating", "comment"])
        writer.writerow([
            datetime.now().isoformat(),
            question,
            response_summary[:200],
            rating,
            comment,
        ])


def load_recent_feedback():
    """Loads recent feedback entries from the feedback CSV. Returns a formatted
    string for inclusion in the LLM prompt, or empty string if none."""
    if not os.path.exists(FEEDBACK_CSV_PATH):
        return ""

    try:
        fb_df = pd.read_csv(FEEDBACK_CSV_PATH)
    except Exception:
        return ""

    if fb_df.empty:
        return ""

    recent = fb_df.tail(FEEDBACK_LOOKBACK)

    lines = ["USER FEEDBACK ON PAST ANSWERS (use this to improve your response):"]
    for _, row in recent.iterrows():
        rating = row.get("rating", "")
        rating_label = "👍" if rating == "up" else "👎" if rating == "down" else str(rating)
        comment = str(row.get("comment", "")).strip()
        entry = f"- Q: \"{row['question']}\" — Rated: {rating_label}"
        if comment and comment != "nan":
            entry += f" — Comment: \"{comment}\""
        lines.append(entry)

    return "\n".join(lines)
