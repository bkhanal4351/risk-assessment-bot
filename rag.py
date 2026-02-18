import os
import hashlib
import difflib
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from groq import Groq
from dotenv import load_dotenv

from config import (
    CSV_PATH, EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE, TOP_K, MAX_RECORDS,
    HIGH_CONFIDENCE_THRESHOLD, MODERATE_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_FALLBACK_THRESHOLD,
    SPELL_CORRECTION_CUTOFF, SPELL_LENGTH_TOLERANCE,
    LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, CHAT_HISTORY_WINDOW,
    SUMMARIZATION_THRESHOLD, FOLLOW_UP_WORD_COUNT, MAX_EFFECTIVE_QUERY_WORDS,
    FOLLOW_UP_INDICATORS, AGGREGATE_KEYWORDS, STOPWORDS, SYSTEM_PROMPT,
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


def summarize_records(records_text, max_individual=SUMMARIZATION_THRESHOLD):
    """Groups large record sets by risk title instead of listing every record.
    Prevents token overflow when sending context to the LLM."""
    if len(records_text) <= max_individual:
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

    summary_lines.append(f"\nSample records (showing {max_individual} of {len(records_text)}):")
    for record in records_text[:max_individual]:
        summary_lines.append(record)

    return "\n".join(summary_lines)


def resolve_follow_up(question, messages, last_effective_query=None):
    """If the question looks like a follow-up, prepend the last effective
    search query (or fall back to last user message) so retrieval has
    enough context to find the right records.

    The effective query is capped at MAX_EFFECTIVE_QUERY_WORDS words to
    prevent embedding signal dilution after 3+ consecutive follow-ups."""
    if not messages:
        return question

    q_lower = question.lower()
    is_follow_up = (
        len(question.split()) < FOLLOW_UP_WORD_COUNT
        or any(indicator in q_lower for indicator in FOLLOW_UP_INDICATORS)
    )

    if is_follow_up:
        # Prefer the stored effective query (already includes accumulated topic context)
        if last_effective_query:
            # Cap length to prevent snowballing after many follow-ups
            capped = " ".join(last_effective_query.split()[:MAX_EFFECTIVE_QUERY_WORDS])
            return f"{capped} {question}"
        # Fallback: use the last raw user message
        for msg in reversed(messages):
            if msg["role"] == "user":
                return f"{msg['content']} {question}"
                break

    return question


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


@st.cache_data
def build_vocabulary(csv_hash):
    """Builds a vocabulary of known terms from the dataset for fuzzy matching."""
    vocab = set()
    for title in df['risk_title'].dropna().unique():
        vocab.update(w.lower() for w in title.split() if len(w) > 3)
    for desc in df['risk_description'].dropna().unique():
        vocab.update(w.lower() for w in desc.split() if len(w) > 3)
    for level in df['risk_level'].dropna().unique():
        vocab.add(level.lower())
    for ctrl in df['primary_control_type'].dropna().unique():
        vocab.add(ctrl.lower())
    for ctrl in df['secondary_control_type'].dropna().unique():
        vocab.add(ctrl.lower())
    return list(vocab)


vocabulary = build_vocabulary(csv_hash)


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

def correct_query(question):
    """Corrects misspelled words in the user's question by matching against
    known terms from the dataset. Skips stopwords and short words."""
    words = question.split()
    corrected = []
    for word in words:
        if len(word) <= 3:
            corrected.append(word)
            continue
        clean = word.lower().strip("?.,!;:")
        if clean in STOPWORDS:
            corrected.append(word)
            continue
        matches = difflib.get_close_matches(
            clean, vocabulary, n=1, cutoff=SPELL_CORRECTION_CUTOFF
        )
        if matches and matches[0] != clean and abs(len(matches[0]) - len(clean)) <= SPELL_LENGTH_TOLERANCE:
            corrected.append(matches[0])
        else:
            corrected.append(word)
    return " ".join(corrected)


def is_aggregate_question(question):
    """Two-pass aggregate detection. Returns True only if no specific risk
    title is mentioned AND aggregate keywords are present."""
    q_lower = question.lower()

    # Pass 1: Check if a specific risk title is mentioned
    for title in df['_risk_title_lower'].dropna().unique():
        title_words = [w for w in title.split() if len(w) > 3]
        if len(title_words) >= 2 and sum(1 for w in title_words if w in q_lower) >= 2:
            return False
        if len(title_words) == 1 and title_words[0] in q_lower:
            return False

    # Pass 2: Check for aggregate keywords
    return any(kw in q_lower for kw in AGGREGATE_KEYWORDS)


def lookup_records(question):
    """Keyword-based record search using pre-computed lowercase columns."""
    q_lower = question.lower()
    matched_indices = set()

    for level in df['_risk_level_lower'].dropna().unique():
        if level in q_lower:
            matched_indices.update(df.index[df['_risk_level_lower'] == level])

    for ctrl in df['_primary_control_type_lower'].dropna().unique():
        if ctrl in q_lower:
            matched_indices.update(df.index[df['_primary_control_type_lower'] == ctrl])

    for title in df['_risk_title_lower'].dropna().unique():
        title_words = [w for w in title.split() if len(w) > 3]
        matching_count = sum(1 for w in title_words if w in q_lower)
        if len(title_words) >= 2 and matching_count >= 2:
            matched_indices.update(df.index[df['_risk_title_lower'] == title])
        elif len(title_words) == 1 and matching_count == 1:
            matched_indices.update(df.index[df['_risk_title_lower'] == title])

    if not matched_indices:
        q_words = [w for w in q_lower.split() if len(w) > 3]
        for idx, row in df.iterrows():
            if sum(1 for w in q_words if w in row['_risk_description_lower']) >= 2:
                matched_indices.add(idx)

    return df.loc[list(matched_indices)]


def ask_llm_stream(question, context, chat_history=None):
    """Streams the LLM response via Groq API. Includes recent chat history
    for conversational context."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if chat_history:
        for msg in chat_history[-CHAT_HISTORY_WINDOW:]:
            messages.append(msg)

    messages.append({
        "role": "user",
        "content": f"{context}\n\nQuestion: {question}"
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


def _run_retrieval(query):
    """Inner retrieval: keyword lookup + embedding search. Returns
    (lookup_texts, embedding_records, top_score, corrected, sources)."""
    corrected = correct_query(query)

    lookup_matches = lookup_records(corrected)
    lookup_texts = [row_to_text(row) for _, row in lookup_matches.iterrows()]

    q_embedding = embedding_model.encode(corrected, convert_to_tensor=True)
    scores = util.cos_sim(q_embedding, row_embeddings)[0]
    top_results = scores.topk(k=min(TOP_K, len(row_sentences)))
    top_indices = top_results.indices.tolist()
    top_scores = top_results.values.tolist()
    top_score = top_scores[0]
    embedding_records = [row_sentences[i] for i in top_indices]

    sources = []
    for idx, sim_score in zip(top_indices[:5], top_scores[:5]):
        row_num = idx + 1
        title = df.iloc[idx]['risk_title']
        sources.append((row_num, title, round(sim_score, 2)))

    return lookup_texts, embedding_records, top_score, corrected, sources


def retrieve_context(question, messages=None, last_effective_query=None):
    """Main RAG pipeline: resolves follow-ups, corrects spelling, runs
    keyword + embedding search, and builds LLM context."""
    search_query = resolve_follow_up(question, messages or [], last_effective_query)
    lookup_texts, embedding_records, top_score, corrected, sources = _run_retrieval(search_query)

    # Fallback: if confidence is low and we have chat history, retry with
    # the stored effective query or previous user message prepended
    if top_score < LOW_CONFIDENCE_FALLBACK_THRESHOLD and messages and search_query == question:
        fallback_context = last_effective_query
        if not fallback_context:
            for msg in reversed(messages):
                if msg["role"] == "user":
                    fallback_context = msg['content']
                    break
        if fallback_context:
            enriched = f"{fallback_context} {question}"
            lookup_texts, embedding_records, top_score, corrected, sources = _run_retrieval(enriched)

    is_agg = is_aggregate_question(corrected)

    all_records = lookup_texts + [r for r in embedding_records if r not in lookup_texts]

    record_count = len(all_records)
    if record_count > MAX_RECORDS:
        all_records = all_records[:MAX_RECORDS]

    if is_agg:
        context_records = summarize_records(all_records)
        context = f"Dataset Summary:\n{data_summary}\n\nMatching Records ({record_count} found):\n{context_records}"
    else:
        best_match = all_records[0] if all_records else ""
        other_records = all_records[1:] if len(all_records) > 1 else []
        context = f"BEST MATCH:\n{best_match}"
        if other_records:
            other_text = summarize_records(other_records)
            context += f"\n\nOTHER RELATED RECORDS ({len(other_records)}):\n{other_text}"

    return context, top_score, corrected, is_agg, sources
