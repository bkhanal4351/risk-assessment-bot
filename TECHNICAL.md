# Technical Documentation — EPA Risk Assessment Assistant

This document provides a line-by-line walkthrough of every module in the EPA Risk Assessment Assistant, covering algorithms, data structures, control flow, and design decisions.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Module: config.py](#module-configpy)
3. [Module: rag.py](#module-ragpy)
   - [Imports and Initialization](#imports-and-initialization)
   - [Helper Functions](#helper-functions)
   - [Data Loading and Caching](#data-loading-and-caching)
   - [Groq Client](#groq-client)
   - [Core Retrieval Logic](#core-retrieval-logic)
   - [LLM Generation](#llm-generation)
   - [Feedback System](#feedback-system)
4. [Module: app.py](#module-apppy)
   - [Page Configuration and Imports](#page-configuration-and-imports)
   - [Styling](#styling)
   - [Sidebar](#sidebar)
   - [Feedback UI](#feedback-ui)
   - [Source Citations](#source-citations)
   - [Main Chat Loop](#main-chat-loop)
5. [Feedback Loop — Full Lifecycle](#feedback-loop--full-lifecycle)
   - [How Feedback Is Collected](#how-feedback-is-collected)
   - [How Feedback Is Stored](#how-feedback-is-stored)
   - [How Feedback Is Loaded](#how-feedback-is-loaded)
   - [How Feedback Is Injected Into the LLM](#how-feedback-is-injected-into-the-llm)
   - [How Feedback Influences Future Answers](#how-feedback-influences-future-answers)
   - [Feedback Data Flow Diagram](#feedback-data-flow-diagram)
6. [Algorithms In Depth](#algorithms-in-depth)
   - [Cosine Similarity](#cosine-similarity)
   - [Sentence Embeddings (BAAI/bge-base-en-v1.5)](#sentence-embeddings-baaibge-base-en-v15)
   - [Two-Layer Fuzzy Matching (Prefix + Damerau-Levenshtein)](#two-layer-fuzzy-matching-prefix--damerau-levenshtein)
   - [Two-Pass Aggregate Detection](#two-pass-aggregate-detection)
   - [Distinctive Word Matching](#distinctive-word-matching)
   - [Record Summarization](#record-summarization)
   - [SHA-256 CSV Change Detection](#sha-256-csv-change-detection)
7. [Data Flow — End to End](#data-flow--end-to-end)
8. [Caching Strategy](#caching-strategy)
9. [Error Handling](#error-handling)
10. [Session State Management](#session-state-management)

---

## Architecture Overview

The app is split into three modules following separation of concerns:

```
config.py  →  Constants, thresholds, prompts, keyword lists
rag.py     →  RAG pipeline: data loading, retrieval, LLM calls, feedback I/O
app.py     →  Streamlit UI: sidebar, chat messages, feedback buttons, styling
```

Data flows in one direction: **User Input → app.py → rag.py (retrieval + LLM) → app.py (render)**. Configuration values are imported from `config.py` by both `rag.py` and `app.py`. The feedback CSV is the only shared mutable state between sessions.

---

## Module: config.py

`config.py` is a pure constants file — no logic, no imports, no side effects. Every tunable parameter lives here so the RAG pipeline and UI can be adjusted without modifying business logic.

### Line-by-Line

```python
# Lines 1-2: Module docstring comment
# config.py -- All constants, thresholds, prompts, and keyword lists.
# Change values here to tune the app's behavior without touching business logic.
```

**File Paths (line 7):**
```python
CSV_PATH = "Epa risk and control registry.csv"
```
Relative path to the EPA dataset. Used by `rag.py` to load the DataFrame and compute the SHA-256 hash for change detection.

**Embedding & Retrieval (lines 12-15):**
```python
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBEDDING_DEVICE = "cpu"
TOP_K = 10
MAX_RECORDS = 50
```
- `EMBEDDING_MODEL_NAME`: HuggingFace model ID for the sentence transformer. BGE-base produces 768-dimensional vectors and is top-ranked on the MTEB retrieval benchmark. Downloads ~110MB on first run, cached afterward by HuggingFace.
- `EMBEDDING_DEVICE`: Forces CPU inference. Set to `"cuda"` if a GPU is available for ~5x faster encoding.
- `TOP_K`: Number of top embedding results returned by semantic search. Higher values increase recall but also increase LLM context size.
- `MAX_RECORDS`: Hard cap on records sent to the LLM. Prevents token overflow for aggregate queries that match hundreds of records.

**Confidence Thresholds (lines 20-21):**
```python
HIGH_CONFIDENCE_THRESHOLD = 0.6
MODERATE_CONFIDENCE_THRESHOLD = 0.4
```
Used by `confidence_label()` to classify the top cosine similarity score into a color-coded badge:
- `>= 0.6` → High (green)
- `>= 0.4` → Moderate (orange)
- `< 0.4` → Low (red)

These thresholds are empirically tuned for the BGE-base model on EPA risk data. Other embedding models may require different thresholds.

**LLM Settings (lines 26-29):**
```python
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1500
CHAT_HISTORY_WINDOW = 6
```
- `LLM_MODEL`: Groq model identifier. Llama 3.3 70B provides strong instruction-following with fast inference via Groq's LPU hardware.
- `LLM_TEMPERATURE`: Near-deterministic output (0.1). Low temperature reduces hallucination for fact-based Q&A. Set higher (0.3-0.7) for more creative responses.
- `LLM_MAX_TOKENS`: Maximum response length. 1500 tokens ≈ 1000-1200 words, sufficient for structured risk summaries.
- `CHAT_HISTORY_WINDOW`: Number of recent messages included for conversational context. 6 messages = 3 user-assistant exchanges. Higher values provide more context but consume more tokens.

**Record Summarization (line 34):**
```python
SUMMARIZATION_THRESHOLD = 15
```
When more than 15 records match a query, `summarize_records()` groups them by risk title with counts instead of listing every record. This prevents exceeding the LLM's context window while still providing a complete picture.

**Aggregate Detection Keywords (lines 39-48):**
```python
AGGREGATE_KEYWORDS = [
    "most", "least", "how many", "count", "total", "average",
    "top", "bottom", "list all", "every", "each",
    "which level", "which type", "biggest", "smallest",
    "percentage", "breakdown", "summary", "statistics",
    "by level", "by type", "by control", "per level", "per type",
    "group by", "grouped by", "list of risks", "list risks",
    "how many risks", "high risk", "medium risk", "low risk",
    "all controls", "all risks",
]
```
Substring patterns checked against the lowercased user question in `is_aggregate_question()`. If any pattern appears AND no specific risk title is mentioned, the query is classified as aggregate. This changes the retrieval strategy from "find best match" to "provide dataset summary + all matching records."

**System Prompt (lines 53-95):**

The system prompt defines the LLM's persona, output format, and behavior rules. Key sections:

1. **Role definition** (line 54): "You are an EPA risk assessment assistant."
2. **Grounding rule** (line 55): "Answer questions using ONLY the risk and control data provided." — prevents hallucination from training data.
3. **Aggregate format** (lines 58-62): For broad questions, present counts and breakdowns.
4. **Specific format** (lines 63-79): Structured output with Best Match (risk title, risk level, cosine score, primary/secondary controls) and Other Controls to Consider.
5. **Bold formatting rule** (lines 80-84): Enforces consistent markdown bold labels for controls in every section.
6. **Typo tolerance** (lines 85-92): Instructs the LLM to interpret user typos generously with examples (`fraus→fraud`, `hazadorus→hazardous`, `spilling→spill`).
7. **No-match rule** (lines 93-94): Only say "No matching information found" if records are completely unrelated.

**Feedback Constants (lines 100-101):**
```python
FEEDBACK_CSV_PATH = "feedback.csv"
FEEDBACK_LOOKBACK = 10
```
- `FEEDBACK_CSV_PATH`: Location of the feedback file. Created at runtime, gitignored.
- `FEEDBACK_LOOKBACK`: Maximum number of recent feedback entries loaded into the LLM prompt. Higher values give the LLM more context about past user satisfaction but consume more tokens.

**UI Settings (lines 106-120):**
```python
CHAT_FONT_SIZE_PX = 17
APP_TITLE = "EPA Risk Assessment Assistant"
APP_ICON = "shield"
APP_DESCRIPTION = "..."
EXAMPLE_QUESTIONS = [...]
```
- `CHAT_FONT_SIZE_PX`: Custom font size injected via CSS for readability.
- `APP_TITLE` / `APP_ICON` / `APP_DESCRIPTION`: Used by `app.py` for `st.set_page_config()` and sidebar header.
- `EXAMPLE_QUESTIONS`: Pre-defined questions shown as clickable buttons in the sidebar.

---

## Module: rag.py

This is the core engine — ~410 lines covering data loading, dual retrieval, LLM generation, and feedback I/O.

### Imports and Initialization

```python
# Lines 1-9: Standard library + third-party imports
import os, csv, hashlib
from datetime import datetime
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from groq import Groq
from dotenv import load_dotenv
```

- `hashlib`: SHA-256 hashing for CSV change detection.
- `csv`: Low-level CSV writing for feedback (avoids pandas overhead for append-only writes).
- `sentence_transformers`: HuggingFace library for loading the BGE-base model and computing cosine similarity.
- `groq`: Official Groq Python SDK for streaming LLM inference.
- `dotenv`: Loads `.env` file into `os.environ` for the Groq API key.

```python
# Lines 11-17: Import all config constants
from config import (
    CSV_PATH, EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE, TOP_K, MAX_RECORDS,
    HIGH_CONFIDENCE_THRESHOLD, MODERATE_CONFIDENCE_THRESHOLD,
    LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, CHAT_HISTORY_WINDOW,
    SUMMARIZATION_THRESHOLD, AGGREGATE_KEYWORDS, SYSTEM_PROMPT,
    FEEDBACK_CSV_PATH, FEEDBACK_LOOKBACK,
)
```

```python
# Line 19: Load environment variables from .env
load_dotenv()
```
This must run before `_init_groq_client()` accesses `os.environ["GROQ_API_KEY"]`.

### Helper Functions

#### `row_to_text(row)` — Lines 26-35

```python
def row_to_text(row):
    return (
        f"Risk: {row['risk_title']}. "
        f"Description: {row['risk_description']}. "
        f"Risk Level: {row['risk_level']}. "
        f"Primary Control ({row['primary_control_type']}): {row['primary_control_description']}. "
        f"Secondary Control ({row['secondary_control_type']}): {row['secondary_control_description']}."
    )
```

**Purpose**: Converts a DataFrame row into a single natural-language sentence. This text serves two purposes:
1. **Embedding input**: The sentence transformer encodes this text into a 768-dim vector.
2. **LLM context**: This exact text is passed to the LLM as a retrieved record.

**Design decision**: A single sentence (not JSON or tabular format) is used because sentence transformers are trained on natural text and produce higher-quality embeddings for sentence-like inputs. The field labels (`Risk:`, `Description:`, etc.) help the LLM parse the structure.

#### `confidence_label(score)` — Lines 38-45

```python
def confidence_label(score):
    if score >= HIGH_CONFIDENCE_THRESHOLD:
        return "High", "green"
    elif score >= MODERATE_CONFIDENCE_THRESHOLD:
        return "Moderate", "orange"
    else:
        return "Low", "red"
```

**Purpose**: Maps a cosine similarity float to a human-readable label and Streamlit color code. Used by `app.py` to render the relevance badge.

**Return**: Tuple `(label_string, color_string)`. The color string is used with Streamlit's `:{color}[text]` markdown syntax.

#### `_damerau_levenshtein(s1, s2)` — Lines 48-66

```python
def _damerau_levenshtein(s1, s2):
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
```

**Algorithm**: Damerau-Levenshtein distance. Extends standard Levenshtein by counting adjacent transpositions as a single edit operation (cost 1 instead of 2). This is critical for catching keyboard typos where users swap two adjacent letters (e.g., "farud" → "fraud").

**Operations counted** (each costs 1):
- **Insertion**: "faud" → "fraud" (insert "r")
- **Deletion**: "frauds" → "fraud" (delete "s")
- **Substitution**: "fraod" → "fraud" (replace "o" with "u")
- **Transposition**: "farud" → "fraud" (swap "ar" to "ra")

Standard Levenshtein would count the "farud" → "fraud" transposition as 2 operations (delete "a" + insert "a"), making it harder to distinguish real typos from unrelated words.

**Time complexity**: O(len1 × len2). For typical query words (4-15 chars), this is negligible.

#### `_stem_match(word1, word2, min_chars=4, ratio=0.75)` — Lines 69-90

```python
def _stem_match(word1, word2, min_chars=4, ratio=0.75):
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
    if abs(len(word1) - len(word2)) > 2:
        return False
    max_dist = 1 if shorter_len <= 5 else 2
    return _damerau_levenshtein(word1, word2) <= max_dist
```

**Algorithm**: Two-layer fuzzy matching. First tries prefix matching for inflected forms, then falls back to Damerau-Levenshtein distance for transposition typos.

**Layer 1 — Prefix match** (fast path):
1. If either word is shorter than `min_chars` (4), require exact equality. Prevents false positives like "is" matching "it".
2. Count shared leading characters (`shared`).
3. If `shared >= max(4, 75% of shorter word)`, return `True`.

**Layer 2 — Edit distance fallback** (when prefix fails):
1. Reject if word lengths differ by more than 2 characters (prevents matching unrelated words of very different lengths).
2. For short words (≤5 chars): allow Damerau-Levenshtein distance ≤ 1.
3. For longer words (6+ chars): allow distance ≤ 2.

**Examples**:
| word1 | word2 | Layer 1 (prefix) | Layer 2 (DL distance) | match? |
|-------|-------|-------------------|----------------------|--------|
| "spill" | "spilling" | shared=5, threshold=4 → Yes | — | Yes (prefix) |
| "prevent" | "preventive" | shared=7, threshold=6 → Yes | — | Yes (prefix) |
| "fraud" | "fraus" | shared=4, threshold=4 → Yes | — | Yes (prefix) |
| "farud" | "fraud" | shared=1, threshold=4 → No | DL=1, max=1 → Yes | Yes (edit dist) |
| "enviroment" | "environment" | shared=6, threshold=8 → No | DL=1, max=2 → Yes | Yes (edit dist) |
| "hazadorus" | "hazardous" | shared=3, threshold=7 → No | DL=2, max=2 → Yes | Yes (edit dist) |
| "grant" | "plant" | shared=0, threshold=4 → No | DL=2, max=1 → No | No |
| "risk" | "response" | shared=1, threshold=4 → No | len diff=4 > 2 → skip | No |

**Why two layers?** Prefix matching is O(n) and handles the common case (inflected forms) efficiently. Edit distance is O(n²) but only runs when the prefix check fails. The length-difference guard and distance thresholds prevent false positives between unrelated words of similar length.

**Why not use NLTK/SpaCy stemmers?** To avoid adding a heavy NLP dependency for a single function. The two-layer approach handles both inflections and typos — standard stemmers only handle inflections, not transposition typos like "farud" → "fraud".

#### `summarize_records(records_text, max_individual)` — Lines 64-85

```python
def summarize_records(records_text, max_individual=SUMMARIZATION_THRESHOLD):
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
```

**Purpose**: Prevents token overflow when a query matches many records (e.g., "show all high risks" might match 400+ records).

**Algorithm**:
1. If records fit within `max_individual` (15), return them all joined by newlines.
2. Otherwise, extract risk titles using string parsing (split on `. Description:` delimiter).
3. Count occurrences of each title, sorted descending by count.
4. Return: total count → grouped title counts → sample of first 15 individual records.

**Why string parsing instead of DataFrame lookup?** The function receives pre-formatted text strings (output of `row_to_text()`), not DataFrame rows. Parsing the title back out avoids passing the DataFrame through the function chain.

### Data Loading and Caching

#### `_get_csv_hash()` — Lines 92-96

```python
def _get_csv_hash():
    with open(CSV_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()
```

**Purpose**: Computes a SHA-256 hash of the entire CSV file. This hash is used as a cache key for `load_data()` and `load_model_and_embeddings()`. If the CSV changes (rows added, edited, deleted), the hash changes, Streamlit's cache invalidates, and data + embeddings are recomputed automatically.

**Cost**: Reading a ~500KB CSV and computing SHA-256 takes <1ms. This runs on every Streamlit rerun (every user interaction), which is acceptable.

```python
# Line 99: Compute hash at module load time
csv_hash = _get_csv_hash()
```

#### `load_data(csv_hash)` — Lines 102-125

```python
@st.cache_data
def load_data(csv_hash):
    data = pd.read_csv(CSV_PATH)
    data.rename(columns={...}, inplace=True)
    data["secondary_control_type"] = data["secondary_control_type"].fillna("N/A")
    data["secondary_control_description"] = data["secondary_control_description"].fillna("N/A")

    # Pre-compute lowercase columns for fast keyword lookup
    data["_risk_level_lower"] = data["risk_level"].str.lower()
    data["_primary_control_type_lower"] = data["primary_control_type"].str.lower()
    data["_risk_title_lower"] = data["risk_title"].str.lower()
    data["_risk_description_lower"] = data["risk_description"].astype(str).str.lower()

    sentences = [row_to_text(row) for _, row in data.iterrows()]
    return data, sentences
```

**Decorator**: `@st.cache_data` — Streamlit serializes the return value and caches it. Subsequent calls with the same `csv_hash` return the cached copy instantly. Cache persists across user sessions until the server restarts or the CSV changes.

**Steps**:
1. **Load CSV**: `pd.read_csv()` parses the file into a DataFrame.
2. **Rename columns**: Standardizes column names from the original Excel-style headers to snake_case for consistent programmatic access.
3. **Fill NaN**: Secondary controls may be missing in the source data. `fillna("N/A")` prevents `NaN` from appearing in LLM context.
4. **Pre-compute lowercase columns**: Creates `_risk_level_lower`, `_primary_control_type_lower`, `_risk_title_lower`, `_risk_description_lower`. These are used by `lookup_records()` for case-insensitive matching without calling `.lower()` on every comparison. Prefix `_` indicates internal/helper columns.
5. **Build sentences**: Converts each row to a text string via `row_to_text()`. These sentences are the embedding input and the LLM context.

#### `load_model_and_embeddings(csv_hash)` — Lines 131-136

```python
@st.cache_resource
def load_model_and_embeddings(csv_hash):
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE)
    embeddings = model.encode(row_sentences, convert_to_tensor=True)
    return model, embeddings
```

**Decorator**: `@st.cache_resource` — Used instead of `@st.cache_data` because the model object is not serializable. `cache_resource` stores a reference to the object in memory, shared across all sessions.

**Steps**:
1. **Load model**: Downloads from HuggingFace Hub on first run (~110MB), cached locally at `~/.cache/huggingface/`. The BGE-base model uses a 12-layer BERT architecture producing 768-dim vectors.
2. **Encode all rows**: `model.encode(row_sentences, convert_to_tensor=True)` produces a `(1650, 768)` PyTorch tensor. Each row is a 768-dimensional vector in semantic space. `convert_to_tensor=True` keeps the result as a PyTorch tensor (instead of NumPy) for efficient GPU-compatible cosine similarity computation.

**Performance**: Encoding 1,650 records takes ~5-10 seconds on CPU. This only runs once per CSV hash.

#### `build_summary(csv_hash)` — Lines 142-173

```python
@st.cache_data
def build_summary(csv_hash):
    ...
    return "\n".join(lines)
```

Pre-computes aggregate statistics (total risks, counts per risk level, per control type, most frequent titles). This summary is prepended to the LLM context for aggregate questions, giving the LLM a bird's-eye view of the dataset without needing to count records itself.

### Groq Client

#### `_init_groq_client()` — Lines 180-190

```python
def _init_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found. Please set it in your .env file.")
        st.stop()
    return Groq(api_key=api_key)

client = _init_groq_client()
```

Initializes the Groq SDK client at module load time. `st.stop()` halts the Streamlit app if the key is missing — no point continuing without LLM access.

### Core Retrieval Logic

#### `is_aggregate_question(question)` — Lines 197-213

```python
def is_aggregate_question(question):
    q_lower = question.lower()

    # Pass 1: Check if a specific risk title is mentioned
    q_words = [w for w in q_lower.split() if len(w) > 3]
    for title in df['_risk_title_lower'].dropna().unique():
        title_words = [w for w in title.split() if len(w) > 3]
        matches = sum(1 for tw in title_words
                      if any(_stem_match(tw, qw) for qw in q_words))
        if len(title_words) >= 2 and matches >= 2:
            return False
        if len(title_words) == 1 and matches == 1:
            return False

    # Pass 2: Check for aggregate keywords
    return any(kw in q_lower for kw in AGGREGATE_KEYWORDS)
```

**Algorithm — Two-Pass Aggregate Detection**:

**Pass 1 (Specificity Check)**: Before checking aggregate keywords, scan all unique risk titles. If the user's question mentions a specific title (2+ significant-word matches for multi-word titles, or 1 match for single-word titles), return `False` immediately — this is a specific query even if it contains aggregate-sounding words.

*Why this matters*: "What are **all controls** for Chemical **Spill** Response **Delay**?" contains the aggregate keyword "all controls" but the user wants specific details about a named risk. Without Pass 1, this would be misclassified as aggregate.

**Pass 2 (Keyword Check)**: If no specific title was found, check if any aggregate keyword substring appears in the question. If yes, it's an aggregate question.

**Word filtering**: `len(w) > 3` strips short words ("the", "for", "is", "and") that would cause false positive title matches.

#### `lookup_records(question)` — Lines 216-258

```python
def lookup_records(question):
    q_lower = question.lower()
    q_words = [w for w in q_lower.split() if len(w) > 3]
    matched_indices = set()
```

**Purpose**: Keyword-based record search. Complements embedding search by finding exact categorical matches that semantic search might miss.

**Step 1 — Risk Level Matching (lines 224-226)**:
```python
for level in df['_risk_level_lower'].dropna().unique():
    if level in q_lower:
        matched_indices.update(df.index[df['_risk_level_lower'] == level])
```
If the question contains "high", "medium", "low", or "critical", returns ALL records with that risk level. Uses exact substring matching against the pre-computed lowercase column.

**Step 2 — Control Type Matching (lines 228-230)**:
```python
for ctrl in df['_primary_control_type_lower'].dropna().unique():
    if ctrl in q_lower:
        matched_indices.update(df.index[df['_primary_control_type_lower'] == ctrl])
```
Same pattern for "preventive", "detective", "corrective".

**Step 3 — Risk Title Matching (lines 232-250)**:

This is the most complex matching logic with three tiers:

```python
_common_title_words = {
    "risk", "control", "management", "system", "process", "data",
    "failure", "response", "delay", "lack", "loss", "review",
    "monitoring", "compliance", "program", "plan", "report",
}
```

The `_common_title_words` set contains generic words that appear in many risk titles. These are excluded from single-word matching to prevent false positives (e.g., "risk" matching every title with "risk" in it).

**Tier 1 — Strong match (2+ words, multi-word title)**:
```python
if len(title_words) >= 2 and matching_count >= 2:
    matched_indices.update(...)
```
For titles like "Chemical Spill Response Delay" (4 significant words), at least 2 must stem-match query words. This is the primary matching strategy.

**Tier 2 — Single-word title**:
```python
elif len(title_words) == 1 and matching_count == 1:
    matched_indices.update(...)
```
Titles with only one significant word (rare) need just 1 match.

**Tier 3 — Distinctive single-word match (relaxed matching)**:
```python
elif len(title_words) >= 2 and matching_count == 1:
    matched_tw = [tw for tw in title_words
                  if any(_stem_match(tw, qw) for qw in q_words)]
    if matched_tw and matched_tw[0] not in _common_title_words:
        matched_indices.update(...)
```
If only 1 word in a multi-word title matches, allow the match ONLY if the matched word is distinctive (not in `_common_title_words`). This handles cases like:
- "total **spil** control" → matches "Chemical **Spill** Response Delay" because "spill" is distinctive.
- But "total **control** risk" would NOT match every title containing "control" because "control" is in the common words set.

**Step 4 — Description Fallback (lines 252-256)**:
```python
if not matched_indices:
    for idx, row in df.iterrows():
        desc_words = [w for w in row['_risk_description_lower'].split() if len(w) > 3]
        if sum(1 for qw in q_words
               if any(_stem_match(qw, dw) for dw in desc_words)) >= 2:
            matched_indices.add(idx)
```
Only used when no title, level, or control type matched. Scans risk descriptions for 2+ word overlap. This is a O(n * m) operation (n=rows, m=words per description) so it's the last resort.

**Return**: `df.loc[list(matched_indices)]` — a DataFrame subset of matched rows.

#### `_score_record(record_text, q_embedding)` — Lines 294-297

```python
def _score_record(record_text, q_embedding):
    rec_embedding = embedding_model.encode(record_text, convert_to_tensor=True)
    return float(util.cos_sim(q_embedding, rec_embedding)[0][0])
```

**Purpose**: Computes cosine similarity for a single record against the query. Used for keyword-only matches that weren't in the top-K embedding results and therefore don't have a pre-computed score.

**Cost**: Encoding a single record takes ~3-5ms on CPU. This is called once per keyword-only match (typically 0-5 records).

#### `_run_retrieval(query)` — Lines 300-330

```python
def _run_retrieval(query):
    # 1. Keyword lookup
    lookup_matches = lookup_records(query)
    lookup_texts = [row_to_text(row) for _, row in lookup_matches.iterrows()]

    # 2. Semantic search
    q_embedding = embedding_model.encode(query, convert_to_tensor=True)
    all_scores = util.cos_sim(q_embedding, row_embeddings)[0]
    top_results = all_scores.topk(k=min(TOP_K, len(row_sentences)))
    top_indices = top_results.indices.tolist()
    top_scores = top_results.values.tolist()
    top_score = top_scores[0]
    embedding_records = [row_sentences[i] for i in top_indices]

    # 3. Build scores map
    scores_map = {}
    for idx, sim_score in zip(top_indices, top_scores):
        scores_map[row_sentences[idx]] = round(sim_score, 2)

    # 4. Score keyword-only matches
    for text in lookup_texts:
        if text not in scores_map:
            scores_map[text] = round(_score_record(text, q_embedding), 2)

    # 5. Build source citations
    sources = []
    for idx, sim_score in zip(top_indices[:5], top_scores[:5]):
        row_num = idx + 1
        title = df.iloc[idx]['risk_title']
        sources.append((row_num, title, round(sim_score, 2)))

    return lookup_texts, embedding_records, top_score, query, sources, scores_map
```

**Algorithm — Dual Retrieval with Score Unification**:

1. **Keyword lookup**: Calls `lookup_records()` to get exact matches. Converts matched rows to text.
2. **Semantic search**: Encodes the query into a 768-dim vector, computes cosine similarity against ALL 1,650 pre-encoded record vectors in a single batched operation (`util.cos_sim`), then selects the top-K (10) highest scores using PyTorch's `topk()`.
3. **Scores map (embedding results)**: Maps each top-K record's text → its cosine score. This provides scores for records found via embedding search.
4. **Scores map (keyword-only results)**: For records found by keyword lookup but NOT in the embedding top-K, compute their cosine score individually via `_score_record()`. This ensures every record in the final context has a cosine score.
5. **Source citations**: Takes the top 5 embedding results for the collapsible "Sources from Registry" panel. Each source includes CSV row number (1-indexed), risk title, and similarity score.

**Return**: Tuple of `(lookup_texts, embedding_records, top_score, query, sources, scores_map)`.

#### `retrieve_context(question)` — Lines 333-361

```python
def retrieve_context(question):
    lookup_texts, embedding_records, top_score, query, sources, scores_map = _run_retrieval(question)

    is_agg = is_aggregate_question(query)

    # Combine: keyword results first, then embedding results (deduped)
    all_records = lookup_texts + [r for r in embedding_records if r not in lookup_texts]

    # Hard cap
    if len(all_records) > MAX_RECORDS:
        all_records = all_records[:MAX_RECORDS]
```

**Combination strategy**: Keyword results come FIRST because they are exact matches (higher precision). Embedding results are appended with deduplication (list comprehension filters out records already in `lookup_texts`). This ordering means the "best match" for non-aggregate queries will be a keyword match when available.

**Aggregate branch (lines 345-347)**:
```python
if is_agg:
    context_records = summarize_records(all_records)
    context = f"Dataset Summary:\n{data_summary}\n\nMatching Records ({record_count} found):\n{context_records}"
```
Prepends the pre-computed dataset summary (from `build_summary()`) with grouped matching records.

**Specific branch (lines 348-359)**:
```python
else:
    best_match = all_records[0] if all_records else ""
    best_score = scores_map.get(best_match, 0)
    other_records = all_records[1:] if len(all_records) > 1 else []
    context = f"BEST MATCH (Cosine Score: {best_score}):\n{best_match}"
    if other_records:
        scored_others = []
        for rec in other_records:
            rec_score = scores_map.get(rec, 0)
            scored_others.append(f"[Cosine Score: {rec_score}] {rec}")
        other_text = summarize_records(scored_others)
        context += f"\n\nOTHER RELATED RECORDS ({len(other_records)}):\n{other_text}"
```
Labels the first record as "BEST MATCH" with its cosine score. All other records are tagged with their individual cosine scores in `[Cosine Score: X.XX]` prefix format. This allows the LLM to display scores in its output.

**Return**: `(context_string, top_score, is_aggregate_bool, sources_list)`.

### LLM Generation

#### `ask_llm_stream(question, context, chat_history, feedback_context)` — Lines 261-291

```python
def ask_llm_stream(question, context, chat_history=None, feedback_context=""):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if chat_history:
        for msg in chat_history[-CHAT_HISTORY_WINDOW:]:
            messages.append(msg)

    user_content = f"{context}\n\nQuestion: {question}"
    if feedback_context:
        user_content = f"{feedback_context}\n\n{user_content}"

    messages.append({"role": "user", "content": user_content})

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
```

**Message construction order**:
1. System prompt (defines persona and output format)
2. Chat history (last 6 messages for conversational context)
3. User content: `[feedback_context] + [retrieved context] + [question]`

**Feedback injection**: When `feedback_context` is non-empty, it's prepended to the user content. The LLM sees past user ratings and comments BEFORE the current question's context, allowing it to adjust its response style based on prior feedback.

**Streaming**: `stream=True` enables server-sent events. The generator yields each token as it arrives from Groq's LPU, allowing `st.write_stream()` to display the response progressively.

**Error handling**: Catches all exceptions (network errors, rate limits, invalid responses) and yields a user-friendly error message instead of crashing.

### Feedback System

#### `save_feedback(question, response_summary, rating, comment)` — Lines 368-381

```python
def save_feedback(question, response_summary, rating, comment=""):
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
```

**Append-only writes**: Opens in `"a"` (append) mode. Never reads or modifies existing data. Thread-safe for single-process Streamlit.

**CSV schema**:
| Column | Type | Example |
|--------|------|---------|
| timestamp | ISO 8601 | 2026-03-04T14:23:45.123456 |
| question | string | "What controls for fraud?" |
| response_summary | string (max 200 chars) | "## Best Match\n**Risk Title**: Fraud..." |
| rating | "up" or "down" | "down" |
| comment | string (optional) | "Missing secondary controls" |

**Response truncation**: `response_summary[:200]` prevents the CSV from bloating with full LLM responses. 200 characters is enough for the LLM to recall what it said.

**Auto-creation**: If `feedback.csv` doesn't exist, writes the header row first.

#### `load_recent_feedback()` — Lines 384-410

```python
def load_recent_feedback():
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
```

**Purpose**: Reads the most recent `FEEDBACK_LOOKBACK` (10) entries from the feedback CSV and formats them as a text block for the LLM.

**Logic**:
1. Return empty string if no feedback file exists (first-time users).
2. Read with pandas. Catch any parse errors gracefully.
3. Take the last 10 rows (`tail(FEEDBACK_LOOKBACK)`).
4. Format each entry with the original question, a thumbs emoji for the rating, and any comment.
5. The header line "USER FEEDBACK ON PAST ANSWERS (use this to improve your response):" instructs the LLM to consider this feedback.

**NaN handling**: `str(row.get("comment", "")).strip()` converts NaN to "nan" string, then the `comment != "nan"` check filters it out.

---

## Module: app.py

The Streamlit UI layer — ~179 lines handling page config, sidebar, chat rendering, feedback UI, and the main retrieval + generation loop.

### Page Configuration and Imports

```python
# Lines 1-14
import streamlit as st

from config import (
    EXAMPLE_QUESTIONS, CHAT_FONT_SIZE_PX,
    APP_TITLE, APP_ICON, APP_DESCRIPTION,
)

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=f":{APP_ICON}:",
    layout="wide",
)
```

**Critical ordering**: `st.set_page_config()` MUST be the first Streamlit command. Any `@st.cache_data` or `@st.cache_resource` decorator in imported modules (like `rag.py`) triggers Streamlit internally. That's why `config.py` (no Streamlit usage) is imported first, `set_page_config()` is called, and THEN `rag.py` is imported.

```python
# Lines 16-19
from rag import (  # noqa: E402
    retrieve_context, ask_llm_stream, confidence_label,
    save_feedback, load_recent_feedback,
)
```
`# noqa: E402` suppresses the PEP 8 "module level import not at top of file" warning.

### Styling

```python
# Lines 25-35
st.markdown(f"""
<style>
    .stChatMessage p, .stChatMessage li, .stChatMessage td {{
        font-size: {CHAT_FONT_SIZE_PX}px !important;
    }}
    .stChatMessage {{
        border-bottom: 1px solid rgba(128,128,128,0.1);
        padding-bottom: 1rem;
    }}
</style>
""", unsafe_allow_html=True)
```

Injects custom CSS:
- **Font size**: Sets chat message text to 17px (configurable via `CHAT_FONT_SIZE_PX`).
- **Message separator**: Adds a subtle bottom border between chat messages for visual clarity.
- **`!important`**: Overrides Streamlit's default styles.
- **Double braces `{{}}`**: Escaped for f-string interpolation within CSS.

### Sidebar

```python
# Lines 41-54
with st.sidebar:
    st.header(f":{APP_ICON}: {APP_TITLE}")
    st.markdown(APP_DESCRIPTION)
    st.divider()
    st.subheader("Example Questions")
    for example in EXAMPLE_QUESTIONS:
        if st.button(example, key=f"sidebar_{example}", use_container_width=True):
            st.session_state.pending_question = example
            st.rerun()
    st.divider()
    st.caption("Powered by Llama 3.3 70B via Groq")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
```

**Example question buttons**: Each button uses `use_container_width=True` to span the full sidebar width. On click, sets `st.session_state.pending_question` and triggers a rerun. The main chat loop picks this up and processes it as if the user typed it.

**Clear Chat**: Resets `st.session_state.messages` to an empty list, clearing all conversation history and feedback states.

### Feedback UI

```python
# Lines 60-97
def _render_feedback(msg_index, message):
    feedback_key = f"feedback_{msg_index}"

    # Already submitted — show confirmation
    if feedback_key in st.session_state:
        saved = st.session_state[feedback_key]
        icon = "👍" if saved == "up" else "👎"
        comment = st.session_state.get(f"saved_comment_{msg_index}", "")
        label = f"Feedback recorded: {icon}"
        if comment:
            label += f' — "{comment}"'
        st.caption(label)
        return

    # Not yet submitted — show all three elements at once
    col1, col2, col3 = st.columns([1, 1, 6])
    with col1:
        thumb_up = st.button("👍", key=f"up_{msg_index}")
    with col2:
        thumb_down = st.button("👎", key=f"down_{msg_index}")

    comment = st.text_input(
        "Additional feedback (optional)",
        key=f"comment_input_{msg_index}",
        placeholder="e.g. Missing details, incorrect info, great answer...",
    )

    if thumb_up or thumb_down:
        rating = "up" if thumb_up else "down"
        save_feedback(
            question=message.get("question", ""),
            response_summary=message["content"],
            rating=rating,
            comment=comment.strip(),
        )
        st.session_state[feedback_key] = rating
        if comment.strip():
            st.session_state[f"saved_comment_{msg_index}"] = comment.strip()
        st.rerun()
```

**UI layout**: Three elements visible simultaneously:
1. 👍 button (column 1)
2. 👎 button (column 2)
3. Optional comment text input (spans remaining width)

**Submission flow**:
1. User optionally types a comment.
2. User clicks either 👍 or 👎.
3. `save_feedback()` writes to CSV with the rating, question, response summary, and comment.
4. Session state is updated to track that this message has been rated.
5. `st.rerun()` refreshes the UI to show the confirmation caption.

**Double-submission prevention**: `feedback_key` in `st.session_state` acts as a guard. Once set, the function renders a read-only caption instead of buttons.

**Column ratio `[1, 1, 6]`**: Gives thumbs buttons narrow columns and the comment input a wide column.

### Source Citations

```python
# Lines 103-109
def _render_sources(msg_index, message):
    sources = message.get("sources", [])
    if sources:
        with st.expander("Sources from Registry"):
            for rank, (row_num, title, sim_score) in enumerate(sources, 1):
                st.markdown(f"**#{rank}** | Row {row_num} | {title} | Score: {sim_score}")
```

Renders a collapsible panel with the top 5 source records. Each entry shows rank, CSV row number (for manual verification), risk title, and cosine similarity score.

### Main Chat Loop

```python
# Lines 115-178
st.title(APP_TITLE)
st.caption("Ask questions about risks, controls, and mitigations...")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.info("Get started by typing a question below or selecting an example from the sidebar.")
```

**Session state initialization**: `st.session_state.messages` is a list of message dictionaries. Each message has `role` ("user" or "assistant"), `content` (text), and optionally `question`, `sources`, and `score` for assistant messages.

**History rendering (lines 125-134)**:
```python
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if "score" in message:
                label, color = confidence_label(message["score"])
                st.markdown(f"**Relevance:** :{color}[{label}] ({message['score']:.2f})")
            _render_sources(i, message)
            _render_feedback(i, message)
```

On every rerun, all past messages are re-rendered from `st.session_state.messages`. Each assistant message shows: content → relevance badge → source citations → feedback buttons. The `msg_index` (`i`) ensures unique Streamlit widget keys.

**New question handling (lines 143-178)**:
```python
user_question = st.chat_input("Ask a question about a risk or control...")

if "pending_question" in st.session_state:
    user_question = st.session_state.pending_question
    del st.session_state.pending_question

if user_question:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    # Generate assistant response
    with st.chat_message("assistant"):
        context, score, is_agg, sources = retrieve_context(user_question)

        label, color = confidence_label(score)
        st.markdown(f"**Relevance:** :{color}[{label}] ({score:.2f})")

        chat_history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]

        feedback_context = load_recent_feedback()

        response_text = st.write_stream(
            ask_llm_stream(user_question, context, chat_history, feedback_context)
        )

        if sources:
            with st.expander("Sources from Registry"):
                for rank, (row_num, title, sim_score) in enumerate(sources, 1):
                    st.markdown(f"**#{rank}** | Row {row_num} | {title} | Score: {sim_score}")

    # Persist to session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "question": user_question,
        "sources": sources,
        "score": score,
    })
    st.rerun()
```

**Execution flow**:
1. `st.chat_input()` returns the user's typed question (or `None`).
2. Sidebar example buttons set `pending_question`, which overrides the chat input.
3. User message is appended to session state.
4. `retrieve_context()` runs the full RAG pipeline (keyword + embedding search, context building).
5. Relevance badge is shown immediately (before LLM response).
6. Chat history is extracted from session state for conversational context.
7. `load_recent_feedback()` fetches the last 10 feedback entries.
8. `st.write_stream()` renders the LLM response token-by-token as it streams from Groq.
9. Source citations are rendered in a collapsible panel.
10. The assistant message is persisted to session state with `question`, `sources`, and `score` metadata.
11. `st.rerun()` forces a re-render so the new message appears in the history loop with feedback buttons.

**Why `st.rerun()`?**: Without it, the newly streamed response wouldn't have feedback buttons (they're only rendered in the history loop). The rerun re-renders everything from session state, including the feedback UI for the new message.

---

## Feedback Loop — Full Lifecycle

This section provides a detailed walkthrough of how user feedback flows through the system, from collection to storage to injection into the LLM prompt.

### How Feedback Is Collected

Every assistant message in the chat history displays three feedback elements simultaneously:

1. **👍 Thumbs Up Button**: Indicates a positive rating.
2. **👎 Thumbs Down Button**: Indicates a negative rating.
3. **Optional Comment Text Input**: A free-text field where users can type specific feedback (e.g., "Missing secondary controls", "Wrong risk matched", "Great answer").

The UI is implemented in `_render_feedback()` in `app.py` (lines 60-97). All three elements are visible at the same time — the user does NOT need to click a thumb first to see the comment box.

**Submission**: The user optionally types a comment, then clicks either 👍 or 👎 to submit. The thumb click triggers submission of both the rating AND whatever comment text is in the input field at that moment.

**One-time submission**: Each message can only be rated once. After submission, the buttons are replaced with a read-only confirmation caption showing the rating emoji and comment (if any).

### How Feedback Is Stored

When the user clicks a thumb button, `save_feedback()` in `rag.py` (lines 368-381) is called with four arguments:

| Argument | Source | Example |
|----------|--------|---------|
| `question` | The original user question that produced this response | "What controls for fraud?" |
| `response_summary` | First 200 characters of the LLM response | "## Best Match\n**Risk Title**: Fraud..." |
| `rating` | "up" or "down" based on which thumb was clicked | "down" |
| `comment` | Text from the optional comment input (empty string if blank) | "Missing secondary controls" |

The function appends a row to `feedback.csv` with these fields plus an ISO 8601 timestamp:

```csv
timestamp,question,response_summary,rating,comment
2026-03-04T14:23:45.123456,"What controls for fraud?","## Best Match...","down","Missing secondary controls"
2026-03-04T14:25:12.654321,"How many high risks?","There are 423...","up",""
```

Key behaviors:
- **Append-only**: The file is opened in append mode (`"a"`). No existing data is ever modified or deleted.
- **Auto-creation**: If `feedback.csv` doesn't exist, the header row is written first.
- **Truncation**: Response summaries are truncated to 200 characters to keep the CSV manageable.
- **Encoding**: UTF-8 encoding for international characters.
- **Gitignored**: `feedback.csv` is listed in `.gitignore` to prevent committing user feedback data.

### How Feedback Is Loaded

On every new user question, `load_recent_feedback()` in `rag.py` (lines 384-410) is called. It:

1. Checks if `feedback.csv` exists. If not, returns an empty string (no feedback yet).
2. Reads the CSV into a pandas DataFrame.
3. Takes the last `FEEDBACK_LOOKBACK` (10) rows using `fb_df.tail(10)`.
4. Formats each row into a human-readable line:
   ```
   - Q: "What controls for fraud?" — Rated: 👎 — Comment: "Missing secondary controls"
   - Q: "How many high risks?" — Rated: 👍
   ```
5. Prepends a header: `USER FEEDBACK ON PAST ANSWERS (use this to improve your response):`
6. Returns the full text block as a single string.

**Both positive and negative ratings are included**. The LLM sees the full picture — what users liked and disliked — allowing it to reinforce good patterns and avoid bad ones.

**Comments are included when present**. Empty comments are omitted (no ` — Comment: ""` suffix).

### How Feedback Is Injected Into the LLM

In `app.py` (line 159), the feedback context is loaded:
```python
feedback_context = load_recent_feedback()
```

Then in `ask_llm_stream()` (rag.py, lines 261-291), the feedback is prepended to the user content:
```python
user_content = f"{context}\n\nQuestion: {question}"
if feedback_context:
    user_content = f"{feedback_context}\n\n{user_content}"
```

The final message sent to the LLM looks like:

```
[System Prompt]
[Chat History (last 6 messages)]
[User Message]:
  USER FEEDBACK ON PAST ANSWERS (use this to improve your response):
  - Q: "What controls for fraud?" — Rated: 👎 — Comment: "Missing secondary controls"
  - Q: "How many high risks?" — Rated: 👍

  BEST MATCH (Cosine Score: 0.72):
  Risk: Fraud Detection Failure. Description: ...

  OTHER RELATED RECORDS (5):
  [Cosine Score: 0.65] Risk: ...

  Question: What are the controls for fraud risk?
```

### How Feedback Influences Future Answers

The feedback text is part of the LLM's input context, positioned BEFORE the retrieved records and question. This positioning is intentional — the LLM reads feedback first, priming its response generation. The effects are:

1. **Negative ratings with comments**: If a user rated a fraud-related answer 👎 with "Missing secondary controls", the LLM will prioritize including secondary control details in future fraud-related answers.

2. **Positive ratings**: Reinforce the response style. If aggregate summaries consistently get 👍, the LLM learns that users prefer that format.

3. **Pattern recognition**: If multiple recent answers about the same topic received negative feedback, the LLM adjusts its approach for similar future questions.

4. **Comment-driven improvement**: Specific comments like "Wrong risk matched" or "Too verbose" directly inform the LLM about quality issues. The LLM interprets these as instructions and adjusts accordingly.

5. **Recency bias**: Only the last 10 entries are included, so the LLM's behavior reflects recent user satisfaction rather than historical patterns. Older feedback naturally ages out.

**Important**: The feedback does NOT modify the retrieval pipeline. It only affects the LLM's generation. The same records are retrieved regardless of feedback — but the LLM may present them differently (more detail, better formatting, different emphasis) based on past feedback.

### Feedback Data Flow Diagram

```
User clicks 👍/👎          User types comment (optional)
      |                              |
      v                              v
  _render_feedback() in app.py ------+
      |
      v
  save_feedback() in rag.py
      |
      v
  feedback.csv (append row)
      |
      | (on next user question)
      v
  load_recent_feedback() in rag.py
      |
      v
  Format as text block:
  "USER FEEDBACK ON PAST ANSWERS..."
      |
      v
  ask_llm_stream() in rag.py
      |
      v
  Prepended to user content:
  [feedback] + [context] + [question]
      |
      v
  Groq LLM (Llama 3.3 70B)
      |
      v
  LLM generates response influenced
  by past feedback patterns
```

---

## Algorithms In Depth

### Cosine Similarity

**Used in**: `_run_retrieval()` (lines 307-311)

**Formula**:
```
similarity(A, B) = (A · B) / (||A|| × ||B||)
```

Where `A` and `B` are 768-dimensional vectors. The result ranges from -1 (opposite meaning) to 1 (identical meaning), though in practice with BGE embeddings, scores range from ~0.2 (unrelated) to ~0.9 (near-identical).

**Implementation**: `util.cos_sim(q_embedding, row_embeddings)` computes similarity between the query vector and ALL 1,650 record vectors in a single batched matrix multiplication. This is an O(n × d) operation where n=1650 and d=768, taking ~1-2ms on CPU thanks to PyTorch's optimized BLAS backend.

**Top-K selection**: `all_scores.topk(k=10)` uses PyTorch's partial sort (based on `torch.kthvalue`), which is O(n + k log k) — more efficient than sorting all 1,650 scores.

### Sentence Embeddings (BAAI/bge-base-en-v1.5)

**Used in**: `load_model_and_embeddings()`, `_run_retrieval()`, `_score_record()`

**Architecture**: 12-layer BERT with 768 hidden dimensions. Takes text input, tokenizes it, passes through transformer layers, and mean-pools the final layer to produce a single 768-dim vector.

**Training**: Trained on large-scale text pairs (queries ↔ relevant passages) using contrastive learning. Semantically similar texts are pulled closer in vector space; dissimilar texts are pushed apart.

**Why BGE-base?**: Chosen for its balance of quality (top-ranked on MTEB retrieval benchmark) and size (~110MB, fits in memory on any machine). BGE-large (1024-dim) is more accurate but ~3x slower and uses more memory.

### Two-Layer Fuzzy Matching (Prefix + Damerau-Levenshtein)

**Used in**: `lookup_records()`, `is_aggregate_question()`

See the detailed explanation in the `_stem_match()` and `_damerau_levenshtein()` sections above. Key properties:

- No external NLP library required
- **Layer 1 (prefix)**: Handles inflections (e.g., "spilling" → "spill", "preventive" → "prevent") and prefix typos (e.g., "fraus" → "fraud")
- **Layer 2 (Damerau-Levenshtein)**: Catches transposition typos where letters are swapped (e.g., "farud" → "fraud" = DL distance 1) and substitution typos (e.g., "enviroment" → "environment" = DL distance 1)
- Length-difference guard (±2 chars) prevents matching unrelated words of very different lengths
- Distance thresholds (≤1 for short words, ≤2 for long words) prevent false positives between similar-length but unrelated words (e.g., "grant" vs "plant" = DL distance 2 but max allowed = 1 for 5-char words)

### Two-Pass Aggregate Detection

**Used in**: `is_aggregate_question()`

**Problem**: "What are all controls for Chemical Spill?" contains the aggregate keyword "all controls" but is actually a specific query about a named risk.

**Solution**: Check for specific risk title mentions FIRST. If found, always return `False` (specific query), regardless of aggregate keywords. Only check aggregate keywords if no specific title is mentioned.

### Distinctive Word Matching

**Used in**: `lookup_records()` tier 3 matching

**Problem**: "total spil control" has only 1 stem-matching word ("spil" → "spill") in the 4-word title "Chemical Spill Response Delay". The standard 2-word minimum would miss this.

**Solution**: Allow 1-word matches if the matched word is "distinctive" — i.e., not in the `_common_title_words` set. Words like "risk", "control", "management" appear in dozens of titles and would cause false positives. But "spill", "fraud", "chemical", "hazardous" are distinctive enough to identify a specific risk.

### Record Summarization

**Used in**: `summarize_records()`

**Problem**: Aggregate queries can match 400+ records. Sending all of them would exceed the LLM's context window (1500 max tokens for the response, but the input context also has practical limits).

**Solution**: When records exceed the threshold (15):
1. Group by risk title and count occurrences
2. Sort groups by count (descending)
3. Include a sample of 15 individual records
4. This gives the LLM both the big picture (title counts) and specific details (sample records)

### SHA-256 CSV Change Detection

**Used in**: `_get_csv_hash()`

**Problem**: If someone updates the CSV file (adds/removes/edits rows), the cached DataFrame and embeddings become stale.

**Solution**: Compute SHA-256 of the entire CSV file on every Streamlit rerun. Pass the hash as a cache key to `load_data()` and `load_model_and_embeddings()`. If the hash changes, Streamlit invalidates both caches and recomputes everything automatically.

**Cost**: ~0.5ms for a 500KB file. Negligible compared to the rest of the pipeline.

---

## Data Flow — End to End

```
1. User types question in st.chat_input() or clicks sidebar example
       |
2. app.py adds user message to st.session_state.messages
       |
3. app.py calls retrieve_context(question) in rag.py
       |
4. rag.py → _run_retrieval(question)
       |
       ├── lookup_records(question)
       |     ├── Scan risk levels (exact substring)
       |     ├── Scan control types (exact substring)
       |     ├── Scan risk titles (stem match, 3 tiers)
       |     └── Scan descriptions (stem match, fallback)
       |
       ├── Encode question → 768-dim vector
       ├── Cosine similarity vs 1650 row vectors
       ├── Top-K selection (k=10)
       ├── Build scores_map (embedding + keyword scores)
       └── Build sources list (top 5 for citations)
       |
5. rag.py → retrieve_context() combines results
       ├── Keyword-first ordering
       ├── Deduplication
       ├── MAX_RECORDS cap (50)
       ├── Aggregate → dataset summary + grouped records
       └── Specific → BEST MATCH + OTHER RECORDS with scores
       |
6. app.py calls load_recent_feedback() in rag.py
       └── Reads last 10 entries from feedback.csv
       |
7. app.py calls ask_llm_stream() in rag.py
       ├── System prompt
       ├── Chat history (last 6 messages)
       ├── feedback_context + context + question
       └── Streaming via Groq API
       |
8. app.py renders with st.write_stream()
       ├── Relevance badge (High/Moderate/Low)
       ├── Streaming LLM response
       ├── Source citations (collapsible)
       └── Feedback buttons (👍 👎 + comment)
       |
9. app.py persists assistant message to session state
       └── Includes question, sources, score metadata
       |
10. st.rerun() → re-renders everything from session state
```

---

## Caching Strategy

| Cache | Decorator | Keyed By | Stores | Invalidated When |
|-------|-----------|----------|--------|------------------|
| DataFrame + sentences | `@st.cache_data` | `csv_hash` | Serialized DataFrame copy | CSV file changes |
| Model + embeddings | `@st.cache_resource` | `csv_hash` | In-memory reference (shared) | CSV file changes |
| Dataset summary | `@st.cache_data` | `csv_hash` | Serialized string | CSV file changes |

**`@st.cache_data` vs `@st.cache_resource`**:
- `cache_data`: Serializes return value, returns a new copy to each caller. Safe for mutable data (DataFrames).
- `cache_resource`: Stores a single reference in memory, shared across all sessions. Used for the model (not serializable) and embeddings (large tensor, sharing saves memory).

---

## Error Handling

| Scenario | Handled By | Behavior |
|----------|-----------|----------|
| Missing GROQ_API_KEY | `_init_groq_client()` | Shows `st.error()` and calls `st.stop()` |
| Groq API error (rate limit, network) | `ask_llm_stream()` try/except | Yields error message as response text |
| Corrupted feedback.csv | `load_recent_feedback()` try/except | Returns empty string, LLM runs without feedback |
| Missing feedback.csv | `load_recent_feedback()` os.path.exists check | Returns empty string |
| Missing secondary controls in CSV | `load_data()` fillna("N/A") | Replaced with "N/A" string |
| Empty search results | `retrieve_context()` | `best_match = ""`, LLM receives empty context |

---

## Session State Management

| Key | Type | Purpose |
|-----|------|---------|
| `messages` | `list[dict]` | Full chat history (user + assistant messages with metadata) |
| `pending_question` | `str` | Temporary: holds sidebar example question until processed |
| `feedback_{i}` | `"up"` or `"down"` | Tracks whether message at index `i` has been rated |
| `saved_comment_{i}` | `str` | Stores the comment text for rated message at index `i` |
| `comment_input_{i}` | `str` | Streamlit widget state for the comment text input |
| `up_{i}`, `down_{i}` | `bool` | Streamlit widget state for thumb buttons |

All session state is per-browser-tab. Refreshing the page or opening a new tab clears everything. There is no server-side persistence for chat history (feedback.csv is the only persistent storage).
