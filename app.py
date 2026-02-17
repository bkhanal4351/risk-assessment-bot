import os
import difflib
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# ROW TO TEXT CONVERTER
# Takes a single DataFrame row and converts it into a plain-text sentence.
# This sentence format is used for two purposes:
#   1. It gets encoded into a vector by the embedding model for semantic search.
#   2. It gets passed as context to the LLM so it can read the risk details.
# =============================================================================
def row_to_text(row):
    return (
        f"Risk: {row['risk_title']}. "
        f"Description: {row['risk_description']}. "
        f"Risk Level: {row['risk_level']}. "
        f"Primary Control ({row['primary_control_type']}): {row['primary_control_description']}. "
        f"Secondary Control ({row['secondary_control_type']}): {row['secondary_control_description']}."
    )


# =============================================================================
# DATA LOADING (CACHED)
# Reads the EPA Risk and Control Registry CSV file and standardizes all column
# names to lowercase with underscores for consistent access throughout the app.
# Fills NaN values in optional secondary control columns with "N/A" so text
# formatting does not break. Pre-computes lowercase columns for faster keyword
# lookups (avoids calling .str.lower() on every query). Also pre-computes the
# text sentences for every row.
# Decorated with @st.cache_data so this only runs once, not on every rerun.
# =============================================================================
@st.cache_data
def load_data():
    df = pd.read_csv("Epa risk and control registry.csv")
    df.rename(columns={
        "Risk Title": "risk_title",
        "Risk Statement/Description": "risk_description",
        "Risk Level (Inherent)": "risk_level",
        "Primary Control Type": "primary_control_type",
        "Primary Control Description": "primary_control_description",
        "Secondary Control Type": "secondary_control_type",
        "Secondary Control Description": "secondary_control_description"
    }, inplace=True)
    df["secondary_control_type"] = df["secondary_control_type"].fillna("N/A")
    df["secondary_control_description"] = df["secondary_control_description"].fillna("N/A")

    # Pre-compute lowercase columns for faster keyword lookups
    df["_risk_level_lower"] = df["risk_level"].str.lower()
    df["_primary_control_type_lower"] = df["primary_control_type"].str.lower()
    df["_risk_title_lower"] = df["risk_title"].str.lower()
    df["_risk_description_lower"] = df["risk_description"].astype(str).str.lower()

    sentences = [row_to_text(row) for _, row in df.iterrows()]
    return df, sentences


df, row_sentences = load_data()


# =============================================================================
# SPELL CORRECTION VOCABULARY (CACHED)
# Builds a vocabulary of known terms from the dataset (risk titles, levels,
# control types, descriptions) for fuzzy matching against user queries. Uses
# difflib.get_close_matches to correct typos like "hazadorus" -> "hazardous".
# Only corrects words 4+ characters with a close match (cutoff 0.8) so short
# words and correctly spelled words are left untouched.
# =============================================================================
@st.cache_data
def build_vocabulary():
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


vocabulary = build_vocabulary()


def correct_query(question):
    """Corrects misspelled words in the user's question by matching against
    known terms from the dataset. Returns the corrected query string."""
    words = question.split()
    corrected = []
    for word in words:
        if len(word) <= 3:
            corrected.append(word)
            continue
        clean = word.lower().strip("?.,!;:")
        matches = difflib.get_close_matches(clean, vocabulary, n=1, cutoff=0.8)
        if matches and matches[0] != clean:
            corrected.append(matches[0])
        else:
            corrected.append(word)
    return " ".join(corrected)


# =============================================================================
# EMBEDDING MODEL AND VECTOR COMPUTATION (CACHED)
# Loads the all-MiniLM-L6-v2 sentence transformer model which converts text
# into 384-dimensional vectors. Encodes every row sentence into a tensor so
# we can compute cosine similarity between user questions and risk records.
# @st.cache_resource keeps the model and embeddings in memory across reruns.
# =============================================================================
@st.cache_resource
def load_model_and_embeddings(_sentences):
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    embeddings = model.encode(list(_sentences), convert_to_tensor=True)
    return model, embeddings


embedding_model, row_embeddings = load_model_and_embeddings(tuple(row_sentences))


# =============================================================================
# DATASET SUMMARY BUILDER (CACHED)
# Pre-computes aggregate statistics about the entire dataset: total risk count,
# breakdown by risk level, breakdown by primary and secondary control types,
# and most frequent risk titles. This summary text is passed to the LLM when
# the user asks aggregate questions like "how many high risks do we have?"
# so the LLM can answer from stats without needing to see every record.
# =============================================================================
@st.cache_data
def build_summary():
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


data_summary = build_summary()


# =============================================================================
# GROQ CLIENT INITIALIZATION
# Loads the Groq API key from the .env file and initializes the client.
# If the key is missing, displays an error in the Streamlit UI and stops
# execution so the user sees a clear message instead of a crash.
# =============================================================================
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY not found. Please set it in your .env file.")
    st.stop()
client = Groq(api_key=groq_api_key)


# =============================================================================
# AGGREGATE DETECTION (TWO-PASS APPROACH)
# Instead of a flat keyword list that triggers on any match, this uses a
# two-pass approach:
#   Pass 1: Check if the question targets a specific risk by matching against
#           known risk titles in the dataset (2+ significant word overlap).
#   Pass 2: Only classify as aggregate if no specific risk is targeted AND
#           aggregate keywords are present.
# This prevents "what are all the controls for spill risk?" from being
# treated as an aggregate question when the user wants specific details.
# =============================================================================
AGGREGATE_KEYWORDS = [
    "most", "least", "how many", "count", "total", "average",
    "top", "bottom", "list all", "every", "each",
    "which level", "which type", "biggest", "smallest",
    "percentage", "breakdown", "summary", "statistics",
    "by level", "by type", "by control", "per level", "per type",
    "group by", "grouped by", "list of risks", "list risks",
    "how many risks", "high risk", "medium risk", "low risk",
    "all controls", "all risks"
]


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


# =============================================================================
# LLM SYSTEM PROMPT
# Extracted as a constant so it can be reused across streaming and non-streaming
# calls. Instructs the LLM on formatting, typo handling, and output structure.
# =============================================================================
SYSTEM_PROMPT = (
    "You are an EPA risk assessment assistant. "
    "Answer questions using ONLY the risk and control data provided. "
    "Be concise and direct. If someone uses partial terms, "
    "match them to the closest risk or control in the records. "
    "For broad questions (e.g., 'list risks by level', "
    "'how many high risks'), present the counts and "
    "breakdowns from the Dataset Summary. This IS the answer — "
    "showing risk counts per group is the correct response "
    "to these types of questions. "
    "For specific risk/control questions, structure your answer as:\n"
    "## Best Match\n"
    "Show the single most relevant record with:\n"
    "- **Risk Title**: the title\n"
    "- **Risk Level**: the level\n"
    "- **Primary Control** (type): full description\n"
    "- **Secondary Control** (type): full description\n\n"
    "## Other Controls to Consider\n"
    "Briefly list any other matching records with their "
    "primary and secondary control types and descriptions.\n\n"
    "IMPORTANT: Always include BOTH the control type AND the full "
    "control description for primary and secondary controls. "
    "Never omit secondary control details.\n"
    "Users may have typos or use informal terms — interpret their "
    "intent and match to the closest relevant records. For example, "
    "'hazadorus materials' should match 'Hazardous Waste Mishandling'. "
    "Only say 'I could not find that information in the registry' if "
    "none of the provided records are even remotely related to the question."
)


# =============================================================================
# LLM STREAMING FUNCTION
# Sends the user's question along with retrieved context to Llama 3.3 70B
# via the Groq API with streaming enabled. Includes recent chat history
# (last 3 exchanges) so the LLM can handle follow-up questions like
# "tell me more about that" or "what about the secondary control?".
# Returns a generator that yields text chunks for real-time display.
# =============================================================================
def ask_llm_stream(question, context, chat_history=None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if chat_history:
        for msg in chat_history[-6:]:
            messages.append(msg)

    messages.append({
        "role": "user",
        "content": f"{context}\n\nQuestion: {question}"
    })

    try:
        stream = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.1,
            max_tokens=1500,
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"Error communicating with Groq API: {e}"


# =============================================================================
# KEYWORD-BASED RECORD LOOKUP (OPTIMIZED)
# Searches the dataframe directly for records matching the user's question.
# This complements the embedding search by catching exact matches that
# semantic similarity might miss.
#
# Improvements over previous version:
#   - Uses pre-computed lowercase columns (no .str.lower() on every query)
#   - Title matching requires 2+ significant word matches for multi-word
#     titles (prevents "management" alone from pulling dozens of records)
#   - Single-word titles still match on one word
#   - Description fallback only triggers when no other matches found
# =============================================================================
def lookup_records(question):
    q_lower = question.lower()
    matched_indices = set()

    # Search by risk level
    for level in df['_risk_level_lower'].dropna().unique():
        if level in q_lower:
            matched_indices.update(df.index[df['_risk_level_lower'] == level])

    # Search by primary control type
    for ctrl in df['_primary_control_type_lower'].dropna().unique():
        if ctrl in q_lower:
            matched_indices.update(df.index[df['_primary_control_type_lower'] == ctrl])

    # Search by keywords in risk title (require 2+ matches for multi-word titles)
    for title in df['_risk_title_lower'].dropna().unique():
        title_words = [w for w in title.split() if len(w) > 3]
        matching_count = sum(1 for w in title_words if w in q_lower)
        if len(title_words) >= 2 and matching_count >= 2:
            matched_indices.update(df.index[df['_risk_title_lower'] == title])
        elif len(title_words) == 1 and matching_count == 1:
            matched_indices.update(df.index[df['_risk_title_lower'] == title])

    # Search by keywords in risk description (only if no matches yet)
    if not matched_indices:
        q_words = [w for w in q_lower.split() if len(w) > 3]
        for idx, row in df.iterrows():
            if sum(1 for w in q_words if w in row['_risk_description_lower']) >= 2:
                matched_indices.add(idx)

    return df.loc[list(matched_indices)]


# =============================================================================
# RECORD SET SUMMARIZER
# When the number of matching records exceeds a threshold, groups them by
# risk title and shows counts instead of listing every individual record.
# This prevents token overflow when sending context to the LLM and gives
# the LLM a cleaner, more structured view of large result sets.
# =============================================================================
def summarize_records(records_text, max_individual=15):
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


# =============================================================================
# CONFIDENCE LABEL
# Converts the raw cosine similarity score into a human-readable label with
# color coding. Thresholds are based on typical all-MiniLM-L6-v2 score
# distributions:
#   0.6+  = High relevance (strong semantic match)
#   0.4+  = Moderate relevance (related topic)
#   below = Low relevance (may not be directly related)
# =============================================================================
def confidence_label(score):
    if score >= 0.6:
        return "High", "green"
    elif score >= 0.4:
        return "Moderate", "orange"
    else:
        return "Low", "red"


# =============================================================================
# RAG PIPELINE (RETRIEVAL-AUGMENTED GENERATION)
# This is the main retrieval function that ties everything together:
#   1. Corrects spelling in the user's question using difflib
#   2. Determines if the question is aggregate (two-pass detection)
#   3. Runs keyword-based lookup on the dataframe for exact matches
#   4. Runs embedding-based semantic search for the top 10 similar records
#   5. Combines both result sets, removing duplicates
#   6. For large result sets, summarizes records by grouping titles
#   7. For aggregate questions: includes dataset summary statistics
#   8. For specific questions: labels the best match separately
#   9. Returns the context string and metadata for the streaming LLM call
# =============================================================================
TOP_K = 10


def retrieve_context(question):
    corrected = correct_query(question)
    is_agg = is_aggregate_question(corrected)

    lookup_matches = lookup_records(corrected)
    lookup_texts = [row_to_text(row) for _, row in lookup_matches.iterrows()]

    q_embedding = embedding_model.encode(corrected, convert_to_tensor=True)
    scores = util.cos_sim(q_embedding, row_embeddings)[0]
    top_indices = scores.topk(k=min(TOP_K, len(row_sentences))).indices.tolist()
    top_score = scores[top_indices[0]].item()

    embedding_records = [row_sentences[i] for i in top_indices]

    all_records = lookup_texts + [r for r in embedding_records if r not in lookup_texts]

    record_count = len(all_records)
    if record_count > 50:
        all_records = all_records[:50]

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

    return context, top_score, corrected, is_agg


# =============================================================================
# STREAMLIT UI (CHAT INTERFACE)
# Renders a full chat interface with:
#   - Chat history stored in st.session_state across reruns
#   - Example question buttons for new users (shown when chat is empty)
#   - Spell correction notifications when queries are auto-corrected
#   - Qualitative confidence labels (High/Moderate/Low) with color coding
#   - Streaming LLM responses displayed in real-time as tokens arrive
#   - st.chat_input and st.chat_message for a native chat experience
# =============================================================================
st.title("EPA Risk Assessment Assistant")
st.caption("Ask questions about risks, controls, and mitigations from the EPA Risk and Control Registry.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Example question buttons (shown only when chat history is empty)
if not st.session_state.messages:
    st.markdown("**Try an example question:**")
    examples = [
        "What controls exist for Hazardous Waste Mishandling?",
        "How many high risks are there?",
        "Tell me about Chemical Spill Response Delay",
    ]
    cols = st.columns(len(examples))
    for i, example in enumerate(examples):
        if cols[i].button(example, key=f"example_{i}"):
            st.session_state.pending_question = example
            st.rerun()

# Display all previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept input from chat box or example button click
user_question = st.chat_input("Ask a question about a risk or control...")

if "pending_question" in st.session_state:
    user_question = st.session_state.pending_question
    del st.session_state.pending_question

if user_question:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    # Retrieve context and stream response
    with st.chat_message("assistant"):
        context, score, corrected, is_agg = retrieve_context(user_question)

        # Notify if spell correction changed the query
        if corrected.lower() != user_question.lower():
            st.info(f"Showing results for: *{corrected}*")

        # Show qualitative confidence label
        label, color = confidence_label(score)
        st.markdown(f"**Relevance:** :{color}[{label}] ({score:.2f})")

        # Build chat history for conversational context
        chat_history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]

        # Stream the LLM response in real-time
        response_text = st.write_stream(
            ask_llm_stream(user_question, context, chat_history)
        )

    st.session_state.messages.append({"role": "assistant", "content": response_text})
