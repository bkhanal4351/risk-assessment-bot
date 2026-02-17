import os
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
# formatting does not break. Also pre-computes the text sentences for every row.
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
    sentences = [row_to_text(row) for _, row in df.iterrows()]
    return df, sentences


df, row_sentences = load_data()


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
# AGGREGATE KEYWORD LIST
# Keywords that indicate the user is asking a broad, analytical question
# rather than looking up a specific risk. When any of these appear in the
# user's question, the RAG pipeline includes the pre-computed dataset summary
# in the context so the LLM can provide counts, breakdowns, and statistics.
# =============================================================================
AGGREGATE_KEYWORDS = [
    "most", "least", "how many", "count", "total", "average",
    "top", "bottom", "all", "list all", "every", "each",
    "which level", "which type", "biggest", "smallest",
    "percentage", "breakdown", "summary", "statistics",
    "by level", "by type", "by control", "per level", "per type",
    "group by", "grouped by", "list of risks", "list risks",
    "how many risks", "high risk", "medium risk", "low risk",
    "all controls", "all risks"
]


# =============================================================================
# LLM QUERY FUNCTION
# Sends the user's question along with retrieved context to Llama 3.3 70B
# via the Groq API. The system prompt instructs the LLM to:
#   - Only use the provided data (no hallucination)
#   - Handle typos and informal terms by matching to closest records
#   - Format specific questions as "Best Match" + "Other Controls to Consider"
#   - Format aggregate questions with counts and breakdowns
#   - Always show both primary and secondary control types AND descriptions
# Wrapped in try/except so API errors are shown gracefully in the UI.
# =============================================================================
def ask_llm(question, context, is_aggregate=False):
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
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
                },
                {
                    "role": "user",
                    "content": f"{context}\n\nQuestion: {question}"
                }
            ],
            temperature=0.1,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error communicating with Groq API: {e}"


# =============================================================================
# KEYWORD-BASED RECORD LOOKUP
# Searches the dataframe directly for records matching the user's question.
# This complements the embedding search by catching exact matches that
# semantic similarity might miss (e.g., "show all high risks" needs every
# record with risk_level="High", not just the top 10 similar ones).
#
# Search order:
#   1. Match by risk level (High, Medium, Low, Critical)
#   2. Match by primary control type (Preventive, Detective, Corrective)
#   3. Match by significant words (4+ chars) in risk titles
#   4. If still no matches, match by keyword overlap in risk descriptions
#
# Uses a set of indices instead of pd.concat in a loop for performance.
# =============================================================================
def lookup_records(question):
    q_lower = question.lower()
    matched_indices = set()

    for level in df['risk_level'].dropna().str.lower().unique():
        if level in q_lower:
            matched_indices.update(df.index[df['risk_level'].str.lower() == level])

    for ctrl in df['primary_control_type'].dropna().str.lower().unique():
        if ctrl in q_lower:
            matched_indices.update(df.index[df['primary_control_type'].str.lower() == ctrl])

    for title in df['risk_title'].dropna().str.lower().unique():
        title_words = [w for w in title.split() if len(w) > 3]
        if any(w in q_lower for w in title_words):
            matched_indices.update(df.index[df['risk_title'].str.lower() == title])

    if not matched_indices:
        q_words = [w for w in q_lower.split() if len(w) > 3]
        for idx, row in df.iterrows():
            desc = str(row['risk_description']).lower()
            if sum(1 for w in q_words if w in desc) >= 2:
                matched_indices.add(idx)

    return df.loc[list(matched_indices)]


# =============================================================================
# RAG PIPELINE (RETRIEVAL-AUGMENTED GENERATION)
# This is the main function that ties everything together:
#   1. Checks if the question is aggregate (uses keyword detection)
#   2. Runs keyword-based lookup on the dataframe for exact matches
#   3. Runs embedding-based semantic search for the top 10 similar records
#   4. Combines both result sets, removing duplicates
#   5. For aggregate questions: includes dataset summary statistics
#   6. For specific questions: labels the best match separately
#   7. Sends the assembled context to the LLM and returns the answer
#      along with the confidence score (cosine similarity of best match)
# =============================================================================
TOP_K = 10


def get_answer(question):
    q_lower = question.lower()
    is_aggregate = any(kw in q_lower for kw in AGGREGATE_KEYWORDS)

    lookup_matches = lookup_records(question)
    lookup_texts = [row_to_text(row) for _, row in lookup_matches.iterrows()]

    q_embedding = embedding_model.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(q_embedding, row_embeddings)[0]
    top_indices = scores.topk(k=min(TOP_K, len(row_sentences))).indices.tolist()
    top_score = scores[top_indices[0]].item()

    embedding_records = [row_sentences[i] for i in top_indices]

    all_records = lookup_texts + [r for r in embedding_records if r not in lookup_texts]

    record_count = len(all_records)
    if record_count > 50:
        all_records = all_records[:50]

    if is_aggregate:
        context_records = "\n".join(all_records)
        context = f"Dataset Summary:\n{data_summary}\n\nMatching Records ({record_count} found):\n{context_records}"
    else:
        best_match = all_records[0] if all_records else ""
        other_records = all_records[1:] if len(all_records) > 1 else []
        context = f"BEST MATCH:\n{best_match}"
        if other_records:
            context += f"\n\nOTHER RELATED RECORDS ({len(other_records)}):\n" + "\n".join(other_records)

    answer = ask_llm(question, context, is_aggregate=is_aggregate)
    return answer, top_score


# =============================================================================
# STREAMLIT UI
# Renders the web interface with a title, caption, and text input field.
# When the user submits a question, it runs the RAG pipeline and displays
# the answer with a confidence score. The confidence score is the cosine
# similarity of the best matching record (higher = more relevant retrieval).
# =============================================================================
st.title("EPA Risk Assessment Assistant")
st.caption("Ask questions about risks, controls, and mitigations from the EPA Risk and Control Registry.")

user_question = st.text_input("Ask a question about a risk or control:")

if user_question:
    with st.spinner("Analyzing risk registry..."):
        answer, confidence = get_answer(user_question)

    st.write("**Confidence:**", f"{confidence:.2f}")
    st.markdown(f"**Answer:** {answer}")
