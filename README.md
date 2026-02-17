# EPA Risk Assessment Assistant

A Streamlit-powered chatbot that lets users ask natural language questions about the EPA Risk and Control Registry. It uses a RAG (Retrieval-Augmented Generation) pipeline that combines semantic search with keyword lookup to find the most relevant risk records, then passes them to an LLM for a structured, streaming answer.

## Features

- **Chat Interface**: Full conversational UI with chat history so users can ask follow-up questions
- **Spell Correction**: Automatically corrects typos (e.g., "hazadorus" -> "hazardous") using fuzzy matching against the dataset vocabulary
- **Dual Search Strategy**: Combines keyword-based lookup with embedding-based semantic search for better recall
- **Smart Aggregate Detection**: Two-pass system that distinguishes between specific risk queries and analytical questions
- **Streaming Responses**: LLM answers stream in real-time as tokens are generated
- **Confidence Scoring**: Color-coded relevance labels (High/Moderate/Low) based on cosine similarity
- **Example Questions**: Clickable example buttons for new users to get started quickly
- **Large Result Summarization**: Groups duplicate risk titles and shows counts when result sets are large

## How It Works

1. **Data Loading**: Reads the EPA Risk and Control Registry CSV file (1,650 records), standardizes column names, and pre-computes lowercase columns for fast keyword lookups. All data is cached with `@st.cache_data`.

2. **Spell Correction**: Builds a vocabulary from all risk titles, descriptions, levels, and control types. When a user submits a question, each word is compared against the vocabulary using `difflib.get_close_matches` (cutoff 0.8) to fix typos before searching.

3. **Embedding Generation**: Each risk record is converted into a sentence and encoded into a 384-dimensional vector using the `all-MiniLM-L6-v2` sentence transformer model. Cached in memory with `@st.cache_resource`.

4. **User Query Processing**: When a user asks a question, the app runs two parallel search strategies:
   - **Keyword Lookup**: Searches the dataframe for matching risk levels, control types, and title keywords. Title matching requires 2+ significant word matches for multi-word titles to reduce noise.
   - **Semantic Search**: Encodes the question into a vector and finds the top 10 most similar records using cosine similarity.

5. **Aggregate Detection**: Uses a two-pass approach. First checks if the question targets a specific risk (by matching risk titles). Only falls back to aggregate mode if no specific risk is identified AND aggregate keywords are present. This prevents "what are all the controls for spill risk?" from being treated as an aggregate question.

6. **Record Summarization**: When more than 15 matching records are found, groups them by risk title with counts and includes a sample of individual records. This prevents token overflow in the LLM context.

7. **LLM Streaming Response**: The retrieved records (with best match labeled separately) and any chat history (last 3 exchanges) are sent to Llama 3.3 70B via Groq with streaming enabled. Tokens are displayed in real-time using `st.write_stream`.

## Dataset

The app uses `Epa risk and control registry.csv` which contains 1,650 risk records with the following columns:

| Column | Description |
|--------|-------------|
| Risk Title | Name of the risk (e.g., "Hazardous Waste Mishandling") |
| Risk Statement/Description | Detailed description of what the risk entails |
| Risk Level (Inherent) | Severity level: Critical, High, Medium, or Low |
| Primary Control Type | Type of primary control: Preventive, Detective, or Corrective |
| Primary Control Description | Full description of the primary control measure |
| Secondary Control Type | Type of secondary control: Preventive, Detective, or Corrective |
| Secondary Control Description | Full description of the secondary control measure |

## Tech Stack

| Technology | Purpose |
|------------|---------|
| Streamlit | Web UI framework with chat interface |
| Sentence Transformers | Embedding model (`all-MiniLM-L6-v2`) for semantic search |
| Groq | LLM API provider for fast streaming inference |
| Llama 3.3 70B | Large language model for generating answers |
| Pandas | Data loading and manipulation |
| PyTorch | Backend for sentence transformer model |
| difflib | Standard library fuzzy matching for spell correction |

## Project Structure

```
risk_assessment_bot/
  app.py                              Main application file (Streamlit + RAG pipeline)
  Epa risk and control registry.csv   EPA risk and control dataset (1,650 records)
  requirements.txt                    Python dependencies
  .env                                Environment variables (Groq API key)
  README.md                           Project documentation
```

## Setup and Installation

### Prerequisites

- Python 3.10+
- A Groq API key (free at https://console.groq.com)

### Steps

1. Clone the repository:

```bash
git clone <repository-url>
cd risk_assessment_bot
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file and add your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

4. Run the application:

```bash
streamlit run app.py
```

5. Open your browser to `http://localhost:8501`

### Deploying to Streamlit Cloud

1. Push your repo to GitHub (make sure `.env` is in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) and deploy the repo
3. In app settings, go to **Secrets** and add:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

## Example Questions

### Specific Risk Lookups
- "What is the Unliquidated Obligations risk?"
- "Tell me about Hazardous Waste Mishandling"
- "What controls are in place for Chemical Spill Response Delay?"

### Follow-up Questions (uses chat history)
- "Tell me more about the secondary control"
- "What about the preventive controls for that?"
- "Are there similar risks?"

### Filter by Risk Level
- "Show me all high risks"
- "What are the critical risks?"

### Filter by Control Type
- "Which risks have preventive controls?"
- "List risks with detective controls"

### Aggregate / Analytical
- "How many risks are in the registry?"
- "Give me a breakdown of risks by level"
- "What are the most common control types?"

### Semantic / Natural Language (with typo tolerance)
- "What risks are related to environmental damage?"
- "Are there any risks about vendor management?"
- "How to handle hazadorus materials?" (auto-corrects to "hazardous")

## Output Format

For specific risk questions, the assistant responds with:

**Best Match**: The single most relevant record showing risk title, risk level, primary control (type + description), and secondary control (type + description).

**Other Controls to Consider**: Additional related records with their control details for a broader view.

For aggregate questions, the assistant provides counts, breakdowns, and summaries from the full dataset.

## Architecture Diagram

```
User Question
      |
      v
+---------------------+
|   Spell Correction   |
|   (difflib fuzzy     |
|    matching)         |
+---------------------+
      |
      v
+---------------------+
|  Aggregate Detection |
|  (two-pass: specific |
|   risk check first)  |
+---------------------+
      |
      v
+---------------------+     +----------------------+
|  Keyword Lookup     |     |  Semantic Search     |
|  (pre-computed      |     |  (cosine similarity  |
|   lowercase cols,   |     |   using MiniLM-L6    |
|   2+ word matching) |     |   embeddings)        |
+---------------------+     +----------------------+
      |                           |
      +--------+------------------+
               |
               v
     +-------------------+
     | Combine, Dedupe   |
     | & Summarize       |
     | (Best Match +     |
     |  Other Records)   |
     +-------------------+
               |
               v
     +-------------------+
     | Groq LLM          |
     | (Llama 3.3 70B)   |
     | + Chat History     |
     | + Streaming        |
     +-------------------+
               |
               v
     Streaming Answer
     with Confidence Label
```

## Configuration

| Setting | Value | Location |
|---------|-------|----------|
| Embedding Model | all-MiniLM-L6-v2 | `app.py` |
| LLM Model | llama-3.3-70b-versatile | `app.py` |
| Top K Results | 10 | `app.py` |
| LLM Temperature | 0.1 | `app.py` |
| Max Tokens | 1500 | `app.py` |
| Max Records to LLM | 50 | `app.py` |
| Spell Correction Cutoff | 0.8 | `app.py` |
| Chat History Window | Last 3 exchanges | `app.py` |
| Record Summarization Threshold | 15 records | `app.py` |
