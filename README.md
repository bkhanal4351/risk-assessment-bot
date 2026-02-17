# EPA Risk Assessment Assistant

A Streamlit-powered chatbot that lets users ask natural language questions about the EPA Risk and Control Registry. It uses a RAG (Retrieval-Augmented Generation) pipeline that combines semantic search with keyword lookup to find the most relevant risk records, then passes them to an LLM for a structured, streaming answer.

## Features

- **Chat Interface**: Full conversational UI with chat history so users can ask follow-up questions
- **Spell Correction**: Automatically corrects typos (e.g., "hazadorus" -> "hazardous") using fuzzy matching against the dataset vocabulary
- **Dual Search Strategy**: Combines keyword-based lookup with embedding-based semantic search for better recall
- **Smart Aggregate Detection**: Two-pass system that distinguishes between specific risk queries and analytical questions
- **Follow-up Detection**: Detects conversational follow-ups and enriches the search with prior context, with a low-confidence fallback retry
- **Streaming Responses**: LLM answers stream in real-time as tokens are generated
- **Confidence Scoring**: Color-coded relevance labels (High/Moderate/Low) based on cosine similarity
- **Example Questions**: Clickable example buttons for new users to get started quickly
- **Large Result Summarization**: Groups duplicate risk titles and shows counts when result sets are large

## Algorithm and Approach

### RAG (Retrieval-Augmented Generation)

The app follows the RAG pattern: instead of asking the LLM to answer from its training data (which may hallucinate), it first **retrieves** relevant records from the dataset and then passes them as context to the LLM to **generate** an answer grounded in real data.

### Step-by-Step Pipeline

```
User Question
      |
      v
1. FOLLOW-UP RESOLUTION
   Checks if the question references a previous conversation turn.
   Detection triggers on:
     - Short questions (< 8 words)
     - Reference words ("that", "this", "those", "more about", etc.)
     - Low-confidence fallback: if initial retrieval scores < 0.3
       and chat history exists, auto-retries with previous question prepended
      |
      v
2. SPELL CORRECTION
   Each word (4+ chars) is compared against a vocabulary built from
   all risk titles, descriptions, levels, and control types using
   difflib.SequenceMatcher (cutoff: 0.8, length tolerance: ±2 chars).
   This catches typos without false positives:
     "hazadorus" (9 chars) -> "hazardous" (9 chars)   CORRECTED (diff 0)
     "correct"   (7 chars) -> "corrective" (10 chars)  SKIPPED   (diff 3)
      |
      v
3. AGGREGATE DETECTION (Two-Pass)
   Pass 1: Check if the question mentions a specific risk title
           (requires 2+ significant word matches). If yes -> SPECIFIC query.
   Pass 2: If no specific risk found, check for aggregate keywords
           ("how many", "list all", "breakdown", etc.) -> AGGREGATE query.
   This prevents "what are all controls for spill risk?" from being
   treated as aggregate when the user wants specific details.
      |
      v
4. DUAL RETRIEVAL
   Two search strategies run in parallel:

   a) KEYWORD LOOKUP (exact matching)
      Searches pre-computed lowercase dataframe columns for:
        - Risk levels ("high", "medium", "low", "critical")
        - Control types ("preventive", "detective", "corrective")
        - Risk title words (2+ word match required for multi-word titles)
        - Description keywords (fallback, requires 2+ word overlap)

   b) SEMANTIC SEARCH (embedding similarity)
      Encodes the question into a 384-dim vector using all-MiniLM-L6-v2
      and computes cosine similarity against all pre-computed record vectors.
      Returns top 10 most similar records.
      |
      v
5. COMBINE AND DEDUPLICATE
   Merges keyword matches (high precision) with embedding matches
   (high recall), removing duplicates. Keyword results come first
   since they are exact matches.
      |
      v
6. SUMMARIZE (if needed)
   If more than 15 records match, groups them by risk title with counts
   and includes a sample of 15 individual records. This prevents
   exceeding the LLM's context window.
      |
      v
7. LLM GENERATION (Streaming)
   Sends to Llama 3.3 70B (via Groq API) with:
     - System prompt defining output format
     - Last 3 chat exchanges for conversational context
     - "BEST MATCH" labeled separately from "OTHER RELATED RECORDS"
   Response streams token-by-token to the UI.
```

### Key Algorithms

| Algorithm | Where Used | How It Works |
|-----------|-----------|--------------|
| Cosine Similarity | Semantic search | Measures angle between question vector and record vectors. Score 1.0 = identical meaning, 0.0 = unrelated. Used to rank the top 10 most relevant records. |
| SequenceMatcher (difflib) | Spell correction | Finds the longest contiguous matching subsequence between the user's word and vocabulary words. Ratio = 2 * matches / total_chars. Threshold 0.8 means 80%+ character overlap required. |
| Sentence Embedding (all-MiniLM-L6-v2) | Vector encoding | Transformer model that maps text to 384-dimensional dense vectors where semantically similar sentences are close together. Pre-trained on 1B+ sentence pairs. |
| TF-style Keyword Matching | Keyword lookup | Exact substring matching against pre-computed lowercase columns. Fast O(n) scan with set-based index collection instead of DataFrame concatenation. |

### Why Dual Search?

Neither search strategy alone is sufficient:

- **Keyword lookup alone** misses semantically related queries. "How to handle dangerous materials?" won't match "Hazardous Waste Mishandling" because there are no overlapping keywords.
- **Embedding search alone** misses exact categorical filters. "Show all high risks" needs every record where risk_level="High", not just the 10 most semantically similar.

By combining both, the app achieves high precision (keyword) and high recall (semantic).

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
  .gitignore                          Excludes .env and cache files from git
  README.md                           Project documentation
```

## Setup and Installation

### Prerequisites

- Python 3.10+
- A Groq API key (free at https://console.groq.com)

### macOS / Linux

1. Clone the repository:

```bash
git clone https://github.com/bkhanal4351/risk-assessment-bot.git
cd risk-assessment-bot
```

2. Create and activate a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file and add your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

5. Run the application:

```bash
streamlit run app.py
```

6. Open your browser to `http://localhost:8501`

### Windows

1. Clone the repository:

```cmd
git clone https://github.com/bkhanal4351/risk-assessment-bot.git
cd risk-assessment-bot
```

2. Create and activate a virtual environment (recommended):

```cmd
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:

```cmd
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

5. Run the application:

```cmd
streamlit run app.py
```

6. Open your browser to `http://localhost:8501`

**Windows Notes:**
- If `streamlit` is not recognized, try `python -m streamlit run app.py`
- If you get a PyTorch error, install the CPU version: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- Make sure your `.env` file uses UTF-8 encoding (not UTF-16)

### Deploying to Streamlit Cloud

1. Push your repo to GitHub (make sure `.env` is in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) and deploy the repo
3. In app settings, go to **Secrets** and add:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

The app will be live at a URL like `https://your-app-name.streamlit.app`

## Example Questions

### Specific Risk Lookups

- "What is the Unliquidated Obligations risk?"
- "Tell me about Hazardous Waste Mishandling"
- "What controls are in place for Chemical Spill Response Delay?"

### Follow-up Questions (uses chat history)

- "Tell me more about the secondary control"
- "What about the preventive controls for that?"
- "Can you show me 2 of the highest ones?"

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
+------------------------+
|  Follow-up Resolution  |
|  (indicator detection  |
|   + low-score fallback)|
+------------------------+
      |
      v
+------------------------+
|   Spell Correction     |
|   (difflib fuzzy match |
|    cutoff=0.8, ±2 len) |
+------------------------+
      |
      v
+------------------------+
|  Aggregate Detection   |
|  (two-pass: specific   |
|   risk check first)    |
+------------------------+
      |
      v
+---------------------+     +----------------------+
|  Keyword Lookup     |     |  Semantic Search     |
|  (pre-computed      |     |  (cosine similarity  |
|   lowercase cols,   |     |   using MiniLM-L6    |
|   2+ word matching) |     |   384-dim vectors)   |
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
| Spell Correction Length Tolerance | ±2 chars | `app.py` |
| Low Confidence Fallback Threshold | 0.3 | `app.py` |
| Chat History Window | Last 3 exchanges (6 messages) | `app.py` |
| Record Summarization Threshold | 15 records | `app.py` |
| Follow-up Word Count Threshold | < 8 words | `app.py` |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `GROQ_API_KEY not found` | Create a `.env` file with `GROQ_API_KEY=your_key` or set it in Streamlit Cloud secrets |
| `streamlit: command not found` | Use `python -m streamlit run app.py` instead |
| PyTorch installation fails on Windows | Install CPU version: `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| App is slow on first load | The embedding model downloads on first run (~80MB). Subsequent runs use the cached model. |
| Low relevance scores on follow-up questions | The app auto-retries with context from your previous question. Try rephrasing with more specific terms. |
| `ModuleNotFoundError` | Make sure you activated the virtual environment and ran `pip install -r requirements.txt` |

## Known Limitations

- **No cross-encoder re-ranking**: The "best match" is based on cosine similarity or keyword match order, not a dedicated relevance model. Adding a cross-encoder would improve accuracy but exceed free-tier memory limits.
- **CSV-based data**: Data updates require replacing the CSV and redeploying. No admin UI for CRUD operations.
- **Spell correction scope**: Only corrects words to terms found in the dataset. General English typos (e.g., "waht" -> "what") are not corrected.
- **Context window**: Very large aggregate queries may truncate at 50 records and 1,500 tokens.
