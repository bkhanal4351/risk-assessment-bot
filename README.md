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

### Why Use a Virtual Environment?

A virtual environment (`venv`) creates an isolated Python installation for this project so its dependencies (PyTorch, Streamlit, sentence-transformers, etc.) don't conflict with packages installed globally or by other projects. Without one, installing this project's requirements could upgrade or downgrade packages that another project depends on, causing hard-to-debug breakage.

**Alternatives to `venv`:**

- **`conda`** (Anaconda/Miniconda): Full environment manager that also handles non-Python dependencies. Use `conda create -n risk_bot python=3.10` then `conda activate risk_bot`.
- **`poetry`**: Manages dependencies and virtual environments together via `pyproject.toml`. Use `poetry install` instead of `pip install -r requirements.txt`.
- **`pipenv`**: Combines `pip` and `venv` into a single workflow with a `Pipfile.lock` for reproducibility. Use `pipenv install -r requirements.txt`.
- **Docker**: Containerizes the entire app and its dependencies. Eliminates "works on my machine" issues entirely, especially useful for VDI or server deployments.

If you choose not to use any virtual environment, you can skip step 2 in the setup instructions below and install directly with `pip install -r requirements.txt` — but this is not recommended for the reasons above.

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

### Running on a VDI (Virtual Desktop Infrastructure)

If you are running on a corporate VDI or locked-down environment where you cannot deploy to Streamlit Cloud, you can run the app locally as a standalone server that other users on the same network can access.

1. Follow the **macOS / Linux** or **Windows** setup steps above to install dependencies.

2. Set the Groq API key as an environment variable (if `.env` is not supported in your VDI):

```bash
# Linux / macOS
export GROQ_API_KEY=your_groq_api_key_here

# Windows (Command Prompt)
set GROQ_API_KEY=your_groq_api_key_here

# Windows (PowerShell)
$env:GROQ_API_KEY="your_groq_api_key_here"
```

3. Start the app and bind it to all network interfaces so other VDI users can connect:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

4. Other users on the same network can access the app at `http://<your-vdi-ip>:8501`. Find your IP with:

```bash
# Linux / macOS
hostname -I

# Windows
ipconfig
```

**VDI Notes:**

- If your VDI blocks outbound internet, you will need to pre-download the embedding model and Groq API access may require a proxy. Set `HTTPS_PROXY` if needed.
- If port 8501 is blocked by firewall, try a different port: `--server.port 8080`
- For headless VDIs with no browser, use `--server.headless true` to suppress the auto-open behavior.
- If Python is not available on the VDI, ask IT to install Python 3.10+ or use a portable Python distribution.
- To keep the app running after you close your terminal, use `nohup` (Linux) or run it as a background service:

```bash
nohup streamlit run app.py --server.address 0.0.0.0 --server.port 8501 > app.log 2>&1 &
```

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

| Setting | Value | Defined In |
|---------|-------|------------|
| Embedding Model | all-MiniLM-L6-v2 | `load_model_and_embeddings()` |
| LLM Model | llama-3.3-70b-versatile | `ask_llm_stream()` |
| Top K Results | 10 | `TOP_K` constant |
| LLM Temperature | 0.1 | `ask_llm_stream()` |
| Max Tokens | 1500 | `ask_llm_stream()` |
| Max Records to LLM | 50 | `retrieve_context()` |
| Spell Correction Cutoff | 0.8 | `correct_query()` via `difflib.get_close_matches()` |
| Spell Correction Length Tolerance | ±2 chars | `correct_query()` length check |
| Low Confidence Fallback Threshold | 0.3 | `retrieve_context()` fallback condition |
| Chat History Window | Last 3 exchanges (6 messages) | `ask_llm_stream()` slicing `chat_history[-6:]` |
| Record Summarization Threshold | 15 records | `summarize_records()` `max_individual` parameter |
| Follow-up Word Count Threshold | < 8 words | `resolve_follow_up()` word count check |
| CSV Change Detection | SHA-256 hash | `_get_csv_hash()` passed to all cached functions |

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

- **No cross-encoder re-ranking**: The "best match" is based on cosine similarity or keyword match order, not a dedicated relevance model. Adding a cross-encoder would improve accuracy but increase memory usage.
- **CSV-based data**: Data updates require replacing the CSV file. The app auto-detects changes via `_get_csv_hash()` and recomputes embeddings, but there is no admin UI for CRUD operations.
- **Spell correction scope**: Only corrects words to terms found in the dataset vocabulary. General English typos (e.g., "waht" -> "what") are not corrected.
- **Context window**: Very large aggregate queries may truncate at 50 records and 1,500 tokens (configured in `retrieve_context()` and `ask_llm_stream()`).
- **Single-process architecture**: Streamlit runs as a single Python process. Each user session shares the same in-memory model and embeddings, which is efficient for memory but limits CPU-bound concurrency.
- **Groq API rate limits**: The free Groq tier has request-per-minute and token-per-minute limits. Under heavy concurrent usage, some requests may be throttled or return rate-limit errors.
- **No authentication**: The app has no login or role-based access control. Anyone with the URL can access the full registry.
- **No persistent chat storage**: Chat history lives in `st.session_state` (browser tab memory). Refreshing the page or opening a new tab clears all conversation history.

## Scaling Recommendations (10,000+ Concurrent Users)

The current architecture is designed for small-to-medium usage (single Streamlit process, in-memory embeddings, direct Groq API calls). If usage grows beyond 10,000 concurrent users, the following changes are recommended.

### Infrastructure

| Concern | Current State | Recommendation |
| ------- | ------------- | -------------- |
| Web server | Single Streamlit process | Deploy behind a load balancer (NGINX, AWS ALB) with multiple Streamlit instances |
| Embedding model | Loaded in each process | Move to a shared embedding service (e.g., Triton Inference Server, TorchServe, or a dedicated FastAPI microservice) |
| Data storage | CSV file on disk | Migrate to a vector database (Pinecone, Weaviate, Qdrant, or pgvector) for scalable similarity search |
| LLM provider | Groq free tier | Upgrade to Groq paid tier, or use multiple LLM providers with failover (OpenAI, Anthropic, Azure OpenAI) |
| Caching | Streamlit in-memory cache | Add Redis or Memcached for shared query-result caching across instances |
| Session storage | `st.session_state` (per-tab) | Use Redis-backed session store or a database for persistent chat history |

### Architecture Changes

1. **Separate the frontend from the backend**: Extract the RAG pipeline (`_run_retrieval()`, `retrieve_context()`, `ask_llm_stream()`) into a standalone FastAPI or Flask service. The Streamlit UI becomes a thin client that calls the API. This allows scaling the backend independently.

2. **Pre-compute and index embeddings externally**: Instead of encoding all rows on startup via `load_model_and_embeddings()`, store pre-computed vectors in a vector database. This eliminates startup latency and supports datasets much larger than 1,650 records.

3. **Add a request queue for LLM calls**: At 10k+ users, LLM calls become the bottleneck. Use a message queue (RabbitMQ, Redis Streams, or AWS SQS) to manage requests and prevent overloading the LLM provider.

4. **Implement rate limiting and authentication**: Add API key or OAuth-based authentication. Rate-limit per user to prevent abuse and stay within LLM provider quotas.

5. **Use horizontal scaling**: Deploy multiple app instances behind a load balancer. Since each instance loads its own embedding model (~80MB), consider a shared model server to reduce total memory footprint.

### Estimated Resource Requirements (10k concurrent users)

| Resource | Estimate |
| -------- | -------- |
| App instances | 5-10 Streamlit workers (or FastAPI equivalents) behind a load balancer |
| Memory per instance | ~500MB (embedding model + DataFrame + PyTorch) |
| Vector database | Any managed service (Pinecone Starter, Qdrant Cloud, or self-hosted pgvector) |
| LLM API throughput | ~100-500 requests/min depending on user activity patterns |
| Redis cache | 1-2 GB for query caching and session storage |

### Quick Wins (No Architecture Change)

If you need to handle moderate growth (hundreds of users) before a full rewrite:

- **Groq paid tier**: Removes rate limits, the single biggest bottleneck for concurrent users.
- **Streamlit Community Cloud scaling**: Supports multiple simultaneous sessions out of the box, though with shared CPU.
- **Response caching**: Cache LLM responses for repeated/similar queries using `@st.cache_data` with TTL, reducing redundant API calls.
- **CDN for static assets**: If self-hosting, put Streamlit behind Cloudflare or AWS CloudFront to reduce server load from static file requests.
