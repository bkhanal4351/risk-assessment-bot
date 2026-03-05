# EPA Risk Assessment Assistant

A Streamlit-powered chatbot that lets users ask natural language questions about the EPA Risk and Control Registry. It uses a RAG (Retrieval-Augmented Generation) pipeline that combines semantic search with keyword lookup to find the most relevant risk records, then passes them to an LLM for a structured, streaming answer.

## Features

- **Chat Interface**: Full conversational UI with chat history and sidebar navigation
- **User Feedback**: Thumbs up/down rating and optional comment box on every response, saved to CSV for continuous improvement
- **Feedback-Informed Answers**: Past user ratings and comments are included in the LLM prompt to improve future responses
- **Cosine Score Display**: Every record in the LLM response shows its cosine similarity score for transparency
- **Typo-Tolerant Search**: Two-layer matching -prefix-based for inflected forms ("spill" matches "spilling") and Damerau-Levenshtein edit distance for transposition typos ("farud" matches "fraud")
- **Dual Search Strategy**: Combines keyword-based lookup with embedding-based semantic search for better recall
- **Smart Aggregate Detection**: Two-pass system that distinguishes between specific risk queries and analytical questions
- **Streaming Responses**: LLM answers stream in real-time as tokens are generated
- **Confidence Scoring**: Color-coded relevance labels (High/Moderate/Low) based on cosine similarity
- **Source Citations**: Collapsible "Sources from Registry" panel showing the top 5 matched CSV rows with row numbers, risk titles, and similarity scores for full traceability
- **Inline Clear Chat Commands**: Type "clear", "reset", "new chat", etc. directly in the chat input to reset the conversation -no need to reach for the sidebar button
- **Sidebar with Examples**: Example question buttons in the sidebar for quick access, plus a "Clear Chat" button
- **Large Result Summarization**: Groups duplicate risk titles and shows counts when result sets are large

## Algorithm and Approach

### RAG (Retrieval-Augmented Generation)

The app follows the RAG pattern: instead of asking the LLM to answer from its training data (which may hallucinate), it first **retrieves** relevant records from the dataset and then passes them as context to the LLM to **generate** an answer grounded in real data.

### Step-by-Step Pipeline

```
User Question
      |
      v
1. AGGREGATE DETECTION (Two-Pass)
   Pass 1: Check if the question mentions a specific risk title
           (requires 2+ significant word matches). If yes -> SPECIFIC query.
   Pass 2: If no specific risk found, check for aggregate keywords
           ("how many", "list all", "breakdown", etc.) -> AGGREGATE query.
   This prevents "what are all controls for spill risk?" from being
   treated as aggregate when the user wants specific details.
      |
      v
2. DUAL RETRIEVAL
   Two search strategies run in parallel:

   a) KEYWORD LOOKUP (stem matching)
      Searches pre-computed lowercase dataframe columns for:
        - Risk levels ("high", "medium", "low", "critical")
        - Control types ("preventive", "detective", "corrective")
        - Risk title words (3 tiers):
            Tier 1: 2+ stem matches for multi-word titles
            Tier 2: 1 match for single-word titles
            Tier 3: 1 match if the word is distinctive (not generic)
        - Description keywords (fallback, requires 2+ word overlap)

   b) SEMANTIC SEARCH (embedding similarity)
      Encodes the question into a 768-dim vector using BAAI/bge-base-en-v1.5
      and computes cosine similarity against all pre-computed record vectors.
      Returns top 10 most similar records.
      |
      v
3. COMBINE AND DEDUPLICATE
   Merges keyword matches (high precision) with embedding matches
   (high recall), removing duplicates. Keyword results come first
   since they are exact matches.
      |
      v
4. SUMMARIZE (if needed)
   If more than 15 records match, groups them by risk title with counts
   and includes a sample of 15 individual records. This prevents
   exceeding the LLM's context window.
      |
      v
5. COSINE SCORE ATTACHMENT
   Every record (both keyword and embedding matches) gets a cosine
   similarity score. Embedding results already have scores from step 2b.
   Keyword-only matches are scored individually against the query vector.
   Scores are included in the LLM context for display in the response.
      |
      v
6. LLM GENERATION (Streaming)
   Sends to Llama 3.3 70B (via Groq API) with:
     - System prompt defining output format
     - Last 3 chat exchanges for conversational context
     - Recent user feedback (ratings + comments) for quality improvement
     - "BEST MATCH" labeled separately from "OTHER RELATED RECORDS"
     - Cosine scores attached to each record
   Response streams token-by-token to the UI.
```

### Key Algorithms

| Algorithm | Where Used | How It Works |
|-----------|-----------|--------------|
| Cosine Similarity | Semantic search | Measures angle between question vector and record vectors. Score 1.0 = identical meaning, 0.0 = unrelated. Used to rank the top 10 most relevant records. |
| Sentence Embedding (BAAI/bge-base-en-v1.5) | Vector encoding | Transformer model that maps text to 768-dimensional dense vectors where semantically similar sentences are close together. Top-ranked on the MTEB retrieval benchmark. |
| Prefix-based Stem Matching | Keyword lookup | Compares word prefixes to handle inflected forms (e.g., "spill" matches "spilling", "prevent" matches "preventive"). Requires shared prefix of at least 4 chars covering 75%+ of the shorter word. |
| Damerau-Levenshtein Distance | Keyword lookup (fallback) | Catches transposition and substitution typos that break prefix matching (e.g., "farud" matches "fraud", "enviroment" matches "environment"). Allows edit distance ≤1 for short words (≤5 chars) and ≤2 for longer words. Only compares words of similar length (±2 chars). |
| Distinctive Word Matching | Keyword lookup | Allows single-word title matches for distinctive (non-generic) words. Words like "spill", "fraud", "chemical" trigger matches, while generic words like "risk", "control", "management" do not. |
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
| Sentence Transformers | Embedding model (`BAAI/bge-base-en-v1.5`) for semantic search |
| Groq | LLM API provider for fast streaming inference |
| Llama 3.3 70B | Large language model for generating answers |
| Pandas | Data loading and manipulation |
| PyTorch | Backend for sentence transformer model |

## Project Structure

```
risk_assessment_bot/
  config.py                           All constants, thresholds, prompts, and keyword lists
  rag.py                              RAG pipeline: data loading, retrieval, LLM, feedback
  app.py                              Streamlit UI layer (sidebar, chat, feedback, clear commands)
  epa_logo.png                        EPA seal used as favicon and header logo
  .streamlit/config.toml              Dark theme configuration (colors, font)
  DEMO_GUIDE.md                       Demo script and feature showcase for presentations
  Epa risk and control registry.csv   EPA risk and control dataset (1,650 records)
  feedback.csv                        User feedback log (created at runtime, gitignored)
  requirements.txt                    Python dependencies
  .env                                Environment variables (Groq API key)
  .gitignore                          Excludes .env, feedback.csv, and cache files from git
  LICENSE                             MIT license
  README.md                           Project documentation
  TECHNICAL.md                        Line-by-line code documentation and algorithm details
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

If we choose not to use any virtual environment, we can skip step 2 in the setup instructions below and install directly with `pip install -r requirements.txt` -but this is not recommended for the reasons above.

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

4. Create a `.env` file and add our Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

5. Run the application:

```bash
streamlit run app.py
```

6. Open a browser to `http://localhost:8501`

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

4. Create a `.env` file in the project root and add our Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

5. Run the application:

```cmd
streamlit run app.py
```

6. Open a browser to `http://localhost:8501`

**Windows Notes:**
- If `streamlit` is not recognized, try `python -m streamlit run app.py`
- If we get a PyTorch error, install the CPU version: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- Make sure the `.env` file uses UTF-8 encoding (not UTF-16)

### Deploying to Streamlit Cloud

1. Push the repo to GitHub (make sure `.env` is in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) and deploy the repo
3. In app settings, go to **Secrets** and add:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

The app will be live at a URL like `https://our-app-name.streamlit.app`

### Running on a VDI (Virtual Desktop Infrastructure)

If we are running on a corporate VDI or locked-down environment where we cannot deploy to Streamlit Cloud, we can run the app locally as a standalone server that other users on the same network can access.

1. Follow the **macOS / Linux** or **Windows** setup steps above to install dependencies.

2. Set the Groq API key as an environment variable (if `.env` is not supported in our VDI):

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

4. Other users on the same network can access the app at `http://<our-vdi-ip>:8501`. Find the IP with:

```bash
# Linux / macOS
hostname -I

# Windows
ipconfig
```

**VDI Notes:**

- If our VDI blocks outbound internet, we will need to pre-download the embedding model and Groq API access may require a proxy. Set `HTTPS_PROXY` if needed.
- If port 8501 is blocked by firewall, try a different port: `--server.port 8080`
- For headless VDIs with no browser, use `--server.headless true` to suppress the auto-open behavior.
- If Python is not available on the VDI, ask IT to install Python 3.10+ or use a portable Python distribution.
- To keep the app running after we close the terminal, use `nohup` (Linux) or run it as a background service:

```bash
nohup streamlit run app.py --server.address 0.0.0.0 --server.port 8501 > app.log 2>&1 &
```

## Example Questions

### Specific Risk Lookups

- "What is the Unliquidated Obligations risk?"
- "Tell me about Hazardous Waste Mishandling"
- "What controls are in place for Chemical Spill Response Delay?"

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

### Semantic / Natural Language

- "What risks are related to environmental damage?"
- "Are there any risks about vendor management?"
- "How to handle hazardous materials?"

## Output Format

For specific risk questions, the assistant responds with:

**Best Match**: The single most relevant record showing risk title, risk level, cosine score, primary control (type + description), and secondary control (type + description).

**Other Controls to Consider**: Additional related records, each with the same format -risk title, risk level, cosine score, and both controls.

Every record displays its **cosine similarity score** so users can gauge how relevant each match is.

For aggregate questions, the assistant provides counts, breakdowns, and summaries from the full dataset.

### Sources from Registry

Every answer includes a collapsible "Sources from Registry" panel that shows the top 5 CSV rows the system matched against, ranked by similarity score. Each entry displays:

- **Row number** (1-indexed, matching the CSV/Excel row for easy lookup)
- **Risk title** from that row
- **Similarity score** (cosine similarity rounded to 2 decimal places)

This lets users trace any answer back to the exact records in the registry and verify the information independently.

## Architecture Diagram

```
User Question
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
|   lowercase cols,   |     |   using BGE-base     |
|   2+ word matching) |     |   768-dim vectors)   |
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
     +-------------------+     +-------------------+
     | Groq LLM          |<----| User Feedback     |
     | (Llama 3.3 70B)   |     | (from feedback.csv|
     | + Chat History     |     |  ratings+comments)|
     | + Streaming        |     +-------------------+
     +-------------------+
               |
               v
     Streaming Answer
     with Cosine Scores
     + Confidence Label
     + Source Citations
     + Feedback (thumbs + comment)
```

## Feedback System

The app includes a feedback loop that allows users to rate responses and have those ratings influence future answers.

### How Feedback Is Collected

Every assistant message displays three feedback elements simultaneously:

1. **Thumbs Up (👍)** -Positive rating
2. **Thumbs Down (👎)** -Negative rating
3. **Optional Comment Box** -Free-text input for specific feedback (e.g., "Missing secondary controls", "Wrong risk matched")

The user types an optional comment, then clicks either 👍 or 👎 to submit. Both the rating and comment are saved together. Each message can only be rated once -after submission, the buttons are replaced with a read-only confirmation.

### How Feedback Is Stored

Feedback is appended to `feedback.csv` with the following columns:

| Column | Description | Example |
| ------ | ----------- | ------- |
| timestamp | ISO 8601 timestamp | 2026-03-04T14:23:45 |
| question | The user's original question | "What controls for fraud?" |
| response_summary | First 200 chars of the LLM response | "## Best Match..." |
| rating | "up" or "down" | "down" |
| comment | Optional user comment | "Missing secondary controls" |

The file is append-only, auto-created on first feedback, and gitignored.

### How Feedback Influences Future Answers

On every new user question, the app:

1. **Loads** the most recent 10 feedback entries from `feedback.csv`
2. **Formats** them as a text block with the question, rating (👍/👎), and comment
3. **Prepends** this block to the LLM's input, BEFORE the retrieved context and question
4. The LLM reads past feedback first, then generates its response accordingly

Example of what the LLM sees:

```text
USER FEEDBACK ON PAST ANSWERS (use this to improve your response):
- Q: "What controls for fraud?" -Rated: 👎 -Comment: "Missing secondary controls"
- Q: "How many high risks?" -Rated: 👍

BEST MATCH (Cosine Score: 0.72):
Risk: Fraud Detection Failure. Description: ...

Question: What are the controls for fraud risk?
```

**Effects on LLM behavior**:

- **Negative ratings with comments** cause the LLM to address those specific issues (e.g., including more detail on secondary controls)
- **Positive ratings** reinforce the response style the LLM used
- **Recency**: Only the last 10 entries are used, so behavior reflects recent satisfaction
- **Scope**: Feedback affects LLM generation only -the retrieval pipeline (keyword + embedding search) is not modified by feedback

## Configuration

| Setting | Value | Defined In |
|---------|-------|------------|
| Embedding Model | BAAI/bge-base-en-v1.5 (768-dim) | `config.py` |
| LLM Model | llama-3.3-70b-versatile | `config.py` |
| Top K Results | 10 | `config.py` |
| LLM Temperature | 0.1 | `config.py` |
| Max Tokens | 1500 | `config.py` |
| Max Records to LLM | 50 | `config.py` |
| Chat History Window | Last 3 exchanges (6 messages) | `config.py` |
| Record Summarization Threshold | 15 records | `config.py` |
| Feedback Lookback | 10 recent entries | `config.py` |
| Feedback Storage | feedback.csv (local) | `config.py` |
| Clear Chat Phrases | clear, reset, new chat, start over, etc. | `config.py` |
| CSV Change Detection | SHA-256 hash | `rag.py` |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `GROQ_API_KEY not found` | Create a `.env` file with `GROQ_API_KEY=your_key` or set it in Streamlit Cloud secrets |
| `streamlit: command not found` | Use `python -m streamlit run app.py` instead |
| PyTorch installation fails on Windows | Install CPU version: `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| App is slow on first load | The embedding model downloads on first run (~110MB for BGE-base). Subsequent runs use the cached model. |
| Feedback not saving | Check that the app has write permissions to the project directory for `feedback.csv`. |
| `ModuleNotFoundError` | Make sure we activated the virtual environment and ran `pip install -r requirements.txt` |

## Known Limitations

- **No cross-encoder re-ranking**: The "best match" is based on cosine similarity or keyword match order, not a dedicated relevance model. Adding a cross-encoder would improve accuracy but increase memory usage.
- **CSV-based data**: Data updates require replacing the CSV file. The app auto-detects changes via `_get_csv_hash()` and recomputes embeddings, but there is no admin UI for CRUD operations.
- **Context window**: Very large aggregate queries may truncate at 50 records and 1,500 tokens (configured in `retrieve_context()` and `ask_llm_stream()`).
- **Single-process architecture**: Streamlit runs as a single Python process. Each user session shares the same in-memory model and embeddings, which is efficient for memory but limits CPU-bound concurrency.
- **Groq API rate limits**: The free Groq tier has request-per-minute and token-per-minute limits. Under heavy concurrent usage, some requests may be throttled or return rate-limit errors.
- **No authentication**: The app has no login or role-based access control. Anyone with the URL can access the full registry.
- **No persistent chat storage**: Chat history lives in `st.session_state` (browser tab memory). Refreshing the page or opening a new tab clears all conversation history.
- **Local feedback storage**: Feedback is stored in a local CSV file. In multi-instance deployments, each instance has its own feedback file. For shared feedback across instances, migrate to a database.

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

5. **Use horizontal scaling**: Deploy multiple app instances behind a load balancer. Since each instance loads its own embedding model (~110MB), consider a shared model server to reduce total memory footprint.

### Estimated Resource Requirements (10k concurrent users)

| Resource | Estimate |
| -------- | -------- |
| App instances | 5-10 Streamlit workers (or FastAPI equivalents) behind a load balancer |
| Memory per instance | ~500MB (embedding model + DataFrame + PyTorch) |
| Vector database | Any managed service (Pinecone Starter, Qdrant Cloud, or self-hosted pgvector) |
| LLM API throughput | ~100-500 requests/min depending on user activity patterns |
| Redis cache | 1-2 GB for query caching and session storage |

### Production Path: Scaling the LLM and UI

#### The Core Idea: It Is Just an API Call

Our entire LLM integration is a single API call in `rag.py`. Right now that call goes to Groq, but swapping to any other provider is a minimal code change. The RAG pipeline (retrieval, context building, prompt formatting) stays exactly the same. Only the function that sends the prompt and receives tokens changes.

For example, switching from Groq to OpenAI, Anthropic, or any OpenAI-compatible endpoint:

```python
# Current (Groq)
from groq import Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=messages,
    stream=True,
)

# OpenAI (same SDK interface, different base URL and key)
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    stream=True,
)

# Anthropic
import anthropic
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
response = client.messages.stream(
    model="claude-sonnet-4-20250514",
    messages=messages,
    max_tokens=1500,
)

# AWS Bedrock (uses boto3, no third-party SDK needed)
import boto3, json
bedrock = boto3.client("bedrock-runtime")
response = bedrock.invoke_model_with_response_stream(
    modelId="anthropic.claude-3-sonnet-20240229-v1:0",
    body=json.dumps({"messages": messages, "max_tokens": 1500}),
)

# Any OpenAI-compatible provider (Ollama, LM Studio, vLLM, Together AI)
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:11434/v1",   # Ollama example
    api_key="not-needed",
)
response = client.chat.completions.create(
    model="llama3.3:70b",
    messages=messages,
    stream=True,
)
```

The point is: our app is not locked into Groq or any specific model. Changing the LLM provider is a 5-line code change in `rag.py` and a new environment variable for the API key. Everything else (retrieval, prompt, UI, feedback) stays untouched.

Similarly, for the UI: Streamlit is great for prototyping and small-team use (we already cover VDI deployment in the setup section above). But if we need to serve 10,000+ users with custom branding, authentication, and persistent chat, we swap the frontend to React or use an AWS-managed chat interface. The backend API stays the same.

#### Why AWS Bedrock (for Agency-Scale Deployment)

AWS Bedrock is a fully managed service that gives us access to foundation models (Claude, Llama, Titan, etc.) without provisioning GPU instances. Key advantages for our use case:

- **No rate limit concerns**: Bedrock scales automatically with provisioned throughput. We purchase model units based on expected traffic instead of hitting per-minute API caps.
- **Data stays in our AWS account**: Unlike third-party APIs (Groq, OpenAI), data sent to Bedrock never leaves our VPC. This matters for EPA risk data that may have sensitivity restrictions.
- **IAM integration**: Access control ties directly into existing AWS IAM roles and policies. We can restrict who can call the model, audit every request via CloudTrail, and enforce encryption at rest and in transit.
- **Knowledge Bases for Amazon Bedrock**: Bedrock has a built-in RAG feature called Knowledge Bases. We can upload our CSV (or convert it to a supported format), and Bedrock handles chunking, embedding, vector storage (backed by OpenSearch Serverless or Pinecone), and retrieval automatically. This replaces our custom `rag.py` pipeline entirely.
- **Model flexibility**: Bedrock gives us access to multiple foundation models through a single API. We can swap between them with a config change while keeping the retrieval layer the same.

#### LLM Options on Bedrock

We currently use Llama 3.3 70B via Groq. On Bedrock, we have several model families to choose from depending on cost, speed, and accuracy requirements:

| Model | Provider | Strengths | Best For |
| --- | --- | --- | --- |
| Claude 3.5 Sonnet / Claude 4 | Anthropic | Strong reasoning, structured output, low hallucination | Our primary choice for risk/control Q&A where accuracy matters |
| Llama 3.1 70B / 405B | Meta | Open-weight, good general performance | Drop-in replacement for our current Groq-hosted Llama setup |
| Amazon Titan Text | Amazon | Native AWS integration, lowest latency within Bedrock | High-throughput aggregate queries where speed matters more than nuance |
| Mistral Large | Mistral AI | Fast inference, strong multilingual support | If we need to support non-English users in the future |
| Cohere Command R+ | Cohere | Built-in RAG optimization, grounded generation | Alternative if we want the LLM to handle retrieval-aware generation natively |

We can also set up **model fallback**: if our primary model (e.g., Claude) hits a throughput limit or returns an error, the API layer automatically retries with a secondary model (e.g., Titan). This is a config-level change in our Lambda function, no frontend changes needed.

#### Replacing the Retrieval Pipeline

| Current Component | Bedrock Equivalent |
| --- | --- |
| `sentence-transformers` (BAAI/bge-base-en-v1.5) | Bedrock Titan Embeddings or Cohere Embed via Bedrock |
| In-memory cosine similarity search | OpenSearch Serverless (vector engine) or Pinecone via Knowledge Bases |
| `rag.py` keyword + embedding hybrid search | Knowledge Bases retrieval with metadata filtering (risk level, control type) |
| Groq API (`ask_llm_stream`) | Bedrock `InvokeModelWithResponseStream` API |
| `feedback.csv` | DynamoDB table for feedback, queried at generation time |

With Knowledge Bases, the entire `retrieve_context()` function becomes a single API call:

```python
bedrock_agent = boto3.client("bedrock-agent-runtime")
response = bedrock_agent.retrieve_and_generate(
    input={"text": user_question},
    retrieveAndGenerateConfiguration={
        "type": "KNOWLEDGE_BASE",
        "knowledgeBaseConfiguration": {
            "knowledgeBaseId": "our-kb-id",
            "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet",
        },
    },
)
```

#### Replacing Streamlit with a Production Frontend

Streamlit works well for prototyping and internal demos (we already cover VDI deployment in the setup section above), but it has limitations at scale:

- **Single-threaded Python process** per user session
- **No built-in authentication** or role-based access
- **Session state is ephemeral** (lost on page refresh)
- **Limited UI customization** compared to a full frontend framework

For 10,000+ users, we have two options: build a custom React frontend deployed on EC2, or use an AWS-managed chat UI that requires no frontend code at all.

#### Option A: Custom React Frontend on EC2

This gives us full control over the UI, branding, and user experience. We build a React app that talks to our Python API layer, and deploy both on EC2.

**Architecture:**

| Layer | Technology | Purpose |
| --- | --- | --- |
| Frontend | React / Next.js on EC2 (behind ALB) | Custom chat UI with sidebar, feedback, source citations |
| Auth | AWS Cognito or agency SSO (SAML/OAuth) | User authentication and access control |
| API | FastAPI on the same EC2 (or separate instance) | Python backend that handles retrieval and LLM calls |
| RAG | Bedrock Knowledge Bases | Managed retrieval pipeline with vector search |
| LLM | Bedrock (Claude, Llama, Titan, or Mistral) | Streaming response generation |
| Chat Storage | DynamoDB | Persistent chat history across sessions and devices |
| Feedback | DynamoDB | Feedback ratings and comments, queryable for analytics |
| Monitoring | CloudWatch + X-Ray | Request tracing, latency dashboards, error alerting |

**Step-by-step EC2 deployment:**

1. **Launch an EC2 instance**: We use an Amazon Linux 2023 or Ubuntu 22.04 AMI. For 10,000+ users, start with a `t3.xlarge` (4 vCPUs, 16 GB RAM) for the API and a `t3.medium` for the frontend. Both go in the same VPC.

2. **Set up the API server on EC2**:

```bash
# SSH into the API instance
ssh -i our-key.pem ec2-user@<api-instance-ip>

# Install Python and dependencies
sudo yum install python3.11 git -y   # Amazon Linux
sudo apt install python3.11 git -y   # Ubuntu

# Clone the repo and install
git clone https://github.com/bkhanal4351/risk-assessment-bot.git
cd risk-assessment-bot
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install fastapi uvicorn boto3

# Configure AWS credentials for Bedrock access
aws configure   # or attach an IAM role to the EC2 instance (preferred)

# Create a systemd service so the API starts on boot
sudo tee /etc/systemd/system/risk-api.service << 'EOF'
[Unit]
Description=Risk Assessment API
After=network.target

[Service]
User=ec2-user
WorkingDirectory=/home/ec2-user/risk-assessment-bot
ExecStart=/home/ec2-user/risk-assessment-bot/venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000
Restart=always
Environment=AWS_DEFAULT_REGION=us-east-1

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable risk-api
sudo systemctl start risk-api
```

3. **Build and deploy the React frontend on EC2**:

```bash
# SSH into the frontend instance (or use the same instance)
ssh -i our-key.pem ec2-user@<frontend-instance-ip>

# Install Node.js
curl -fsSL https://rpm.nodesource.com/setup_20.x | sudo bash -
sudo yum install nodejs nginx -y

# Build the React app
cd risk-assessment-frontend
npm install
npm run build   # produces a static build/ folder

# Copy build output to NGINX web root
sudo cp -r build/* /usr/share/nginx/html/

# Configure NGINX to serve the React app and proxy API calls
sudo tee /etc/nginx/conf.d/risk-app.conf << 'EOF'
server {
    listen 80;
    server_name _;

    root /usr/share/nginx/html;
    index index.html;

    # React client-side routing: serve index.html for all routes
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Proxy API calls to the FastAPI backend
    location /api/ {
        proxy_pass http://<api-instance-private-ip>:8000/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 300s;   # long timeout for streaming LLM responses
    }
}
EOF

sudo systemctl enable nginx
sudo systemctl restart nginx
```

4. **Put an Application Load Balancer (ALB) in front**:
   - Create an ALB in the EC2 console, attach it to our VPC's public subnets
   - Create a target group pointing to the frontend EC2 instance(s) on port 80
   - Add an HTTPS listener (port 443) with an ACM certificate for our domain
   - Route `/api/*` to the API target group, everything else to the frontend target group
   - Enable sticky sessions if we use server-side session state

5. **Auto Scaling for traffic spikes**:
   - Create an AMI from our configured EC2 instances
   - Set up an Auto Scaling Group with min 2, desired 3, max 10 instances
   - Use CPU utilization (target 70%) or request count as the scaling metric
   - The ALB distributes traffic across healthy instances automatically

6. **DNS and SSL**:
   - Point our domain (e.g., `risk-bot.epa.gov`) to the ALB using Route 53
   - ACM provides free SSL certificates that auto-renew

#### Option B: AWS Managed Chat UI (No Frontend Code)

If we want to skip building a custom React app entirely, AWS provides managed chat interfaces that connect directly to Bedrock:

##### Amazon Bedrock Agents + Bedrock Chat Playground

Bedrock's built-in console playground supports multi-turn chat with Knowledge Base integration. For internal agency use, team members can access it directly through the AWS Console with their IAM credentials. This works for smaller teams (50-200 users) but is not ideal for a public-facing or large-scale deployment since it requires AWS Console access.

##### Amazon Q Business (recommended for managed UI)

Amazon Q Business is a fully managed AI assistant service from AWS. We configure it with our data source (S3 bucket containing the risk registry CSV), and Amazon Q provides:

- A **hosted chat web app** with a shareable URL, no frontend code needed
- **Built-in authentication** via IAM Identity Center (SSO)
- **Document-grounded answers** with source citations (similar to our current "Sources from Registry" feature)
- **Admin controls** for managing data sources, access permissions, and response guardrails
- **Feedback collection** built into the UI (thumbs up/down)

Setup:

1. Upload `Epa risk and control registry.csv` to an S3 bucket
2. In the Amazon Q Business console, create an application and add the S3 bucket as a data source
3. Configure IAM Identity Center for user authentication
4. Amazon Q indexes the data and provides a chat URL we can share with all 10,000+ users
5. No EC2 instances, no NGINX, no React build process

**Trade-offs between Option A and Option B:**

| | Option A (React + EC2) | Option B (Amazon Q Business) |
| --- | --- | --- |
| Custom UI/branding | Full control over look and feel | Limited to Amazon Q's built-in interface |
| Development effort | 4-8 weeks to build and deploy | 1-2 days to configure |
| Maintenance | We manage EC2, NGINX, scaling, SSL | Fully managed by AWS |
| Feedback system | Custom (our DynamoDB-backed approach) | Built-in thumbs up/down |
| Authentication | Cognito or any SAML/OAuth provider | IAM Identity Center (SSO) |
| Cost | Higher (EC2 instances + ALB + DevOps time) | Lower (pay per indexed document + per query) |
| Best for | Public-facing app with EPA branding | Internal agency tool for staff |

#### Deployment Architecture (Option A)

```text
Users (10,000+)
      |
      v
Route 53 (DNS)
      |
      v
ALB (HTTPS, auto-scaling)
      |
      +--> EC2 Frontend (React + NGINX)
      |
      +--> EC2 API (FastAPI + Python)
              |
              +-------> Bedrock Knowledge Bases (retrieval)
              |              |
              |              v
              |         OpenSearch Serverless (vector store)
              |
              +-------> Bedrock LLM (streaming generation)
              |
              +-------> DynamoDB (chat history + feedback)
              |
              +-------> CloudWatch (logging + monitoring)
```

#### Migration Steps

1. **Export the CSV to a Bedrock Knowledge Base**: Upload `Epa risk and control registry.csv` to S3, create a Knowledge Base with Titan Embeddings, and let Bedrock handle indexing.
2. **Replace `rag.py` with a FastAPI service**: The API calls Bedrock's `retrieve_and_generate` endpoint and streams the response back using Server-Sent Events (SSE) or WebSocket.
3. **Build the React frontend** (Option A) or **configure Amazon Q Business** (Option B): For React, port the chat UI, sidebar, feedback buttons, and source citations from Streamlit to React components. For Amazon Q, add the S3 data source and configure IAM Identity Center.
4. **Deploy to EC2** (Option A): Follow the step-by-step guide above to set up NGINX, the API service, ALB, and Auto Scaling.
5. **Set up authentication**: Cognito user pool for Option A, or IAM Identity Center for Option B.
6. **Migrate feedback to DynamoDB**: Create a table with `message_id` as partition key, storing question, rating, comment, and timestamp. Query recent feedback at generation time.
7. **Deploy infrastructure as code**: Define the full stack using CDK or Terraform for repeatable deployments across environments (dev, staging, prod).

#### Cost Estimate (10,000 monthly active users)

| Service | Estimated Monthly Cost |
| --- | --- |
| Bedrock LLM (Claude Sonnet, ~500K requests) | $500 - $1,500 |
| Bedrock Knowledge Bases + OpenSearch Serverless | $200 - $500 |
| Lambda (API compute) | $50 - $150 |
| DynamoDB (chat + feedback storage) | $25 - $50 |
| S3 + CloudFront (frontend hosting) | $10 - $30 |
| Cognito (authentication) | Free tier covers 50K MAUs |
| **Total** | **~$800 - $2,200/month** |

Costs scale roughly linearly with usage. Bedrock's per-token pricing means we only pay for actual LLM calls, not idle GPU time.

### Quick Wins (No Architecture Change)

If we need to handle moderate growth (hundreds of users) before a full rewrite:

- **Groq paid tier**: Removes rate limits, the single biggest bottleneck for concurrent users.
- **Streamlit Community Cloud scaling**: Supports multiple simultaneous sessions out of the box, though with shared CPU.
- **Response caching**: Cache LLM responses for repeated/similar queries using `@st.cache_data` with TTL, reducing redundant API calls.
- **CDN for static assets**: If self-hosting, put Streamlit behind Cloudflare or AWS CloudFront to reduce server load from static file requests.
