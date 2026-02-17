# EPA Risk Assessment Assistant

A Streamlit-powered chatbot that lets users ask natural language questions about the EPA Risk and Control Registry. It uses a RAG (Retrieval-Augmented Generation) pipeline that combines semantic search with keyword lookup to find the most relevant risk records, then passes them to an LLM for a structured answer.

## How It Works

1. **Data Loading**: Reads the EPA Risk and Control Registry CSV file (1,650 records) and standardizes column names for internal use.

2. **Embedding Generation**: Each risk record is converted into a sentence and encoded into a 384-dimensional vector using the `all-MiniLM-L6-v2` sentence transformer model. These embeddings are cached in memory so they only compute once.

3. **User Query Processing**: When a user asks a question, the app runs two parallel search strategies:
   - **Keyword Lookup**: Searches the dataframe directly for matching risk levels, control types, and title keywords.
   - **Semantic Search**: Encodes the question into a vector and finds the top 10 most similar records using cosine similarity.

4. **Aggregate Detection**: If the question contains aggregate keywords (e.g., "how many", "list all", "breakdown"), the app includes pre-computed dataset summary statistics in the context.

5. **LLM Response**: The retrieved records are sent to Llama 3.3 70B (via Groq) with a system prompt that instructs it to format the answer with a "Best Match" section and "Other Controls to Consider" section.

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
| Streamlit | Web UI framework |
| Sentence Transformers | Embedding model (`all-MiniLM-L6-v2`) for semantic search |
| Groq | LLM API provider for fast inference |
| Llama 3.3 70B | Large language model for generating answers |
| Pandas | Data loading and manipulation |
| PyTorch | Backend for sentence transformer model |

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

**Best Match**: The single most relevant record showing risk title, risk level, primary control (type + description), and secondary control (type + description).

**Other Controls to Consider**: Additional related records with their control details for a broader view.

For aggregate questions, the assistant provides counts, breakdowns, and summaries from the full dataset.

## Architecture Diagram

```
User Question
      |
      v
+---------------------+
|   Streamlit UI      |
+---------------------+
      |
      v
+---------------------+     +----------------------+
|  Keyword Lookup     |     |  Semantic Search     |
|  (exact matching    |     |  (cosine similarity  |
|   on risk level,    |     |   using MiniLM-L6    |
|   title, control)   |     |   embeddings)        |
+---------------------+     +----------------------+
      |                           |
      +--------+------------------+
               |
               v
     +-------------------+
     | Combine & Dedupe  |
     | (Best Match +     |
     |  Other Records)   |
     +-------------------+
               |
               v
     +-------------------+
     | Groq LLM          |
     | (Llama 3.3 70B)   |
     +-------------------+
               |
               v
     Structured Answer
```

## Configuration

| Setting | Value | Location |
|---------|-------|----------|
| Embedding Model | all-MiniLM-L6-v2 | `app.py` line 49 |
| LLM Model | llama-3.3-70b-versatile | `app.py` line 118 |
| Top K Results | 10 | `app.py` line 207 |
| LLM Temperature | 0.1 | `app.py` line 157 |
| Max Tokens | 1500 | `app.py` line 158 |
| Max Records to LLM | 50 | `app.py` line 231 |
