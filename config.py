# config.py -- All constants, thresholds, prompts, and keyword lists.
# Change values here to tune the app's behavior without touching business logic.

# =============================================================================
# FILE PATHS
# =============================================================================
CSV_PATH = "Epa risk and control registry.csv"

# =============================================================================
# EMBEDDING & RETRIEVAL
# =============================================================================
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBEDDING_DEVICE = "cpu"
TOP_K = 10                  # number of top embedding results to retrieve
MAX_RECORDS = 50            # max records sent to the LLM as context

# =============================================================================
# CONFIDENCE THRESHOLDS
# =============================================================================
HIGH_CONFIDENCE_THRESHOLD = 0.6     # cosine similarity >= this = "High"
MODERATE_CONFIDENCE_THRESHOLD = 0.4  # cosine similarity >= this = "Moderate"

# =============================================================================
# LLM SETTINGS
# =============================================================================
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1500
CHAT_HISTORY_WINDOW = 6  # number of recent messages (3 exchanges) to include

# =============================================================================
# RECORD SUMMARIZATION
# =============================================================================
SUMMARIZATION_THRESHOLD = 15  # max individual records before grouping by title

# =============================================================================
# AGGREGATE DETECTION
# =============================================================================
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

# =============================================================================
# LLM SYSTEM PROMPT
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
    "- **Risk Title**: title — **Risk Level**: level — **Cosine Score**: score\n"
    "  - **Primary Control** (type): full description\n"
    "  - **Secondary Control** (type): full description\n\n"
    "## Other Controls to Consider\n"
    "List any other matching records. For EACH record, use the SAME format "
    "as Best Match — one line with title, level, and cosine score, followed "
    "by indented primary and secondary controls. "
    "Example format for each record:\n"
    "- **Risk Title**: title — **Risk Level**: level — **Cosine Score**: score\n"
    "  - **Primary Control** (type): description\n"
    "  - **Secondary Control** (type): description\n\n"
    "IMPORTANT: The words 'Primary Control' and 'Secondary Control' "
    "must ALWAYS be bold (**Primary Control**, **Secondary Control**) "
    "in every section of the response — both Best Match and Other "
    "Controls to Consider. Always include the control type AND the "
    "full control description. Never omit secondary control details.\n"
    "Users WILL have typos, misspellings, and informal terms — "
    "ALWAYS interpret their intent generously and match to the closest "
    "relevant records. Examples: 'fraus' means 'fraud', "
    "'hazadorus' means 'hazardous', 'spilling' means 'spill', "
    "'enviroment' means 'environment'. If the provided records contain "
    "topics clearly related to what the user is asking about (even with "
    "typos), treat them as matches and present them using the standard "
    "Best Match / Other Controls format.\n"
    "Only say 'No matching information was found in the registry.' if "
    "none of the provided records are even remotely related to the question."
)

# =============================================================================
# FEEDBACK
# =============================================================================
FEEDBACK_CSV_PATH = "feedback.csv"
FEEDBACK_LOOKBACK = 10  # max recent feedback entries to include in LLM prompt

# =============================================================================
# UI SETTINGS
# =============================================================================
CHAT_FONT_SIZE_PX = 17

APP_TITLE = "EPA Risk Assessment Assistant"
APP_ICON = "shield"
APP_DESCRIPTION = (
    "Ask natural language questions about the EPA Risk and Control Registry. "
    "The assistant uses semantic search and keyword matching to find relevant "
    "risk records, then generates structured answers."
)

EXAMPLE_QUESTIONS = [
    "What controls exist for Hazardous Waste Mishandling?",
    "How many high risks are there?",
    "Tell me about Chemical Spill Response Delay",
]
