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
LOW_CONFIDENCE_FALLBACK_THRESHOLD = 0.3  # below this, retry with prior context

# =============================================================================
# SPELL CORRECTION
# =============================================================================
SPELL_CORRECTION_CUTOFF = 0.8   # difflib ratio threshold for corrections
SPELL_MIN_WORD_LENGTH = 4       # skip words shorter than this
SPELL_LENGTH_TOLERANCE = 2      # max char difference between word and match

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
# FOLLOW-UP DETECTION
# =============================================================================
FOLLOW_UP_WORD_COUNT = 8  # questions shorter than this may be follow-ups
MAX_EFFECTIVE_QUERY_WORDS = 30  # cap accumulated query length for follow-ups

FOLLOW_UP_INDICATORS = [
    "that", "this", "these", "it", "them", "those", "the same",
    "more about", "tell me more", "what about", "how about",
    "can you explain", "elaborate", "expand on", "go deeper",
    "previous", "above", "earlier", "you mentioned",
    "any other", "the rest", "instead", "similar", "related", "compared",
]

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
# SPELL CORRECTION STOPWORDS
# Common English words that should never be spell-corrected to dataset terms.
# Without this, difflib can match e.g. "this" -> "ethics" (ratio 0.8, len diff 2).
# =============================================================================
STOPWORDS = {
    "this", "that", "these", "those", "them", "they", "their", "there",
    "then", "than", "what", "when", "where", "which", "while", "with",
    "will", "would", "could", "should", "about", "above", "after",
    "again", "also", "been", "before", "below", "between", "both",
    "came", "come", "does", "done", "down", "each", "even", "every",
    "find", "first", "from", "gave", "give", "goes", "gone", "good",
    "have", "here", "hers", "high", "hold", "into", "just", "keep",
    "kind", "know", "last", "left", "like", "list", "long", "look",
    "made", "make", "many", "more", "most", "much", "must", "name",
    "near", "need", "next", "note", "once", "only", "open", "over",
    "part", "past", "same", "show", "side", "some", "such", "sure",
    "take", "tell", "text", "time", "told", "took", "turn", "type",
    "upon", "very", "want", "well", "were", "whom", "wide", "work",
    "your", "being", "doing", "going", "having", "other", "still",
    "thing", "think", "under", "until", "using", "whose",
    "primary", "secondary", "control", "controls", "risk", "risks",
    "level", "levels", "describe", "description",
}

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
    "- **Risk Title**: the title\n"
    "- **Risk Level**: the level\n"
    "- **Primary Control** (type): full description\n"
    "- **Secondary Control** (type): full description\n\n"
    "## Other Controls to Consider\n"
    "List any other matching records. For EACH record, always use "
    "the exact bold labels **Primary Control** and **Secondary Control** "
    "followed by the control type in parentheses and the full description. "
    "Example format for each record:\n"
    "- **Risk Title**: title — **Risk Level**: level\n"
    "  - **Primary Control** (type): description\n"
    "  - **Secondary Control** (type): description\n\n"
    "IMPORTANT: The words 'Primary Control' and 'Secondary Control' "
    "must ALWAYS be bold (**Primary Control**, **Secondary Control**) "
    "in every section of the response — both Best Match and Other "
    "Controls to Consider. Always include the control type AND the "
    "full control description. Never omit secondary control details.\n"
    "Users may have typos or use informal terms — interpret their "
    "intent and match to the closest relevant records. For example, "
    "'hazadorus materials' should match 'Hazardous Waste Mishandling'. "
    "Only say 'I could not find that information in the registry' if "
    "none of the provided records are even remotely related to the question."
)

# =============================================================================
# UI SETTINGS
# =============================================================================
CHAT_FONT_SIZE_PX = 17

EXAMPLE_QUESTIONS = [
    "What controls exist for Hazardous Waste Mishandling?",
    "How many high risks are there?",
    "Tell me about Chemical Spill Response Delay",
]
