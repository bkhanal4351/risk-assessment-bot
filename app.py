import streamlit as st

# set_page_config MUST be the first Streamlit command, before any imports
# that trigger @st.cache_data or @st.cache_resource (e.g. rag.py).
from config import (
    EXAMPLE_QUESTIONS, CHAT_FONT_SIZE_PX,
    APP_TITLE, APP_ICON, APP_DESCRIPTION,
    CLEAR_CHAT_PHRASES,
)

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=f":{APP_ICON}:",
    layout="wide",
)

from rag import (  # noqa: E402
    retrieve_context, ask_llm_stream, confidence_label,
    save_feedback, load_recent_feedback,
)

# =============================================================================
# STYLING
# =============================================================================

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

# =============================================================================
# SIDEBAR
# =============================================================================

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

# =============================================================================
# FEEDBACK HELPER
# =============================================================================

def _render_feedback(msg_index, message):
    """Renders thumbs up/down + optional comment, all visible at once."""
    feedback_key = f"feedback_{msg_index}"

    if feedback_key in st.session_state:
        saved = st.session_state[feedback_key]
        icon = "👍" if saved == "up" else "👎"
        comment = st.session_state.get(f"saved_comment_{msg_index}", "")
        label = f"Feedback recorded: {icon}"
        if comment:
            label += f' - "{comment}"'
        st.caption(label)
        return

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

# =============================================================================
# SOURCE CITATIONS HELPER
# =============================================================================

def _render_sources(msg_index, message):
    """Renders source citations stored with the message."""
    sources = message.get("sources", [])
    if sources:
        with st.expander("Sources from Registry"):
            for rank, (row_num, title, sim_score) in enumerate(sources, 1):
                st.markdown(f"**#{rank}** | Row {row_num} | {title} | Score: {sim_score}")

# =============================================================================
# MAIN CHAT AREA
# =============================================================================

st.title(APP_TITLE)
st.caption("Ask questions about risks, controls, and mitigations from the EPA Risk and Control Registry.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.info("Get started by typing a question below or selecting an example from the sidebar.")

# Display previous messages with sources and feedback
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            # Show relevance badge
            if "score" in message:
                label, color = confidence_label(message["score"])
                st.markdown(f"**Relevance:** :{color}[{label}] ({message['score']:.2f})")
            _render_sources(i, message)
            _render_feedback(i, message)

# Chat input
user_question = st.chat_input("Ask a question about a risk or control...")

if "pending_question" in st.session_state:
    user_question = st.session_state.pending_question
    del st.session_state.pending_question

if user_question and user_question.strip().lower() in CLEAR_CHAT_PHRASES:
    st.session_state.messages = []
    st.rerun()

if user_question:
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

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

    # Store sources and score with the message so they persist across reruns
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "question": user_question,
        "sources": sources,
        "score": score,
    })
    st.rerun()
