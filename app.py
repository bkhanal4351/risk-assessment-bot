import streamlit as st
from rag import retrieve_context, ask_llm_stream, confidence_label
from config import EXAMPLE_QUESTIONS, CHAT_FONT_SIZE_PX

# =============================================================================
# STREAMLIT UI (CHAT INTERFACE)
# Thin UI layer that wires up user interaction to the RAG pipeline in rag.py.
# All business logic (retrieval, spell correction, LLM streaming) lives in
# rag.py; all tunable constants live in config.py.
# =============================================================================

st.markdown(f"""
<style>
    .stChatMessage p, .stChatMessage li, .stChatMessage td {{
        font-size: {CHAT_FONT_SIZE_PX}px !important;
    }}
</style>
""", unsafe_allow_html=True)

st.title("EPA Risk Assessment Assistant")
st.caption("Ask questions about risks, controls, and mitigations from the EPA Risk and Control Registry.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Example question buttons (shown only when chat history is empty)
if not st.session_state.messages:
    st.markdown("**Try an example question:**")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for i, example in enumerate(EXAMPLE_QUESTIONS):
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
        context, score, corrected, is_agg, sources = retrieve_context(
            user_question, st.session_state.messages[:-1],
            st.session_state.get("last_effective_query")
        )

        # Store the effective query so future follow-ups reference the right topic
        st.session_state.last_effective_query = corrected

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

        # Show source citations in a collapsible expander
        if sources:
            with st.expander("Sources from Registry"):
                for rank, (row_num, title, sim_score) in enumerate(sources, 1):
                    st.markdown(f"**#{rank}** | Row {row_num} | {title} | Score: {sim_score}")

    st.session_state.messages.append({"role": "assistant", "content": response_text})
