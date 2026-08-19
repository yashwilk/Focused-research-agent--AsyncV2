"""
Streamlit chat page for the Focused Research Agent UI.

This module implements the conversational research interface. It uses
Streamlit's multi-page convention — placing it in the pages/ folder
makes it appear as a separate page in the sidebar navigation.

Architecturally, this module is a UI transport entrypoint alongside
1_🔍_Research.py. It follows the same pattern: thin wiring layer that calls
api_client for data and delegates rendering to views and inline
st.* calls for chat-specific widgets.
"""

import streamlit as st

from focused_research_agent.ui.api_client import (
    call_chat,
    check_health,
    get_conversation,
    get_conversations,
)
from focused_research_agent.ui.exceptions import BackendUnavailableError
from focused_research_agent.ui.views import render_health_status


def _init_session_state() -> None:
    """
    Initialise session state keys for the chat page.

    Uses the guard pattern to avoid resetting state on every rerun.

    Returns:
        None
    """
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def _render_sidebar() -> None:
    """
    Render all sidebar content for the chat page.

    Displays the health status, a new conversation button, and the
    list of past conversations with load buttons.

    Returns:
        None
    """
    st.sidebar.title("💬 Chat")
    api_health = check_health()
    render_health_status(api_health)

    if st.sidebar.button("New Conversation"):
        st.session_state.conversation_id = None
        st.session_state.messages = []
        st.rerun()

    conversations = get_conversations(token=st.session_state.get("token"))

    if conversations:
        st.sidebar.subheader("📋 Past Conversations")
        for convo in conversations:
            with st.sidebar.expander(convo["title"] or "Untitled"):
                if st.button("Load", key=convo["conversation_id"]):
                    st.session_state.conversation_id = convo["conversation_id"]
                    turns = get_conversation(
                        convo["conversation_id"], token=st.session_state.get("token")
                    )
                    st.session_state.messages = []
                    for turn in turns:
                        st.session_state.messages.append(
                            {
                                "role": "user",
                                "content": turn["question"],
                            }
                        )
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": turn["answer"] or "",
                            }
                        )
                    st.rerun()


def _render_chat_history() -> None:
    """
    Render all messages in the current conversation thread.

    Reads from session state and renders each message using
    st.chat_message so user and assistant messages are visually
    distinguished.

    Returns:
        None
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def _handle_chat_input() -> None:
    """
    Handle chat input and send the question to the backend.

    Renders the chat input bar, appends the user message to session
    state, calls the backend, and renders the assistant response.
    Updates conversation_id in session state so follow-up questions
    are threaded into the same conversation.

    Returns:
        None
    """

    question = st.chat_input("Ask a research question...")
    if question is not None:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        try:
            with st.spinner("Researching..."):
                result = call_chat(
                    question,
                    st.session_state.conversation_id,
                    token=st.session_state.get("token"),
                )
        except BackendUnavailableError as e:
            st.error(str(e))
            st.stop()

        if result["success"]:
            data = result["data"]
            st.session_state.conversation_id = data["conversation_id"]
            answer = data.get("answer") or "No answer returned."
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
        else:
            error_message = result["error"] or "An error occurred."
            st.session_state.messages.append(
                {"role": "assistant", "content": f"❌ {error_message}"}
            )
            with st.chat_message("assistant"):
                st.error(error_message)


st.set_page_config(page_title="Research Chat", layout="centered")
st.title("💬 Research Chat")

if not st.session_state.get("token"):
    st.warning("Please log in on the Home page first.")
    st.stop()

_init_session_state()
_render_sidebar()
_render_chat_history()
_handle_chat_input()
