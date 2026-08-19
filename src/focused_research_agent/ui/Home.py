"""Landing page + login/registration for the Focused Research Agent UI.

Every backend endpoint now requires a bearer token (see
focused_research_agent.auth). This page is where that token is obtained
and stashed in st.session_state["token"] — every other page
(1_Research, 2_Chat, 3_Report) reads it from there and passes it into
the api_client calls.
"""

import streamlit as st

from focused_research_agent.ui import api_client
from focused_research_agent.ui.exceptions import BackendUnavailableError

st.set_page_config(page_title="Focused Research Agent", layout="centered")
st.title("🔍 Focused Research Agent")

if "token" not in st.session_state:
    st.session_state["token"] = None

if st.session_state["token"]:
    st.success("You're logged in. Use the sidebar to navigate to Research, Chat, or Report.")
    if st.button("Log out"):
        st.session_state["token"] = None
        st.rerun()
else:
    st.markdown("Log in or create an account to use the research agent.")

    login_tab, register_tab = st.tabs(["Log in", "Register"])

    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Log in")

        if submitted:
            try:
                result = api_client.login(email, password)
                if result["success"]:
                    st.session_state["token"] = result["data"]["access_token"]
                    st.rerun()
                else:
                    st.error(result["error"])
            except BackendUnavailableError as e:
                st.error(str(e))

    with register_tab:
        with st.form("register_form"):
            reg_email = st.text_input("Email", key="register_email")
            reg_password = st.text_input(
                "Password (min 8 characters)", type="password", key="register_password"
            )
            reg_submitted = st.form_submit_button("Create account")

        if reg_submitted:
            try:
                result = api_client.register(reg_email, reg_password)
                if result["success"]:
                    st.session_state["token"] = result["data"]["access_token"]
                    st.rerun()
                else:
                    st.error(result["error"])
            except BackendUnavailableError as e:
                st.error(str(e))
