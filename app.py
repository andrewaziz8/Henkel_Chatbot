import streamlit as st
import uuid
from langchain.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.types import Command

# Import the compiled agent from your main.py
from main import agent

# Configure the Streamlit page
st.set_page_config(page_title="LangGraph Math Assistant", page_icon="🤖")
st.title("🤖 LangGraph Agent with Human-in-the-Loop")

# --- 1. Initialize Session State ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Track if the graph is currently paused waiting for human review
if "is_interrupted" not in st.session_state:
    st.session_state.is_interrupted = False

# Persistent configuration for the LangGraph Checkpointer
config = {"configurable": {"thread_id": st.session_state.thread_id}}

# --- 2. Render Chat History ---
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)

    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            if msg.content:
                st.write(msg.content)
            # Display tool calls if the model decides to use them
            if msg.tool_calls:
                for tool in msg.tool_calls:
                    st.info(f"🛠️ Calling Tool: `{tool['name']}` with args `{tool['args']}`")

    elif isinstance(msg, ToolMessage):
        # "tool" isn't a built-in chat_message role with a default avatar,
        # so give it an explicit emoji avatar to keep it visually distinct.
        with st.chat_message("tool", avatar="🔧"):
            st.success(f"**Result:** {msg.content}")

# --- 3. Handle User Input ---
if user_input := st.chat_input("Type your message here..."):

    # Display the immediate user input
    with st.chat_message("user"):
        st.write(user_input)

    with st.spinner("Agent is processing..."):
        # Route execution based on whether we are resuming an interrupt or starting fresh
        if st.session_state.is_interrupted:
            # Resume the graph with the human's review input
            response_state = agent.invoke(Command(resume=user_input), config=config)
        else:
            # Standard invocation
            user_msg = HumanMessage(content=user_input)
            response_state = agent.invoke({"messages": [user_msg]}, config=config)

        # Update our session state with the full message history from the graph
        st.session_state.messages = response_state["messages"]

        # --- 4. Check for Interrupts ---
        if "__interrupt__" in response_state and response_state["__interrupt__"]:
            st.session_state.is_interrupted = True
            # __interrupt__ is a tuple/list of Interrupt objects; grab the value
            # that was actually passed to interrupt() in review_node.
        else:
            st.session_state.is_interrupted = False

    # Rerun the script to sync the UI with the updated state
    st.rerun()