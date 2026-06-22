# Step 1: Define tools and model

import os
import uuid
from typing_extensions import TypedDict, Annotated
from typing import Literal
import operator
from dotenv import load_dotenv

from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

# Load the variables from the .env file into the environment
load_dotenv()
# Access the key using os.environ
api_key = os.environ.get("GROQ_API_KEY")

model = ChatGroq(
    model="qwen/qwen3-32b",
    api_key=api_key,
    temperature=0.4,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # reasoning_format="parsed"
)


# Define RAG tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b


# Augment the LLM with tools
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

# Step 2: Define state

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

# Step 3: Define model node

def llm_call(state: MessagesState):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }


# Step 4: Define tool node and review node for human input interrupts

def tool_node(state: MessagesState):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

# Add this for human input interrupts.
def review_node(state: MessagesState):
    """Review the LLM output and answer it"""

    last_message = state["messages"][-1].content 
    # Pause and show the current content for review (surfaces in result["__interrupt__"])
    # edited_content = interrupt({
    #     "instruction": "Review and reply to this content",
    #     "content": last_message
    # })
    edited_content = interrupt(last_message) # This value will be sent to the client as part of the interrupt information.

    # Update the state with the edited version
    return {"messages": [HumanMessage(content=edited_content)]}


# Step 5: Define logic to determine whether to end 

# Conditional edge function to route to the tool node or review node based upon whether the LLM made a tool call
def should_continue(state: MessagesState) -> Literal["tool_node", "review_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # Check the tool calls first before any human input.
    if last_message.tool_calls:
        return "tool_node"
    
    # If there is text content but no tools, send to human for review.
    if last_message.content:
        return "review_node"
    
    return END

# Step 6: Build agent

# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_node("review_node", review_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", "review_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")
agent_builder.add_edge("review_node", "llm_call")

checkpointer = InMemorySaver()

# Compile the agent
agent = agent_builder.compile(checkpointer=checkpointer)

# # Show the agent
# graph_png = agent.get_graph(xray=True).draw_mermaid_png()
# with open("my_agent_graph.png", "wb") as f:
#     f.write(graph_png)
# print("\nGraph saved as my_agent_graph.png\n")


def main():
    # Invoke
    messages = [HumanMessage(content=input("\nWhat is on your mind for today?\n"))]

    # Initial run - hits the interrupt and pauses
    # thread_id is the persistent pointer (stores a stable ID in production)
    config = {
        "configurable": {
            "thread_id": uuid.uuid4(),
        }
    }

    messages = agent.invoke(
        {"messages": messages},
        config=config
    )

    # Creating a Loop for a dynamic user input response
    while True:
        if "__interrupt__" in messages and messages["__interrupt__"]:
            # Check what was interrupted
            # __interrupt__ contains the payload that was passed to interrupt()
            print(messages["__interrupt__"])

            # Get dynamic user response from the terminal
            user_input = input("\nYour Response: ")

            if user_input.lower() == 'exit':
                break
                
            # Resume the graph with the human's input. The resume payload becomes the return value of interrupt() inside the node.
            messages = agent.invoke(Command(resume=user_input), config=config)

        else:
            break


    for m in messages["messages"]:
        m.pretty_print()

if __name__ == "__main__":
    main()