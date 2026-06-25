import os
import uuid
from typing_extensions import TypedDict, Annotated
from typing import Literal
import operator
from dotenv import load_dotenv

from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

# Load the variables from the .env file into the environment
load_dotenv()
# Access the key using os.environ
google_api_key = os.environ.get("GOOGLE_API_KEY")
qdrant_url = os.environ.get("QDRANT_URL")
api_key_qdrant = os.environ.get("QDRANT_API_KEY")


# Step 1: Define Chat model, embedding model, vector store, and RAG tool

# Chat model
model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    api_key=google_api_key,
    temperature=0.4,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Connect to the already-populated Qdrant vector store
qdrant = QdrantVectorStore.from_existing_collection(
    collection_name="iphone_user_guide",
    embedding=embeddings,
    url=qdrant_url,
    prefer_grpc=True,
    api_key=api_key_qdrant,
)

# RAG Tool
@tool
def search_document(query: str) -> str:
    """
    Search the document for information related to the user's query.
    Always use this tool to retrieve information before answering any question.
    """
    results = qdrant.similarity_search(query, k=5)
    
    if not results:
        return "No relevant information found in the document."

    # Format the results to explicitly inject metadata (Page, Chapter, Section) into the LLM's context window
    formatted_results = []
    for res in results:
        page = res.metadata.get('page', 'Unknown Page')
        chapter = res.metadata.get('Chapter', '')
        section = res.metadata.get('Section', '')

        # Build the citation string dynamically based on available metadata
        citation = f"Page {page}"
        
        if chapter:
            citation += f", Chapter: {chapter}"

        if section:
            citation += f", Section: {section}"
            
        formatted_results.append(f"--- CONTENT SOURCE: [{citation}] ---\n{res.page_content}\n")

    return "\n\n".join(formatted_results)


# Augment the LLM with tools
tools = [search_document]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)


# Step 2: Define state
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

# Step 3: Define model node
def llm_call(state: MessagesState):
    """LLM decides whether to call a tool or not"""
    
# Strict System Prompt to enforce assessment constraints
    system_prompt = """You are an expert assistant designed to answer questions strictly based on the provided document, the "iPhone User Guide For iOS 7.1 Software", which contains all the information related to this iPhone.

Your behavior MUST adhere to the following rules:
1. ALWAYS call the `search_document` tool to retrieve information before answering any question about the iphone guide to the user.
2. If the retrieved context does not contain the answer, you MUST explicitly state: "I cannot find the answer to this in the provided document."
3. NEVER fabricate information, guess, make assumptions, or rely on your general knowledge to answer any question about the iphone guide.
4. Every response MUST include a citation referencing the specific Page and Chapter/Section where the information was found in the retrieved context (e.g., [Page 5, Chapter: Setup]). 
5. CITATION FORMATTING: Consolidate your citations. If multiple sentences in a row or an entire paragraph come from the exact same Page and Chapter, only put the citation ONCE at the very end of that paragraph. Do not repeat the exact same citation multiple times.

Do not break these rules under any circumstances."""

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content=system_prompt
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

    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
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