import uuid

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
import requests

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

from typing import Annotated, Literal, TypedDict

from db_utils import get_chat_history
from langchain_utils import get_rag_chain
from models import QueryInput

@tool
def dog_finder_tool() -> []:
    """finds a bunch of dogs from the api"""
    url = 'https://api.thecatapi.com/v1/breeds'
    headers = {
        'x-api-key': 'API_KEY'
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        dog_names = [dog['name'] for dog in data]
        return dog_names
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        return []


@tool
def chat_tool(query_input: QueryInput):
    """Helps with CVS chat, insurance q and a , general default tool , model value is gemini-1.5-flash"""
    session_id = query_input.session_id
    if not session_id:
        session_id = str(uuid.uuid4())

    chat_history = get_chat_history(session_id)

    rag_chain = get_rag_chain(query_input.model.value)

    answer = rag_chain.invoke({
        "input": query_input.question,
        "chat_history": chat_history
    })['answer']

    return answer


llm = ChatVertexAI(
        model="gemini-1.5-flash",
        project = "my-project-1509111052665"
)

############################################################################################################
# Using Agents with tools
############################################################################################################

# tools = [dog_finder_tool, chat_tool]
# llm_with_tools = llm.bind_tools(tools)
# tool_map = {tool.name: tool for tool in tools}
#
#
# def call_tools(msg: AIMessage) -> Runnable:
#     """Simple sequential tool calling helper."""
#     tool_calls = msg.tool_calls.copy()
#     for tool_call in tool_calls:
#         tool_call["output"] = tool_map[tool_call["name"]].invoke(tool_call["args"])
#     return tool_calls
#
#
# chain = llm_with_tools | call_tools
#
# res = chain.invoke("find dogs ?")
# print("******** RESULT of prompt : find dogs ? ***********")
# print(res[0]['output'])
#
# res = chain.invoke("who owns insurance module ?")
# print("******** RESULT of prompt : who owns insurance module ? ***********")
# print(res[0]['output'])


############################################################################################################
# Using Agents with langgraph
############################################################################################################
tools = [dog_finder_tool, chat_tool]

tool_node  = ToolNode(tools)

llm_with_tools = llm.bind_tools(tools)

# Define the function that determines whether to continue or not
def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return END


# Define the function that calls the llm
def call_llm(state: MessagesState):
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

# Define a new graph
workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_llm)
workflow.add_node("tools", tool_node)



# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.add_edge(START, "agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("tools", 'agent')


# Initialize memory to persist state between graph runs
checkpointer = MemorySaver()

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable.
# Note that we're (optionally) passing the memory when compiling the graph
app = workflow.compile(checkpointer=checkpointer)

# Use the Runnable
final_state = app.invoke(
    {"messages": [HumanMessage(content="who owns insurance module ?")]},
    config={"configurable": {"thread_id": 42}}
)

print(final_state["messages"][-1].content)

final_state = app.invoke(
    {"messages": [HumanMessage(content="how many tables are there in this module ?")]},
    config={"configurable": {"thread_id": 42}}
)

print(final_state["messages"][-1].content)


final_state = app.invoke(
    {"messages": [HumanMessage(content="can you find some dogs for me ?")]},
    config={"configurable": {"thread_id": 42}}
)

print(final_state["messages"][-1].content)
