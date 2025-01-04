import uuid

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
import requests

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

tools = [dog_finder_tool, chat_tool]
llm_with_tools = llm.bind_tools(tools)
tool_map = {tool.name: tool for tool in tools}


def call_tools(msg: AIMessage) -> Runnable:
    """Simple sequential tool calling helper."""
    tool_calls = msg.tool_calls.copy()
    for tool_call in tool_calls:
        tool_call["output"] = tool_map[tool_call["name"]].invoke(tool_call["args"])
    return tool_calls


chain = llm_with_tools | call_tools

res = chain.invoke("find dogs ?")
print("******** RESULT of prompt : find dogs ? ***********")
print(res[0]['output'])

res = chain.invoke("who owns insurance module ?")
print("******** RESULT of prompt : who owns insurance module ? ***********")
print(res[0]['output'])

