# -*- coding:utf-8 -*-
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.messages import AIMessage
import time
import api_key as api
import requests
import json

# 添加自定义异常类
class NotApproved(Exception):
    """Exception raised when tool invocations are not approved by user."""
    pass

api_key = api.ai302_api_key()
google_serper_key = api.google_serper_key()
base_url= 'https://api.302.ai/v1'
model_name='gemini-2.0-flash-exp'

@tool
def get_exchange_rate_from_api(currency_from: str, currency_to: str) -> str:
    """
    Return the exchange rate between currencies
    Args:
        currency_from: str
        currency_to: str
    """
    url = f"https://api.frankfurter.app/latest?from={currency_from}&to={currency_to}"
    api_response = requests.get(url)
    data = api_response.json()
    # 返回更友好的格式
    return f"1 {currency_from} = {data['rates'][currency_to]} {currency_to}"

# Create our new search tool here
search = GoogleSerperAPIWrapper(serper_api_key=google_serper_key)
@tool
def google_search(query: str):
    """
    Perform a search on Google
    Args:
        query: the information to be retrieved with google search
    """
    return search.run(query)

langchain_tools = [
    get_exchange_rate_from_api,
    google_search
]

def call_tools(msg: AIMessage) -> list[dict]:
    """Simple sequential tool calling helper."""
    tool_map = {tool.name: tool for tool in langchain_tools}
    tool_calls = msg.tool_calls.copy()
    for tool_call in tool_calls:
        tool_call["output"] = tool_map[tool_call["name"]].invoke(tool_call["args"])
    return tool_calls

# 添加一个函数来格式化工具调用的结果
def format_tool_outputs_string(tool_calls):
    outputs = []
    for tool_call in tool_calls:
        # 检查是否有 output 字段，如果没有则显示参数信息
        if 'output' in tool_call:
            outputs.append(f"{tool_call['name']} result: {tool_call['output']}")
        else:
            # 显示工具名称和输入参数
            args_str = json.dumps(tool_call.get('args', {}), indent=2)
            outputs.append(f"{tool_call['name']} will be called with:\n{args_str}")
    return "\n\n".join(outputs)

def human_approval(msg: AIMessage) -> AIMessage:
    """Responsible for passing through its input or raising an exception.

    Args:
        msg: output from the chat model

    Returns:
        msg: original output from the msg
    """
    for tool_call in msg.tool_calls:
        print(f"I want to use function [{tool_call.get('name')}] with the following parameters :")
        for k,v in tool_call.get('args').items():
            print(" {} = {}".format(k, v))
            
    print("")
    input_msg = (
        f"Do you approve (Y|y)?\n\n"
        ">>>"
    )
    resp = input(input_msg)
    if resp.lower() not in ("yes", "y"):
        print("Tool invocations not approved. Continuing without executing tools.")
        # 返回一个空消息或原始消息，而不是抛出异常
        return AIMessage(content="User declined tool execution", tool_calls=[])
    return msg

model = ChatOpenAI(
    base_url=base_url,
    api_key=api_key,
    model_name=model_name
)

# Different types of memory can be found in Langchain
memory = InMemoryChatMessageHistory()
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        # First put the history
        ("placeholder", "{chat_history}"),
        # Then the new input
        ("human", "{input}"),
        # Finally the scratchpad
        ("placeholder", "{agent_scratchpad}"),
    ]
)
config = {"configurable": {"session_id": "foo"}}

# bind the tools to the LLM
model_with_tools = model.bind_tools(langchain_tools)
output_parser = StrOutputParser()
# build the chain
chain = prompt | model_with_tools | human_approval | call_tools
# chain = prompt | model_with_tools | call_tools | format_tool_outputs_string | output_parser

def agent_test():
    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: memory,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    output = chain_with_history.invoke(
        {"input": "What is the current CHF EUR exchange rate ?"}, config)
    # output = chain_with_history.invoke(
    #     {"input": "What was the result of Rafael Nadal's latest game ?"}, config)
    print(output)

if __name__ == '__main__':
    start_time = time.time()
    print('waiting...\n')
    agent_test()
    end_time = time.time()
    # 改进时间显示
    elapsed_time = end_time - start_time
    if elapsed_time < 60:
        print(f'Time Used: {elapsed_time:.2f} seconds')
    else:
        print(f'Time Used: {elapsed_time/60:.2f} minutes')