# -*- coding:utf-8 -*-
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import time
import api_key as api
import requests

api_key = api.ai302_api_key()
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

langchain_tools = [
    get_exchange_rate_from_api
]

model = ChatOpenAI(
    base_url=base_url,
    api_key=api_key,
    model_name=model_name
)

# Different types of memory can be found in Langchain
memory = InMemoryChatMessageHistory(session_id="foo")
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

def agent_test():
    agent = create_tool_calling_agent(model, langchain_tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=langchain_tools)
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: memory,
        input_messages_key="input", 
        history_messages_key="chat_history",
    )
    # 使用带记忆的agent
    output = agent_with_chat_history.invoke(
        {"input": "What is the current exchange rate for USD vs EUR ?"},
        config=config
    )
    print(output)
    # 解析大模型输出结果
    parsed_output = StrOutputParser().parse(output['output'])
    print(parsed_output)

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