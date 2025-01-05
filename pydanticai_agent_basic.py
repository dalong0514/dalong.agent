# -*- coding:utf-8 -*-
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
import random
import api_key as api
import time

api_key = api.ai302_api_key()
base_url= 'https://api.302.ai/v1'
model_name='gemini-2.0-flash-exp'

model = OpenAIModel(  
    model_name = model_name,
    api_key = api_key,
    base_url = base_url
)

agent = Agent(  
    model,
    system_prompt='Be concise, reply with one sentence.',  
)

def agent_test():
    result = agent.run_sync('Where does "hello world" come from?')  
    print(result.data)

if __name__ == '__main__':
    start_time = time.time()
    print('waiting...\n')
    agent_test()
    end_time = time.time()
    print('Time Used: ' + str((end_time - start_time)/60) + 'min')