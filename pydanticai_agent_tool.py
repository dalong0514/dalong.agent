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
    deps_type=str,  
    system_prompt=(
        "You're a dice game, you should roll the die and see if the number "
        "you get back matches the user's guess. If so, tell them they're a winner. "
        "Use the player's name in the response."
    ),
)

@agent.tool_plain  
def roll_die() -> str:
    """Roll a six-sided die and return the result."""
    return str(random.randint(1, 6))


@agent.tool  
def get_player_name(ctx: RunContext[str]) -> str:
    """Get the player's name."""
    return ctx.deps

def agent_test():
    dice_result = agent.run_sync('My guess is 4', deps='Anne')  
    print(dice_result.data)

if __name__ == '__main__':
    start_time = time.time()
    print('waiting...\n')
    agent_test()
    end_time = time.time()
    print('Time Used: ' + str((end_time - start_time)/60) + 'min')