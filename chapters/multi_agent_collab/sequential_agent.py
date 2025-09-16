import asyncio
from typing import AsyncGenerator
from google.adk.agents import Agent, BaseAgent, LlmAgent, LoopAgent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    return LiteLlm(
        model="openai/gpt-4o-mini",
        temperature=0.0,
        )

step1 = Agent(name="Step1_Fetch", output_key="data")
step2 = Agent(
    name="Step2_Process", 
    instruction="Analyze the information found in state['data'] and provide a summary."
    )
pipeline = SequentialAgent(
    name="MyPipeline",
    sub_agents=[step1, step2],
)