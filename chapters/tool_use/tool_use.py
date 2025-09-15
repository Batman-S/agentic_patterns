import os
import asyncio
import nest_asyncio
from typing import List
from dotenv import load_dotenv
import logging

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool as langchain_tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()

try:
    llm = ChatOpenAI(temperature=0)
    print(f"Language model initialized: {llm.model_name}")
    
except Exception as e:
    print(f"Error initializing language model: {e}")
    
@langchain_tool
def search_information(query: str) -> str:
    """
    Provides factual information on a given topic. Use this tool to find answers to phrases like 'capital of japan', 'weather in london', 'population of earth', 'tallest mountain', etc.

    Args:
        query (str): _description_

    Returns:
        str: _description_
    """

    print(f"\n--- Tool Called: search_information with query: '{query}' ---")

    simulated_results = {
        "weather in london": "The weather in London is sunny.",
        "capital of japan": "The capital of Japan is Tokyo.",
        "population of earth": "The population of Earth is 7.9 billion.",
        "tallest mountain": "The tallest mountain is Mount Everest.",
        "default": "No information found for the query.",
    }

    result = simulated_results.get(query.lower(), simulated_results["default"])
    print(f"--- TOOL RESULT: {result} ---")
    return result

tools = [search_information]

if llm:
    agent_prompt = ChatPromptTemplate.from_messages([
       ("system", "Youa re a helpful assistant."),
       ("human", "{input}"),
       ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, agent_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

async def run_agent_with_tool(query: str):
    print(f"\n--- Running Agent with Query:'{query}' ---")
    try:
        response = await agent_executor.ainvoke({"input": query})
        print("\n--- Final Agent Response ---")
        print(response["output"])
    except Exception as e:
        print(f"\n--- Error: {e} ---")
    
async def main():
    tasks = [
        run_agent_with_tool("What is the capital of Japan?"),
        run_agent_with_tool("What is the weather in London?"),
        run_agent_with_tool("Tell me something about dogs"),
    ]
    await asyncio.gather(*tasks)

nest_asyncio.apply()
asyncio.run(main())