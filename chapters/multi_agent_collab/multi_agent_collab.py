import os
from pyexpat import model
from typing import AsyncGenerator, Optional

from dotenv import load_dotenv
from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from typing import AsyncGenerator
from google.adk.models.lite_llm import LiteLlm
load_dotenv()

def get_llm():
    return LiteLlm(
        model="openai/gpt-4o-mini",
        temperature=0.0,
        )

class TaskExecutor(BaseAgent):
    """A specialized agent with custom, non-LLM behavior."""
    name: str = "TaskExecutor"
    description: str = "Executes a predefined task."

    async def _run_async_impl(self, context: InvocationContext) -> AsyncGenerator[Event, None]:
        yield Event(author=self.name, content="Task finished successfully.")
    
   
greeter = LlmAgent (
    name="Greeter",
    model=get_llm(),
    instruction="You are a friendly greeter."
)
task_doer = TaskExecutor()

coordinator = LlmAgent(
    name="Coordinator",
    model=get_llm(),
    description="A coordinator that can greet users and execute tasks.",
    instruction="When asked to greet, delegate to the Greeter. When asked to perform a task, delegate to the Task Executor.",
    sub_agents= [greeter,task_doer]
)

print("Agent hierarchy created successfully")