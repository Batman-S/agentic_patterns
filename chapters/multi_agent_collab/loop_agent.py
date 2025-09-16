import asyncio
from typing import AsyncGenerator
from google.adk.agents import BaseAgent, LlmAgent, LoopAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    return LiteLlm(
        model="openai/gpt-4o-mini",
        temperature=0.0,
        )
    
class ConditionChecker(BaseAgent):
    name: str = "ConditionChecker"
    description: str = "Checks if a process is complete and signals the loop to stop."

    async def _run_async_impl(
        self, context: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        status = context.session.state.get("status", "pending")
        is_done = (status == "completed")
        if is_done:
            yield(Event(author=self.name, actions=EventActions(escalate=True))) 
        else:
            yield Event(author=self.name, content="Condition not met, continuing loop.")
     
process_step = LlmAgent(
    name="ProcessingStep",
    model=get_llm(),
    instruction="Youa re a step in a longer process. Perform your task. If you are the final step, update session state by setting 'status' to 'completed'."
)

poller = LoopAgent(
    name="StatusPoller",
    max_iterations=10,
    sub_agents=[
        process_step,
        ConditionChecker()
    ]
)