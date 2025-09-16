from langchain_community.chat_models import ChatLiteLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch

from dotenv import load_dotenv
from pprint import pformat
from functools import wraps

load_dotenv()

try:
    llm = ChatLiteLLM(model="gpt-4o-mini", temperature=0)
    print(f"Language model initialized: {llm}")
except Exception as e:
    print(f"Error initializing language model: {e}")
    
def log_returns(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        try:
            pretty = pformat(result, width=100, compact=False)
        except Exception:
            pretty = str(result)
        print(f"\n↩︎ Return from {func.__name__}:\n{pretty}")
        return result
    return wrapper

@log_returns
def booking_handler(request: str) -> str:
    print("\n--- DELEGATING TO BOOKING HANDLER ---")
    return f"Booking Handliner processed request: '{request}'. Result: Simulatied booking action."

@log_returns
def info_handler(request: str) -> str:
    print("\n--- DELEGATING TO INFO HANDLER ---")
    return f"Info Handler processed request: '{request}'. Result: Simulated info action."

@log_returns
def unclear_handler(request: str) -> str:
    print("\n--- DELEGATING TO UNCLEAR HANDLER ---")
    return f"Coordinator could not delegate request: '{request}'. Please clarify."

# Coordinator Router Chain
coordinator_router_prompt = ChatPromptTemplate.from_messages([
    (
        "system", """Analyze the user's request and determine which
        specialist handler should process it.
        - If the request is related to booking flights or hotels, output 'booker'.
        - For all other general information questions, output 'info'.
        - If the request is unclear or doesn't fit either category ouput 'unclear'.
        ONLY output one word: 'booker', 'info', or 'unclear'."""
),
    ("user", "{request}"),
])

if llm:
    coordinator_router_chain = coordinator_router_prompt | llm | StrOutputParser()
    
branches = {
    "booker": RunnablePassthrough.assign(output = lambda x: booking_handler(x['request']['request'])),
    "info": RunnablePassthrough.assign(output = lambda x: info_handler(x['request']['request'])),
    "unclear": RunnablePassthrough.assign(output = lambda x: unclear_handler(x['request']['request']))
}

delegation_branch = RunnableBranch(
    (lambda x: x['decision'].strip() == 'booker', branches["booker"]),
    (lambda x: x['decision'].strip() == 'info', branches["info"]),
    branches["unclear"]
)

coordinator_agent = {
    "decision": coordinator_router_chain,
    "request": RunnablePassthrough()
} | delegation_branch | (lambda x: ['output'])

def main(): 
    if not llm:
        print("\nSkipping eecution due to LLM initialization failure.")
        return
    
    print("--- Running with a booking request ---")
    request_a = "Book me a flight to London."
    result_a = coordinator_agent.invoke({"request": request_a})
    print("Final Result A:\n" + pformat(result_a, width=100, compact=False))

    print("\n--- Running with an info request ---")
    request_b = "What is the capital of Japan?"
    result_b = coordinator_agent.invoke({"request": request_b})
    print("Final Result B:\n" + pformat(result_b, width=100, compact=False))
    
    print("\n--- Running with an unclear request ---")
    request_c = "Tell me about the person named squish cheeks"
    result_c = coordinator_agent.invoke({"request": request_c})
    print("Final Result C:\n" + pformat(result_c, width=100, compact=False))

if __name__ == "__main__":
    main()
    