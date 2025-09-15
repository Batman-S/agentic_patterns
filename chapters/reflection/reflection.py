import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

def run_reflection_loop():
    task_prompt = """
    Your task is to create a Python function named 'calculate_factorial'.
    This function should do the following:
    1. Accept a single integer 'n' as input.
    2. Calculate its factorial (n!).
    3. Include a clear docstring explaining what the function does.
    4. Handle edge cases: The factorial of 0 is 1.
    5. Handle invalid input: Raise a ValueError if the input is a negative number.
    """
    
    max_iterations = 3
    current_code = ""
    
    message_history = [HumanMessage(content=task_prompt)]
    
    for i in range(max_iterations):
        print("\n" + "="*25 + f"REFLECTION LOOP: ITERATION {i+1} "
              + "="*25)
        if i == 0:
            print("\n>>> STAGE 1: GENERATING initial code...")
            response = llm.invoke(message_history)
            current_code = response.content
        else:
            print("\n>>> STAGE 1: REFINING code based on previous critique...")
            message_history.append(HumanMessage(content="Please refine the code using the critiques provided."))
            response = llm.invoke(message_history)
            current_code = response.content
        
        print("\n --- Generated Code (v" + str(i + 1) + ") ---\n" + current_code)
        
        message_history.append(response)
        
        # Reflect Stage
        print("\n>>> STAGE 2: REFLECTING on the generated code...")
        
        reflector_prompt = [
            SystemMessage(content="""
                You are a senior software engineer and an expert in Python.
                Your role is to perform a meticulous code review.
                Critically evaluate the provided Python code based
                on the original task requirements.
                If the code is perfect and meets all requirements,
                respond with the single phrase 'CODE_IS_PERFECT'.
                Otherwise, provide a bulleted list of your critiques.    
            """),
            HumanMessage(content=f"Original Task: \n{task_prompt}\n\nCode to Review: \n{current_code}")
        ]
        critique_response = llm.invoke(reflector_prompt)
        critique = critique_response.content
        
        if "CODE_IS_PERFECT" in critique:
            print("\n--- CODE IS PERFECT! ---")
            break
            
        print("\n--- CODE REVIEW CRITIQUES ---\n" + critique)

        message_history.append(HumanMessage(content=f"Critique of the previous code: \n{critique}"))
    
    print("\n" + "="*30 + " FINAL RESULT " + "="*30 + "\n" + current_code)

if __name__ == "__main__":
    run_reflection_loop()