from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from pprint import pformat

load_dotenv()

llm = ChatOpenAI(temperature=0)

# Prompt 1
prompt_extract = ChatPromptTemplate.from_template(
    "Extract the technical specifications from the following text:\n\n{text_input}"
)

# Prompt 2
prompt_transform = ChatPromptTemplate.from_template(
    "Transform the following specifications into a JSON object with 'cpu', 'memory', and 'storage' as keys:\n\n{specifications}"
)

# Extraction Chain
extraction_chain = prompt_extract | llm | StrOutputParser()

# Full Chain
full_chain = (
    {"specifications": extraction_chain}
    | prompt_transform
    | llm
    | StrOutputParser()
)

input_text = "The new laptop model features a 3.5 GHz octa-core processor, 16GB of RAM, and 1TB NVMe SSD."

final_result = full_chain.invoke({"text_input": input_text})
print("Final Result:\n" + pformat(final_result, width=100, compact=False))