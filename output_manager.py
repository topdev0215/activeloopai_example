from langchain.output_parsers import PydanticOutputParser, RetryWithErrorOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List
from langchain_openai import OpenAI

import os

# Set OpenAI api key to environment
openai_api_key = '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'
os.environ['OPENAI_API_KEY'] = openai_api_key

# set chatgpt model
model = OpenAI(model_name='gpt-3.5-turbo-instruct', temperature=0.0)

# Define your desired data structure.
class Suggestions(BaseModel):
    words: List[str] = Field(description="list of substitue words based on context")
    reasons: List[str] = Field(description="the reasoning of why this word fits the context")
# set parser
parser = PydanticOutputParser(pydantic_object=Suggestions)
# set retry parser
retry_parser = RetryWithErrorOutputParser.from_llm(parser=parser, llm=model)

template = """
Offer a list of suggestions to substitue the specified target_word based the presented context.
{format_instructions}
target_word={target_word}
context={context}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=['target_word', 'context'],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

model_input = prompt.format_prompt(
    target_word='behaviour',
    context='The behaviour of the students in the classroom was disruptive and made it difficult for the teacher to conduct the lesson.'
)

output = model.invoke(model_input.to_string())

# if the error is detected during parsing, retry parser is operated with llm model to fix error
try:
    parser.parse(output)
except:
    output = retry_parser.parse_with_prompt(output, model_input)
    
print(output)