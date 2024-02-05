import requests
from newspaper import Article
from langchain.schema import(
    HumanMessage
)
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import validator, BaseModel, Field
from typing import List
from langchain.prompts import PromptTemplate

import os

# Set OpenAI api key to environment
openai_api_key = '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'
os.environ['OPENAI_API_KEY'] = openai_api_key

# get article by newspaper3k

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
}

article_url = "https://www.gemini.com/blog/spot-bitcoin-etfs-start-trading-ether-on-the-rise"

session = requests.Session()

try:
    response = session.get(article_url, headers=headers, timeout=10)
    if response.status_code == 200:
        article = Article(article_url)
        article.download()
        article.parse()
        article_title = article.title
        article_text = article.text
        # print(f"Title: {article.title}")
        # print(f"Text: {article.text}")
    else:
        print(f"failed to fetch article at {article_url}")
except Exception as e:
    print(f"Error occurred while fetching article at {article_url}: {e}")

# create output parser
class ArticleSummary(BaseModel):
    title: str = Field(description='Title of the article')
    summary: List[str] = Field(description='Bulleted list summary of the article')
    
    #validating whether the summary has at list three lines
    @validator("summary", allow_reuse=True)
    def has_three_or_more_lines(cls, list_of_lines):
        if len(list_of_lines) < 3:
            raise ValueError("Generated summary has less than three bullet points!")
        return list_of_lines

#setup output parser
parser = PydanticOutputParser(pydantic_object=ArticleSummary)
    
    
# prepare template for prompt

template = """
As an advanced AI, you've been tasked to summarize online articles into bulleted points. Here are a few examples of how you've done this in the past:

Example 1:
Original Article: 'The Effects of Climate Change
Summary:
- Climate change is causing a rise in global temperatures.
- This leads to melting ice caps and rising sea levels.
- Resulting in more frequent and severe weather conditions.

Example 2:
Original Article: 'The Evolution of Artificial Intelligence
Summary:
- Artificial Intelligence (AI) has developed significantly over the past decade.
- AI is now used in multiple fields such as healthcare, finance, and transportation.
- The future of AI is promising but requires careful regulation.

Now, here's the article you need to summarize:

==================
Title: {article_title}
Text: {article_text}
==================

Please provide a summarized version of the article in a bulleted list format.

{format_instructions}
"""

# Format the prompt
prompt = PromptTemplate(
    template=template,
    input_variables=["article_title", "article_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Formate the prompt using the article title and text from scraping
formatted_prompt = prompt.format_prompt(article_title=article_title, article_text=article_text)

print(formatted_prompt.to_string())

# instantiate model class
model = ChatOpenAI(model_name ='gpt-3.5-turbo', temperature=0.0)

# use model to generate summary
output = model.invoke(formatted_prompt.to_string())
print(output)

# this is the sample output
# output = '{"title": "Spot Bitcoin ETFs Start Trading, Ether Is on the Rise With its Own ETF Conversation Starting, Inflation Ticks Up in December as Markets Await Interest Rate Cuts","summary": ["Spot bitcoin ETFs approved by the SEC began trading on US exchanges, marking a historic day for bitcoin and crypto.","BTC rose to $49k before paring back gains to sit below $45k by Friday morning, with over $4.6 billion traded across all spot bitcoin ETFs on Thursday.","Ether surged as focus shifted to potential ether ETF, with the ETHBTC pair rallying higher and currently trading near 0.06 as of Friday morning.","Inflation ticked up in December, reflecting a 3.4% yearly increase, potentially complicating the interest rate outlook for 2024.","Altcoins performed well as total crypto market cap neared $1.8 trillion, with Ethereum Classic (ETC) trading up more than 55% over the past seven days and Lido DAO (LDO) adding 20% over the same period.","Circle, USDC issuer, filed for an initial public offering (IPO) with the SEC, seeking to launch shares for public sale."]}'
# Parse the output into the pydantic model
# parsed_output = parser.parse(output)
# print(parsed_output.summary)