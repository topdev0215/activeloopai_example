from langchain.embeddings.openai import OpenAIEmbeddings 
from langchain.text_splitter import CharacterTextSplitter 
from langchain.document_loaders import TextLoader 
from langchain.vectorstores import DeepLake 
import os 

os.environ['OPENAI_API_KEY'] = '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' 

source_text = "https://github.com/activeloopai/examples/raw/main/colabs/starting_data/paul_graham_essay.txt" 
# Example example below. Switch from curl to wget if using Linux 
# !curl -LO https://github.com/activeloopai/examples/raw/main/colabs/starting_data/paul_graham_essay.txt --output "paul_graham_essay.txt" 
# source_text = "paul_graham_essay.txt" 

dataset_path = "hub://hirakawa0215/text_embedding" 

documents = TextLoader(source_text).load() 
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0) 
docs = text_splitter.split_documents(documents) 

db = DeepLake.from_documents(docs, dataset_path=dataset_path, embedding=OpenAIEmbeddings())