from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool
import os

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

filings = Chroma(
    database=os.getenv("CHROMADB"),
    collection_name="company_filings", 
    embedding_function=embeddings,
    chroma_cloud_api_key=os.getenv("CHROMADB_API_KEY"),
    tenant=os.getenv("CHROMADB_TENANT"),
)

parser = Chroma(
    database=os.getenv("CHROMADB"),
    collection_name="parser_data", 
    embedding_function=embeddings,
    chroma_cloud_api_key=os.getenv("CHROMADB_API_KEY"),
    tenant=os.getenv("CHROMADB_TENANT"),
)

news = Chroma(
    database=os.getenv("CHROMADB"),
    collection_name="news_articles", 
    embedding_function=embeddings,
    chroma_cloud_api_key=os.getenv("CHROMADB_API_KEY"),
    tenant=os.getenv("CHROMADB_TENANT"),
)

financials = Chroma(
    database=os.getenv("CHROMADB"),
    collection_name="financial_data", 
    embedding_function=embeddings,
    chroma_cloud_api_key=os.getenv("CHROMADB_API_KEY"),
    tenant=os.getenv("CHROMADB_TENANT"),
)

model = init_chat_model("gpt-4o", model_provider="openai")


def analyze(query, collection, k=10):
    try:
        res = collection.similarity_search(query=query, k=k)

        # Truncate chunks to ~1500 characters to prevent context overflow with high k
        truncated_res = []
        for r in res:
            text = r.page_content if len(r.page_content) <= 1500 else r.page_content[:1500] + "... [truncated]"
            truncated_res.append(f"Content: {text} | Metadata: {r.metadata}")

        messages = [SystemMessage(content="You are a professional technical financial analyst."),
                    HumanMessage(content=f"Summarize the following data: {truncated_res} . Do not repeat yourself"),
                    ]
        
        return model.invoke(messages).content
    except Exception as e:
        return f"Error while running tool: {e}"

@tool
def analyze_filings(query):
    """
    Fetches info relevant to what you would find in a 10-K or 10-Q filing of a company using the data fetched in the database
    Takes just one string as an argument 
    """
    return analyze(query, filings, k=15)

@tool
def analyze_news(query):
    """
    Fetches info relevant to what you would find in the news about a company using the data fetched in the database
    Takes just one string as an argument 
    """
    return analyze(query, news, k=10)

@tool
def analyze_parser(query):
    """
    Fetches comprehensive parsing data like 10-K extraction items and specific financial sections from parsed filings for a given company.
    Takes just one string as an argument.
    """
    return analyze(query, parser, k=15)

@tool
def analyze_financials(query):
    """
    Fetches info about a company's ticker data  
    Takes just one string as an argument 
    """
    return analyze(query, financials)

from social_sentiment_loader import analyze_social_sentiment

@tool
def analyze_social_sentiment_tool(query):
    """
    Fetches social media sentiment metrics, post volume, and trending topics for a stock ticker.
    Takes just one string as an argument (the company ticker).
    """
    return str(analyze_social_sentiment(query))

