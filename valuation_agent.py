import os, dcf
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.tools import tool
from analyst_ratings_loader import load_analyst_ratings


load_dotenv()
'''INPUT FACTORS'''
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

parser_data = Chroma(
    database=os.getenv("CHROMADB"),
    collection_name="parser_data",
    embedding_function=embeddings,
    chroma_cloud_api_key=os.getenv("CHROMADB_API_KEY"),
    tenant=os.getenv("CHROMADB_TENANT"))

news_data = Chroma(
    database=os.getenv("CHROMADB"),
    collection_name="news_articles",
    embedding_function=embeddings,
    chroma_cloud_api_key=os.getenv("CHROMADB_API_KEY"),
    tenant=os.getenv("CHROMADB_TENANT"))

model = init_chat_model("gpt-4o", model_provider="openai")


def valuation(company, year):
    query = f"Detailed valuation of {company} in recent times"
    res = parser_data.similarity_search(query=query, k=10)
    res.extend(news_data.similarity_search(query=query, k=10))
    dcf_calculation = dcf.find_dcf(company, year)
    analyst_data = load_analyst_ratings(company)

    messages = [
                SystemMessage(content=f"""
                                You are a professional equity research editor. 
                                I will provide you with a valuation analysis draft of {company} written by a financial analyst. 
                                Your goals are to:
                                1) Identify and correct all quantitative and logical inconsistencies 
                                   (e.g., incorrect interpretation of undervaluation vs. overvaluation, mismatched numbers, or reversed percentages).
                                2) Ensure terminology and metrics are financially accurate, e.g., correct use of “undervalued” vs. “overvalued,” 
                                   clarify “terminal value,” fix dividend/split facts. 
                                3) Explicitly compare our DCF-derived fair value to Wall Street's consensus price target:
                                      - Use the DCF output fields (intrinsic value, current price, undervaluation %)
                                        versus the Street average/low/high targets in the analyst data.
                                      - Quantify the percentage difference between DCF fair value and the Street average target.
                                      - Clearly state whether there is a significant disagreement (e.g., >20–30% difference)
                                        or whether the DCF is broadly in line with the Street.
                                4) Interpret the analyst rating information:
                                      - Use the consensus numeric rating and label (Strong Buy / Buy / Hold / Sell / Strong Sell),
                                        price targets, and the rating_trend signal (trend_label, summary, upgrades/downgrades).
                                      - Treat a “Hold” following recent upgrades (especially from Sell) as more constructive
                                        than a stagnant Hold with no recent changes, and explain this nuance briefly.
                                Preserve the author's tone and structure, but improve clarity and conciseness. 
                                Add brief, inline clarifications (in parentheses) if necessary to explain corrected numbers or terms. 
                                Do not invent new data—adjust logic using only the information given. 
                                Also, lightly enhance transitions and coherence between quantitative and qualitative sections, 
                                but keep the word count within ±10 percent of the original."""), 
                            
                HumanMessage(content=f"""Summarize and analyze the following data. 
                            Keep data recent and give me both qualitative and quantitative measures of valuation: {res}

                            DCF calculation: {dcf_calculation}

                            Analyst ratings and targets: {analyst_data}

                            """)
                ]

    return model.invoke(messages).content

@tool
def valuation_tool(company: str, year: str):
    """
    Returns a valuation of a company in a given year 
    Takes two arguments: the company (as a string), and the year (also as a string, formatted as XXXX)
    """
    return valuation(company, year)

# Example: valuation("Microsoft", "2024")

