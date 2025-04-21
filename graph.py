from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langchain_core.runnables import Runnable
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from agents import research_agent

# Define the State type
class ResearchState(TypedDict):
    query: str
    search_result: Annotated[str, "The research results from Tavily"]
    answer: Annotated[str, "The final answer"]

# Initialize the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Define the prompt template
prompt = PromptTemplate.from_template("""
You are a highly intelligent research analyst. Using the information below, write a structured and cited response.

Context:
{context}

Question:
{question}
""")

# Create the answer chain
answer_chain = LLMChain(llm=llm, prompt=prompt)

def run_research(query):
    return research_agent.run(query)

def summarize(data, query):
    return answer_chain.run({"context": data, "question": query})

def build_graph():
    workflow = StateGraph(ResearchState)

    # Add nodes
    workflow.add_node("search", lambda state: {
        "search_result": run_research(state["query"])
    })
    
    workflow.add_node("summarize", lambda state: {
        "answer": summarize(state["search_result"], state["query"])
    })

    # Set up edges
    workflow.set_entry_point("search")
    workflow.add_edge("search", "summarize")
    workflow.add_edge("summarize", END)

    return workflow.compile()