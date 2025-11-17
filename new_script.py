import streamlit as st
import os
import json
import re
import requests
from urllib.parse import quote
from typing import TypedDict, List, Dict, Any

# LangGraph
from langgraph.graph import StateGraph, START, END

# Existing Chroma DB (LangChain community wrapper)
from langchain_community.vectorstores import Chroma

# Google Search & Wikipedia Tools
from langchain_community.utilities import WikipediaAPIWrapper, GoogleSearchAPIWrapper
from langchain_community.tools import WikipediaQueryRun, GoogleSearchRun

# Google Gemini (direct API)
from google.generativeai import configure, GenerativeModel


# ======================================================
# 🔐 LOAD API KEYS FROM STREAMLIT SECRETS
# ======================================================
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
GOOGLE_CSE_ID = st.secrets["GOOGLE_CSE_ID"]


# os.environ["google_api_key"] = "AIzaSyBMSTBqYv74VqltxMj7G8eUtbuQg8tUROg"
# os.environ["google_cse_id"] = "94a6404e7eb494900"


# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")



configure(api_key=GOOGLE_API_KEY)
gemini = GenerativeModel("gemini-2.5-flash")


# ======================================================
# 📁 LOAD EXISTING CHROMA DB (NO EMBEDDINGS NEEDED)
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# PERSIST_DIR_1 = os.path.join(BASE_DIR, "chroma_db_nomic")
# PERSIST_DIR_2 = os.path.join(BASE_DIR, "chroma_db_jsonl")

# vectorstore already contains embeddings → do NOT pass embedding_function
# db1 = Chroma(persist_directory=PERSIST_DIR_1)
# db2 = Chroma(persist_directory=PERSIST_DIR_2)

PERSIST_DIR_1 = os.path.join(BASE_DIR, "small_vector_db")

db1 = Chroma(persist_directory=PERSIST_DIR_1)


retriever1 = db1.as_retriever(search_kwargs={"k": 6})
# retriever2 = db2.as_retriever(search_kwargs={"k": 6})


# ======================================================
# 🌐 SEARCH TOOLS
# ======================================================
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

google_tool = GoogleSearchRun(
    api_wrapper=GoogleSearchAPIWrapper(
        google_api_key=GOOGLE_API_KEY,
        google_cse_id=GOOGLE_CSE_ID,
    )
)


# ======================================================
# 💾 CHAT MEMORY
# ======================================================
MEMORY_FILE = "chat_memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            return json.load(open(MEMORY_FILE, "r", encoding="utf-8"))
        except:
            return []
    return []


# FIX: Normalize memory to dict format
def normalize_chat(mem):
    fixed = []
    for item in mem:
        if isinstance(item, dict):
            fixed.append(item)
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            fixed.append({"query": item[0], "answer": item[1]})
        else:
            continue
    return fixed


def save_memory(mem):
    json.dump(mem[-15:], open(MEMORY_FILE, "w", encoding="utf-8"), indent=2)


# ======================================================
# 🧰 UTILITY FUNCTIONS
# ======================================================
def clean_query(q: str) -> str:
    return re.sub(r"[\n\r]+", " ", q.strip())


def ask_gemini(prompt: str) -> str:
    try:
        response = gemini.generate_content(prompt)
        return response.text
    except:
        return "Gemini API Error."


def extractive_answer(query: str, docs: List[Any]) -> str:
    if not docs:
        return ""

    ctx = "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs[:4]))

    prompt = f"""
Use ONLY the numbered CONTEXT below to answer.

Every sentence MUST end with a citation like [1], [2], [3].

If context is insufficient, return "NOINFO".

Question: {query}

CONTEXT:
{ctx}
"""

    ans = ask_gemini(prompt)
    if ans.startswith("NOINFO") or len(ans) < 40:
        return ""
    return ans


def scholarly_lookup(query: str, max_results=3):
    refs = []
    try:
        r = requests.get(
            f"https://api.crossref.org/works?rows={max_results}&query={quote(query)}",
            timeout=8
        ).json()

        for item in r.get("message", {}).get("items", []):
            title = item.get("title", ["Untitled"])[0]
            authors = item.get("author", [])
            author_str = ", ".join(a.get("family", "") for a in authors[:2]) or "Unknown"
            if len(authors) > 2:
                author_str += " et al."
            year = item.get("issued", {}).get("date-parts", [[None]])[0][0]
            doi = item.get("DOI", "")
            link = f"https://doi.org/{doi}" if doi else item.get("URL", "")
            refs.append(f"{author_str} ({year}). *{title}*. {link}")

        return refs or ["(No scholarly reference found)"]
    except:
        return ["(No scholarly reference found)"]


# ======================================================
# 🔀 GRAPH WORKFLOW NODES
# ======================================================
class GraphState(TypedDict):
    query: str
    answer: str
    context: str
    citations: List[str]
    chat_history: List[Dict[str, str]]


def db1_node(state: GraphState) -> GraphState:
    q = clean_query(state["query"])
    docs = retriever1.invoke(q)
    ans = extractive_answer(q, docs)
    return {**state, "context": "DB1" if ans else "", "answer": ans}


# def db2_node(state: GraphState) -> GraphState:
#     q = clean_query(state["query"])
#     docs = retriever2.invoke(q)
#     ans = extractive_answer(q, docs)
#     return {**state, "context": "DB2" if ans else "", "answer": ans}


def google_node(state: GraphState) -> GraphState:
    try:
        res = google_tool.invoke({"query": state["query"]})
        ans = ask_gemini(f"Summarize this Google result:\n{res}")
        return {**state, "context": "Google", "answer": ans}
    except:
        return state


def wiki_node(state: GraphState) -> GraphState:
    try:
        res = wiki_tool.invoke({"query": state["query"]})
        ans = ask_gemini(f"Summarize this Wikipedia text:\n{res}")
        return {**state, "context": "Wikipedia", "answer": ans}
    except:
        return state


def final_node(state: GraphState) -> GraphState:
    q = clean_query(state["query"])
    final_answer = state["answer"] or ask_gemini(q)
    refs = scholarly_lookup(q)

    state["citations"] = refs
    state["answer"] = f"{final_answer}\n\n**References:**\n" + "\n".join(refs)
    return state


# ======================================================
# 🔧 BUILD WORKFLOW GRAPH
# ======================================================
workflow = StateGraph(GraphState)

workflow.add_node("db1", db1_node)
# workflow.add_node("db2", db2_node)
workflow.add_node("google", google_node)
workflow.add_node("wiki", wiki_node)
workflow.add_node("final", final_node)

workflow.add_edge(START, "db1")
workflow.add_conditional_edges("db1", lambda s: bool(s["answer"]), {"true": "final", "false": "google"})
# workflow.add_conditional_edges("db2", lambda s: bool(s["answer"]), {"true": "final", "false": "google"})
workflow.add_conditional_edges("google", lambda s: bool(s["answer"]), {"true": "final", "false": "wiki"})
workflow.add_edge("wiki", "final")

graph = workflow.compile()


# ======================================================
# 🎨 STREAMLIT UI
# ======================================================
st.set_page_config(page_title="Hybrid RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Hybrid RAG + Google + Wikipedia Chatbot")

# Load + normalize chat history
if "chat" not in st.session_state:
    st.session_state.chat = normalize_chat(load_memory())
else:
    st.session_state.chat = normalize_chat(st.session_state.chat)

user_input = st.text_input("Ask me anything:")

if st.button("Submit"):
    if user_input.strip():
        mem = st.session_state.chat

        result = graph.invoke({
            "query": user_input,
            "answer": "",
            "context": "",
            "citations": [],
            "chat_history": mem,
        })

        st.write("### Response")
        st.write(result["answer"])
        st.write(f"**Source:** `{result['context']}`")

        # Save new memory
        mem.append({"query": user_input, "answer": result["answer"]})
        save_memory(mem)
        st.session_state.chat = mem


# Display history safely
st.write("---")
st.write("### Recent Chat History")

for c in st.session_state.chat[-10:]:
    query = c.get("query", str(c))
    answer = c.get("answer", "")
    st.markdown(f"**You:** {query}")
    st.markdown(f"**Bot:** {answer}")
