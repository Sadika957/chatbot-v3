# # =====================================================
# # 🌟 FINAL HYBRID CHATBOT: Local Vector DB + Tools + Summaries + Streamlit UI
# # =====================================================
# import os

# # --- FIX Chroma Rust crash on Windows ---
# os.environ["CHROMA_USE_V2"] = "false"          # Disable the unstable Rust backend
# os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"  # Disable telemetry (avoids thread errors)


# import os, re, json, requests, streamlit as st
# from urllib.parse import quote
# from typing import TypedDict, List, Dict, Any
# from langgraph.graph import StateGraph, START, END
# from langchain_google_genai import ChatGoogleGenerativeAI
# # from langchain_community.vectorstores import Chroma
# from langchain_chroma import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.utilities import WikipediaAPIWrapper, GoogleSearchAPIWrapper
# from langchain_community.tools import WikipediaQueryRun, GoogleSearchRun

# # -----------------------------
# # 🔑 API KEYS
# # -----------------------------
# os.environ["google_api_key"] = "AIzaSyBMSTBqYv74VqltxMj7G8eUtbuQg8tUROg"
# os.environ["google_cse_id"] = "94a6404e7eb494900"


# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")


# # -----------------------------
# # 🤖 LLM (flash-lite)
# # -----------------------------
# gemini = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash-lite",
#     temperature=0,
#     api_key=GOOGLE_API_KEY
# )

# # -----------------------------
# # 🧠 Load small_vector_db
# # -----------------------------
# PERSIST_DIR = r"C:\Users\sadika957\Desktop\chatbot\small_vector_db_"

# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# db_local = Chroma(
#     persist_directory=PERSIST_DIR,
#     embedding_function=embeddings
# )

# retriever_local = db_local.as_retriever(search_kwargs={"k": 8})
# print("✅ Loaded small_vector_db successfully.")

# # -----------------------------
# # 🌐 External Tools
# # -----------------------------
# wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
# google_tool = GoogleSearchRun(api_wrapper=GoogleSearchAPIWrapper(
#     google_api_key=GOOGLE_API_KEY,
#     google_cse_id=GOOGLE_CSE_ID
# ))

# # -----------------------------
# # 🗂️ Memory Storage
# # -----------------------------
# MEMORY_FILE = "chat_memory.json"

# def load_memory():
#     if os.path.exists(MEMORY_FILE):
#         try:
#             return json.load(open(MEMORY_FILE, "r", encoding="utf-8"))
#         except:
#             return []
#     return []

# def save_memory(mem):
#     json.dump(mem[-15:], open(MEMORY_FILE, "w", encoding="utf-8"), indent=2)

# # -----------------------------
# # Utility
# # -----------------------------
# def clean_query(q: str) -> str:
#     return re.sub(r"[\n\r]+", " ", q.strip())

# def extractive_answer(query: str, docs):
#     ctx = "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs[:6]))
#     prompt = f"""
# Answer the question strictly using ONLY the context provided.
# Every sentence must end with [number] referencing chunks.
# If answer not found, respond with NOINFO.

# Question: {query}
# CONTEXT:
# {ctx}
# """
#     ans = gemini.invoke(prompt).content.strip()
#     if ans.upper().startswith("NOINFO") or len(ans) < 30:
#         return ""
#     return ans

# def scholarly_lookup(query: str, max_results=3):
#     citations = []
#     try:
#         r = requests.get(f"https://api.crossref.org/works?rows={max_results}&query={quote(query)}", timeout=8).json()
#         for item in r.get("message", {}).get("items", []):
#             title = item.get("title", [""])[0]
#             year = item.get("issued", {}).get("date-parts", [[None]])[0][0]
#             doi = item.get("DOI", "")
#             link = f"https://doi.org/{doi}" if doi else ""
#             citations.append(f"{title} ({year}). {link}")
#         return citations or ["(No scholarly reference found)"]
#     except:
#         return ["(No scholarly reference found)"]

# def format_clickable_citations(citations):
#     out = []
#     for i, c in enumerate(citations, 1):
#         match = re.search(r'(https?://[^\s]+|doi\.org/[^\s]+)', c)
#         if match:
#             link = match.group(1)
#             out.append(f"[{i}] [{c}]({link})")
#         else:
#             out.append(f"[{i}] {c}")
#     return "\n".join(out)

# # -----------------------------
# # 📚 Graph State
# # -----------------------------
# class GraphState(TypedDict):
#     query: str
#     answer: str
#     context: str
#     citations: List[str]
#     chat_history: List[Dict[str, str]]

# # -----------------------------
# # 🧱 Pipeline Nodes
# # -----------------------------
# def local_db_node(state: GraphState):
#     q = clean_query(state["query"])
#     docs = retriever_local.invoke(q)

#     if not docs:
#         return {**state, "context": "no_local"}

#     ans = extractive_answer(q, docs)
#     if not ans:
#         return {**state, "context": "no_local"}

#     ans += f"\n\n📚 Source: Local Vector DB"
#     return {**state, "answer": ans, "context": "local", "citations": []}

# def google_node(state: GraphState):
#     q = clean_query(state["query"])
#     raw = google_tool.run(q)

#     if not raw:
#         return {**state, "context": "no_google"}

#     ans = gemini.invoke(f"Summarize this Google info:\n{raw}").content.strip()
#     link = f"https://www.google.com/search?q={quote(q)}"
#     ans += f"\n\n📚 Source: [Google]({link})"
#     return {**state, "answer": ans, "context": "google", "citations": [link]}

# def wiki_node(state: GraphState):
#     q = clean_query(state["query"])
#     blob = wiki_tool.run(q)

#     if not blob:
#         return {**state, "context": "no_wiki"}

#     link = f"https://en.wikipedia.org/wiki/Special:Search?search={quote(q)}"
#     ans = gemini.invoke(f"Answer using Wikipedia:\n{blob}").content.strip()
#     ans += f"\n\n📚 Source: [Wikipedia]({link})"
#     return {**state, "answer": ans, "context": "wiki", "citations": [link]}

# def gbif_node(state: GraphState):
#     q = clean_query(state["query"])
#     try:
#         r = requests.get(f"https://api.gbif.org/v1/species/search?q={quote(q)}", timeout=8).json()
#         results = r.get("results", [])
#         if not results:
#             return {**state, "context": "no_gbif"}

#         lines = [
#             f"{it.get('scientificName')} – https://www.gbif.org/species/{it.get('key')}"
#             for it in results[:5]
#         ]
#         link = f"https://www.gbif.org/species/search?q={quote(q)}"
#         ans = "\n".join(lines) + f"\n\n📚 Source: [GBIF]({link})"
#         return {**state, "answer": ans, "context": "gbif", "citations": [link]}
#     except:
#         return {**state, "context": "no_gbif"}

# def inat_node(state: GraphState):
#     q = clean_query(state["query"])
#     try:
#         r = requests.get(f"https://api.inaturalist.org/v1/taxa/autocomplete?q={quote(q)}", timeout=8).json()
#         results = r.get("results", [])
#         if not results:
#             return {**state, "context": "no_inat"}

#         lines = [
#             f"{it.get('name')} – https://www.inaturalist.org/taxa/{it.get('id')}"
#             for it in results[:5]
#         ]
#         link = f"https://www.inaturalist.org/search?q={quote(q)}"

#         ans = "\n".join(lines) + f"\n\n📚 Source: [iNaturalist]({link})"
#         return {**state, "answer": ans, "context": "inat", "citations": [link]}
#     except:
#         return {**state, "context": "no_inat"}

# def final_node(state: GraphState):
#     q = clean_query(state["query"])
#     base = state["answer"]

#     summary_prompt = f"""
# Summarize clearly and factually.

# Question: {q}
# Answer: {base}
# """
#     summary = gemini.invoke(summary_prompt).content.strip()

#     if state.get("citations"):
#         summary += f"\n\n📚 Citations:\n{format_clickable_citations(state['citations'])}"
#     return {**state, "answer": summary}

# # -----------------------------
# # 🔀 Build Graph
# # -----------------------------
# workflow = StateGraph(GraphState)
# workflow.add_node("local", local_db_node)
# workflow.add_node("google", google_node)
# workflow.add_node("wiki", wiki_node)
# workflow.add_node("gbif", gbif_node)
# workflow.add_node("inat", inat_node)
# workflow.add_node("final", final_node)

# workflow.add_edge(START, "local")
# workflow.add_conditional_edges("local", lambda s: s["context"], {"local": "final", "no_local": "google"})
# workflow.add_conditional_edges("google", lambda s: s["context"], {"google": "final", "no_google": "wiki"})
# workflow.add_conditional_edges("wiki", lambda s: s["context"], {"wiki": "final", "no_wiki": "gbif"})
# workflow.add_conditional_edges("gbif", lambda s: s["context"], {"gbif": "final", "no_gbif": "inat"})
# workflow.add_edge("inat", "final")
# workflow.add_edge("final", END)

# graph = workflow.compile()

# print("✅ Pipeline ready.")

# # -----------------------------
# # ❓ Ask Function
# # -----------------------------
# def ask(question: str):
#     mem = load_memory()

#     result = graph.invoke({
#         "query": question,
#         "answer": "",
#         "context": "",
#         "citations": [],
#         "chat_history": mem
#     })

#     mem.append({"query": question, "answer": result["answer"]})
#     save_memory(mem)

#     return result["answer"]


# # ============================================================
# # 🎨 STREAMLIT UI (Inside Same Script)
# # ============================================================

# st.title("🤖 Hybrid RAG Chatbot")
# st.write("Ask anything! The chatbot uses Local DB → Google → Wiki → GBIF → iNat.")

# if "chat" not in st.session_state:
#     st.session_state.chat = []

# # Display chat history
# for role, text in st.session_state.chat:
#     with st.chat_message(role):
#         st.markdown(text)

# # User input
# query = st.chat_input("Ask a question...")

# if query:
#     # Display user message
#     st.session_state.chat.append(("user", query))
#     with st.chat_message("user"):
#         st.write(query)

#     # Bot response
#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             answer = ask(query)
#             st.write(answer)
#             st.session_state.chat.append(("assistant", answer))





# =====================================================
#   HYBRID RAG CHATBOT (Single DB + HuggingFace)
# =====================================================

import streamlit as st
import os
import json
import re
from urllib.parse import quote
from typing import TypedDict, List, Dict, Any
import requests

# ---- IMPORTANT: Disable buggy Chroma v2 backend ----
os.environ["CHROMA_USE_V2"] = "false"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

# LangGraph + LangChain
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.utilities import WikipediaAPIWrapper, GoogleSearchAPIWrapper
from langchain_community.tools import WikipediaQueryRun, GoogleSearchRun


# =====================================================
# 🔑 API KEYS
# =====================================================
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")


os.environ["google_api_key"] = "AIzaSyBMSTBqYv74VqltxMj7G8eUtbuQg8tUROg"
os.environ["google_cse_id"] = "94a6404e7eb494900"


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")


if not GOOGLE_API_KEY:
    st.error("❌ Missing GOOGLE_API_KEY environment variable!")
if not GOOGLE_CSE_ID:
    st.error("❌ Missing GOOGLE_CSE_ID environment variable!")


# =====================================================
# 🤖 LLM
# =====================================================
gemini = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",   # free-tier safe
    temperature=0,
    api_key=GOOGLE_API_KEY
)


# =====================================================
# 📂 Vector DB Path
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, "small_vector_db_")   # your DB folder


# =====================================================
# 🧠 Load Vector DB
# =====================================================
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db_local = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings
)

retriever_local = db_local.as_retriever(search_kwargs={"k": 8})


# =====================================================
# 🌐 External Tools
# =====================================================
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

google_tool = GoogleSearchRun(
    api_wrapper=GoogleSearchAPIWrapper(
        google_api_key=GOOGLE_API_KEY,
        google_cse_id=GOOGLE_CSE_ID
    )
)


# =====================================================
# 🧠 Memory Storage (disk)
# =====================================================
MEMORY_FILE = os.path.join(BASE_DIR, "chat_memory.json")


def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            return json.load(open(MEMORY_FILE, "r", encoding="utf-8"))
        except:
            return []
    return []


def save_memory(mem):
    json.dump(mem[-20:], open(MEMORY_FILE, "w", encoding="utf-8"), indent=2)


# =====================================================
# Utility Functions
# =====================================================

def clean_query(q: str) -> str:
    return re.sub(r"[\n\r]+", " ", q.strip())


def extractive_answer(query: str, docs: List[Any]) -> str:
    """LLM extracts using vector DB context."""
    ctx = "\n\n".join(
        f"[{i+1}] {d.page_content}" for i, d in enumerate(docs[:6])
    )

    prompt = f"""
Answer the question using ONLY the context.
Every sentence must include citation markers [1], [2], etc.
If answer cannot be found, reply with NOINFO.

Question: {query}

CONTEXT:
{ctx}
"""

    ans = gemini.invoke(prompt).content.strip()
    if ans.upper().startswith("NOINFO") or len(ans) < 40:
        return ""
    return ans


def scholarly_lookup(query: str, max_results=3):
    out = []
    try:
        r = requests.get(
            f"https://api.crossref.org/works?query={quote(query)}&rows={max_results}",
            timeout=8
        ).json()
        for item in r.get("message", {}).get("items", []):
            title = item.get("title", ["Untitled"])[0]
            year = item.get("issued", {}).get("date-parts", [[0]])[0][0]
            doi = item.get("DOI", "")
            url = f"https://doi.org/{doi}"
            out.append(f"{title} ({year}). {url}")
    except:
        pass

    return out or ["(No scholarly reference found)"]


def format_clickable_citations(citations):
    out = []
    for i, c in enumerate(citations, 1):
        m = re.search(r"(https?://[^\s]+)", c)
        link = m.group(1) if m else ""
        if link:
            out.append(f"[{i}] [{c}]({link})")
        else:
            out.append(f"[{i}] {c}")
    return "\n".join(out)


# =====================================================
# Graph State Definition
# =====================================================

class GraphState(TypedDict):
    query: str
    answer: str
    context: str
    citations: List[str]
    chat_history: List[Dict[str, str]]


# =====================================================
# Pipeline Nodes
# =====================================================

def local_db_node(state: GraphState):
    q = clean_query(state["query"])
    docs = retriever_local.invoke(q)

    if not docs:
        return {**state, "context": "no_local"}

    ans = extractive_answer(q, docs)
    if not ans:
        return {**state, "context": "no_local"}

    google_link = f"https://www.google.com/search?q={quote(q)}"
    ans += f"\n\n📚 Source: [Google Search]({google_link})"

    return {**state, "answer": ans, "context": "local", "citations": [google_link]}


def google_node(state: GraphState):
    q = clean_query(state["query"])
    blob = google_tool.run(q)

    if not blob:
        return {**state, "context": "no_google"}

    ans = gemini.invoke(f"Summarize Google result:\n{blob}").content.strip()
    link = f"https://www.google.com/search?q={quote(q)}"
    ans += f"\n\n📚 Source: [Google Search]({link})"

    return {**state, "answer": ans, "context": "google", "citations": [link]}


def wiki_node(state: GraphState):
    q = clean_query(state["query"])
    blob = wiki_tool.run(q)

    if not blob:
        return {**state, "context": "no_wiki"}

    link = f"https://en.wikipedia.org/wiki/Special:Search?search={quote(q)}"

    ans = gemini.invoke(f"Answer using Wikipedia:\n{blob}").content.strip()
    ans += f"\n\n📚 Source: [Wikipedia]({link})"

    return {**state, "answer": ans, "context": "wiki", "citations": [link]}


def final_node(state: GraphState):
    q = clean_query(state["query"])

    prompt = f"""
Summarize the answer clearly.

Question: {q}
Answer: {state['answer']}
"""

    summary = gemini.invoke(prompt).content.strip()

    if state["citations"]:
        summary += f"\n\n📚 Citations:\n{format_clickable_citations(state['citations'])}"

    return {**state, "answer": summary}


# =====================================================
# Build Workflow Graph
# =====================================================

workflow = StateGraph(GraphState)

workflow.add_node("local", local_db_node)
workflow.add_node("google", google_node)
workflow.add_node("wiki", wiki_node)
workflow.add_node("final", final_node)

workflow.add_edge(START, "local")

workflow.add_conditional_edges(
    "local", lambda s: s["context"], {"local": "final", "no_local": "google"}
)

workflow.add_conditional_edges(
    "google", lambda s: s["context"], {"google": "final", "no_google": "wiki"}
)

workflow.add_edge("wiki", "final")
workflow.add_edge("final", END)

graph = workflow.compile()


# =====================================================
# STREAMLIT UI
# =====================================================

st.set_page_config(page_title="Hybrid RAG Chatbot", page_icon="🤖")
st.title("🤖 Hybrid RAG Chatbot (Single DB + HuggingFace)")

if "chat" not in st.session_state:
    st.session_state.chat = load_memory()

query = st.text_input("Ask a question:")

if st.button("Send"):
    if query.strip():
        mem = st.session_state.chat

        result = graph.invoke({
            "query": query,
            "answer": "",
            "context": "",
            "citations": [],
            "chat_history": mem
        })

        st.write("### 🧠 Answer")
        st.write(result["answer"])
        st.write(f"**Source step:** `{result['context']}`")

        mem.append({"query": query, "answer": result["answer"]})
        save_memory(mem)
        st.session_state.chat = mem

st.write("---")
st.write("### 💬 Chat History")

for c in st.session_state.chat[-10:]:
    st.markdown(f"**You:** {c['query']}")
    st.markdown(f"**Bot:** {c['answer']}")
