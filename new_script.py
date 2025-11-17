import streamlit as st
import os
import json
import re
import requests
from urllib.parse import quote
from typing import TypedDict, List, Dict, Any

# LangGraph
from langgraph.graph import StateGraph, START, END

# Existing Chroma DB
from langchain_community.vectorstores import Chroma

# Google Search & Wikipedia Tools
from langchain_community.utilities import WikipediaAPIWrapper, GoogleSearchAPIWrapper
from langchain_community.tools import WikipediaQueryRun, GoogleSearchRun

# Google Gemini API
from google.generativeai import configure, GenerativeModel


# ======================================================
# 🔐 API KEYS
# ======================================================
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
GOOGLE_CSE_ID = st.secrets["GOOGLE_CSE_ID"]

configure(api_key=GOOGLE_API_KEY)
gemini = GenerativeModel("gemini-2.5-flash")


# ======================================================
# 📁 LOAD EXISTING CHROMA DB
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, "small_vector_db")

db1 = Chroma(persist_directory=PERSIST_DIR)
retriever1 = db1.as_retriever(search_kwargs={"k": 6})


# ======================================================
# 🔧 External Tools
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

def normalize_chat(mem):
    """Ensure memory is list of {query,answer} dicts."""
    fixed = []
    for item in mem:
        if isinstance(item, dict):
            fixed.append(item)
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            fixed.append({"query": item[0], "answer": item[1]})
    return fixed

def save_memory(mem):
    json.dump(mem[-15:], open(MEMORY_FILE, "w", encoding="utf-8"), indent=2)


# ======================================================
# 🧰 Utility Helpers
# ======================================================
def clean_query(q: str) -> str:
    return re.sub(r"[\n\r]+", " ", q.strip())

def detect_greeting(text: str) -> bool:
    """
    Detect greeting phrases like 'hi', 'hello', 'hey', etc.
    Uses regex with word boundaries to avoid partial matches like 'they'.
    """
    text = text.lower().strip()
    greeting_patterns = [
        r"\bhi\b",
        r"\bhello\b",
        r"\bhey\b",
        r"\bgood morning\b",
        r"\bgood afternoon\b",
        r"\bgood evening\b"
    ]
    return any(re.search(pattern, text) for pattern in greeting_patterns)

def ask_gemini(prompt: str) -> str:
    try:
        res = gemini.generate_content(prompt)
        return res.text
    except:
        return "Gemini API Error."

def extractive_answer(query, docs):
    """Use local DB only."""
    if not docs:
        return ""

    ctx = "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs[:4]))

    prompt = f"""
Use ONLY the numbered CONTEXT below to answer.

Every sentence MUST end with a citation like [1], [2].

If no answer is found, return "NOINFO".

Question: {query}

CONTEXT:
{ctx}
"""

    ans = ask_gemini(prompt)
    if ans.startswith("NOINFO") or len(ans) < 40:
        return ""
    return ans

def scholarly_lookup(query: str, max_results=3):
    """Add scholarly citations."""
    try:
        r = requests.get(
            f"https://api.crossref.org/works?rows={max_results}&query={quote(query)}",
            timeout=8,
        ).json()
        items = r.get("message", {}).get("items", [])

        out = []
        for it in items:
            title = it.get("title", ["Untitled"])[0]
            year = it.get("issued", {}).get("date-parts", [[None]])[0][0]
            doi = it.get("DOI", "")
            link = f"https://doi.org/{doi}" if doi else ""
            out.append(f"{title} ({year}). {link}")

        return out or ["(No scholarly reference found)"]

    except:
        return ["(No scholarly reference found)"]


# ======================================================
# 🌉 GRAPH STATE
# ======================================================
class GraphState(TypedDict):
    query: str
    answer: str
    context: str
    citations: List[str]
    chat_history: List[Dict[str, str]]


# ======================================================
# 🔵 NODES (DB → Google → Wiki → Final)
# ======================================================
def db1_node(state: GraphState):
    q = clean_query(state["query"])
    docs = retriever1.invoke(q)
    ans = extractive_answer(q, docs)
    if ans:
        return {**state, "answer": ans, "context": "db1"}
    return {**state, "answer": "", "context": "no_db1"}

def google_node(state: GraphState):
    try:
        res = google_tool.invoke({"query": state["query"]})
        ans = ask_gemini(f"Summarize this Google result:\n{res}")
        if ans:
            return {**state, "answer": ans, "context": "google"}
        return {**state, "context": "no_google"}
    except:
        return {**state, "context": "no_google"}

def wiki_node(state: GraphState):
    try:
        res = wiki_tool.invoke({"query": state["query"]})
        ans = ask_gemini(f"Summarize this Wikipedia text:\n{res}")
        return {**state, "answer": ans, "context": "wiki"}
    except:
        return {**state, "context": "wiki"}

def final_node(state: GraphState):
    q = clean_query(state["query"])
    answer = state["answer"] or ask_gemini(q)
    refs = scholarly_lookup(q)
    state["answer"] = f"{answer}\n\n**References:**\n" + "\n".join(refs)
    state["citations"] = refs
    return state


# ======================================================
# 🔀 BUILD SHALLOW WORKFLOW (NO DB2)
# ======================================================
workflow = StateGraph(GraphState)

workflow.add_node("db1", db1_node)
workflow.add_node("google", google_node)
workflow.add_node("wiki", wiki_node)
workflow.add_node("final", final_node)

workflow.add_edge(START, "db1")

workflow.add_conditional_edges(
    "db1",
    lambda s: s["context"],
    {
        "db1": "final",
        "no_db1": "google",
    },
)

workflow.add_conditional_edges(
    "google",
    lambda s: s["context"],
    {
        "google": "final",
        "no_google": "wiki",
    },
)

workflow.add_edge("wiki", "final")
workflow.add_edge("final", END)

graph = workflow.compile()


# ======================================================
# 🎨 STREAMLIT UI
# ======================================================
st.set_page_config(page_title="Hybrid RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Small-VectorDB + Google + Wikipedia Chatbot")

if "chat" not in st.session_state:
    st.session_state.chat = normalize_chat(load_memory())

user_input = st.text_input("Ask me anything:")

if st.button("Submit"):
    if user_input.strip():
        mem = st.session_state.chat

        # 👋 Greeting check (only if the entire input is a greeting)
        if detect_greeting(user_input):
            bot_reply = "Hey there! 👋 How can I help you today?"
            st.markdown("### 🤖 Response")
            st.write(bot_reply)
            mem.append({"query": user_input, "answer": bot_reply})
            save_memory(mem)
            st.session_state.chat = mem
        else:
            # Regular RAG flow
            result = graph.invoke({
                "query": user_input,
                "answer": "",
                "context": "",
                "citations": [],
                "chat_history": mem,
            })

            st.markdown("### 🧠 Response")
            st.write(result["answer"])
            st.write(f"**Source:** `{result['context']}`")

            mem.append({"query": user_input, "answer": result["answer"]})
            save_memory(mem)
            st.session_state.chat = mem


st.write("---")
st.write("### Recent Chat History")
for c in st.session_state.chat[-10:]:
    st.markdown(f"**You:** {c.get('query','')}")
    st.markdown(f"**Bot:** {c.get('answer','')}")














# import streamlit as st
# import os
# import json
# import re
# import requests
# from urllib.parse import quote
# from typing import TypedDict, List, Dict, Any

# # LangGraph
# from langgraph.graph import StateGraph, START, END

# # Existing Chroma DB
# from langchain_community.vectorstores import Chroma

# # Google Search & Wikipedia Tools
# from langchain_community.utilities import WikipediaAPIWrapper, GoogleSearchAPIWrapper
# from langchain_community.tools import WikipediaQueryRun, GoogleSearchRun

# # Google Gemini API
# from google.generativeai import configure, GenerativeModel


# # ======================================================
# # 🔐 API KEYS
# # ======================================================
# GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
# GOOGLE_CSE_ID = st.secrets["GOOGLE_CSE_ID"]

# configure(api_key=GOOGLE_API_KEY)
# gemini = GenerativeModel("gemini-2.5-flash")


# # ======================================================
# # 📁 LOAD EXISTING CHROMA DB
# # ======================================================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# PERSIST_DIR = os.path.join(BASE_DIR, "small_vector_db")

# db1 = Chroma(persist_directory=PERSIST_DIR)
# retriever1 = db1.as_retriever(search_kwargs={"k": 6})


# # ======================================================
# # 🔧 External Tools
# # ======================================================
# wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
# google_tool = GoogleSearchRun(
#     api_wrapper=GoogleSearchAPIWrapper(
#         google_api_key=GOOGLE_API_KEY,
#         google_cse_id=GOOGLE_CSE_ID,
#     )
# )


# # ======================================================
# # 💾 CHAT MEMORY
# # ======================================================
# MEMORY_FILE = "chat_memory.json"

# def load_memory():
#     if os.path.exists(MEMORY_FILE):
#         try:
#             return json.load(open(MEMORY_FILE, "r", encoding="utf-8"))
#         except:
#             return []
#     return []

# def normalize_chat(mem):
#     """Ensure memory is list of {query,answer} dicts."""
#     fixed = []
#     for item in mem:
#         if isinstance(item, dict):
#             fixed.append(item)
#         elif isinstance(item, (list, tuple)) and len(item) == 2:
#             fixed.append({"query": item[0], "answer": item[1]})
#     return fixed

# def save_memory(mem):
#     json.dump(mem[-15:], open(MEMORY_FILE, "w", encoding="utf-8"), indent=2)


# # ======================================================
# # 🧰 Utility Helpers
# # ======================================================
# def clean_query(q: str) -> str:
#     return re.sub(r"[\n\r]+", " ", q.strip())

# def detect_greeting(text: str) -> bool:
#     """Detect if the user is greeting."""
#     greetings = [
#         "hi", "hello", "hey",
#         "good morning", "good afternoon", "good evening"
#     ]
#     text_lower = text.lower().strip()
#     return any(g in text_lower for g in greetings)

# def ask_gemini(prompt: str) -> str:
#     try:
#         res = gemini.generate_content(prompt)
#         return res.text
#     except:
#         return "Gemini API Error."

# def extractive_answer(query, docs):
#     """Use local DB only."""
#     if not docs:
#         return ""

#     ctx = "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs[:4]))

#     prompt = f"""
# Use ONLY the numbered CONTEXT below to answer.

# Every sentence MUST end with a citation like [1], [2].

# If no answer is found, return "NOINFO".

# Question: {query}

# CONTEXT:
# {ctx}
# """

#     ans = ask_gemini(prompt)
#     if ans.startswith("NOINFO") or len(ans) < 40:
#         return ""
#     return ans

# def scholarly_lookup(query: str, max_results=3):
#     """Add scholarly citations."""
#     try:
#         r = requests.get(
#             f"https://api.crossref.org/works?rows={max_results}&query={quote(query)}",
#             timeout=8,
#         ).json()
#         items = r.get("message", {}).get("items", [])

#         out = []
#         for it in items:
#             title = it.get("title", ["Untitled"])[0]
#             year = it.get("issued", {}).get("date-parts", [[None]])[0][0]
#             doi = it.get("DOI", "")
#             link = f"https://doi.org/{doi}" if doi else ""
#             out.append(f"{title} ({year}). {link}")

#         return out or ["(No scholarly reference found)"]

#     except:
#         return ["(No scholarly reference found)"]


# # ======================================================
# # 🌉 GRAPH STATE
# # ======================================================
# class GraphState(TypedDict):
#     query: str
#     answer: str
#     context: str
#     citations: List[str]
#     chat_history: List[Dict[str, str]]


# # ======================================================
# # 🔵 NODES (DB → Google → Wiki → Final)
# # ======================================================
# def db1_node(state: GraphState):
#     q = clean_query(state["query"])
#     docs = retriever1.invoke(q)
#     ans = extractive_answer(q, docs)
#     if ans:
#         return {**state, "answer": ans, "context": "db1"}
#     return {**state, "answer": "", "context": "no_db1"}

# def google_node(state: GraphState):
#     try:
#         res = google_tool.invoke({"query": state["query"]})
#         ans = ask_gemini(f"Summarize this Google result:\n{res}")
#         if ans:
#             return {**state, "answer": ans, "context": "google"}
#         return {**state, "context": "no_google"}
#     except:
#         return {**state, "context": "no_google"}

# def wiki_node(state: GraphState):
#     try:
#         res = wiki_tool.invoke({"query": state["query"]})
#         ans = ask_gemini(f"Summarize this Wikipedia text:\n{res}")
#         return {**state, "answer": ans, "context": "wiki"}
#     except:
#         return {**state, "context": "wiki"}

# def final_node(state: GraphState):
#     q = clean_query(state["query"])
#     answer = state["answer"] or ask_gemini(q)
#     refs = scholarly_lookup(q)
#     state["answer"] = f"{answer}\n\n**References:**\n" + "\n".join(refs)
#     state["citations"] = refs
#     return state


# # ======================================================
# # 🔀 BUILD SHALLOW WORKFLOW (NO DB2)
# # ======================================================
# workflow = StateGraph(GraphState)

# workflow.add_node("db1", db1_node)
# workflow.add_node("google", google_node)
# workflow.add_node("wiki", wiki_node)
# workflow.add_node("final", final_node)

# workflow.add_edge(START, "db1")

# workflow.add_conditional_edges(
#     "db1",
#     lambda s: s["context"],
#     {
#         "db1": "final",
#         "no_db1": "google",
#     },
# )

# workflow.add_conditional_edges(
#     "google",
#     lambda s: s["context"],
#     {
#         "google": "final",
#         "no_google": "wiki",
#     },
# )

# workflow.add_edge("wiki", "final")
# workflow.add_edge("final", END)

# graph = workflow.compile()


# # ======================================================
# # 🎨 STREAMLIT UI
# # ======================================================
# st.set_page_config(page_title="Hybrid RAG Chatbot", page_icon="🤖", layout="wide")
# st.title("🤖 Small-VectorDB + Google + Wikipedia Chatbot")

# if "chat" not in st.session_state:
#     st.session_state.chat = normalize_chat(load_memory())

# user_input = st.text_input("Ask me anything:")

# if st.button("Submit"):
#     if user_input.strip():
#         mem = st.session_state.chat

#         # 👋 Greeting check
#         if detect_greeting(user_input):
#             bot_reply = "Hey there! 👋 How can I help you today?"
#             st.markdown("### 🤖 Response")
#             st.write(bot_reply)
#             mem.append({"query": user_input, "answer": bot_reply})
#             save_memory(mem)
#             st.session_state.chat = mem
#         else:
#             # Regular RAG flow
#             result = graph.invoke({
#                 "query": user_input,
#                 "answer": "",
#                 "context": "",
#                 "citations": [],
#                 "chat_history": mem,
#             })

#             st.markdown("### 🧠 Response")
#             st.write(result["answer"])
#             st.write(f"**Source:** `{result['context']}`")

#             mem.append({"query": user_input, "answer": result["answer"]})
#             save_memory(mem)
#             st.session_state.chat = mem


# st.write("---")
# st.write("### Recent Chat History")
# for c in st.session_state.chat[-10:]:
#     st.markdown(f"**You:** {c.get('query','')}")
#     st.markdown(f"**Bot:** {c.get('answer','')}")















# # import streamlit as st
# # import os
# # import json
# # import re
# # import requests
# # from urllib.parse import quote
# # from typing import TypedDict, List, Dict, Any

# # # LangGraph
# # from langgraph.graph import StateGraph, START, END

# # # Existing Chroma DB
# # from langchain_community.vectorstores import Chroma

# # # Google Search & Wikipedia Tools
# # from langchain_community.utilities import WikipediaAPIWrapper, GoogleSearchAPIWrapper
# # from langchain_community.tools import WikipediaQueryRun, GoogleSearchRun

# # # Google Gemini API
# # from google.generativeai import configure, GenerativeModel


# # # ======================================================
# # # 🔐 API KEYS
# # # ======================================================
# # GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
# # GOOGLE_CSE_ID = st.secrets["GOOGLE_CSE_ID"]

# # configure(api_key=GOOGLE_API_KEY)
# # gemini = GenerativeModel("gemini-2.5-flash")


# # # ======================================================
# # # 📁 LOAD EXISTING CHROMA DB
# # # ======================================================
# # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# # PERSIST_DIR = os.path.join(BASE_DIR, "small_vector_db")

# # db1 = Chroma(persist_directory=PERSIST_DIR)
# # retriever1 = db1.as_retriever(search_kwargs={"k": 6})


# # # ======================================================
# # # 🔧 External Tools
# # # ======================================================
# # wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
# # google_tool = GoogleSearchRun(
# #     api_wrapper=GoogleSearchAPIWrapper(
# #         google_api_key=GOOGLE_API_KEY,
# #         google_cse_id=GOOGLE_CSE_ID,
# #     )
# # )


# # # ======================================================
# # # 💾 CHAT MEMORY
# # # ======================================================
# # MEMORY_FILE = "chat_memory.json"


# # def load_memory():
# #     if os.path.exists(MEMORY_FILE):
# #         try:
# #             return json.load(open(MEMORY_FILE, "r", encoding="utf-8"))
# #         except:
# #             return []
# #     return []


# # def normalize_chat(mem):
# #     """Ensure memory is list of {query,answer} dicts."""
# #     fixed = []
# #     for item in mem:
# #         if isinstance(item, dict):
# #             fixed.append(item)
# #         elif isinstance(item, (list, tuple)) and len(item) == 2:
# #             fixed.append({"query": item[0], "answer": item[1]})
# #     return fixed


# # def save_memory(mem):
# #     json.dump(mem[-15:], open(MEMORY_FILE, "w", encoding="utf-8"), indent=2)


# # # ======================================================
# # # 🧰 Utility Helpers
# # # ======================================================
# # def clean_query(q: str) -> str:
# #     return re.sub(r"[\n\r]+", " ", q.strip())


# # def ask_gemini(prompt: str) -> str:
# #     try:
# #         res = gemini.generate_content(prompt)
# #         return res.text
# #     except:
# #         return "Gemini API Error."


# # def extractive_answer(query, docs):
# #     """Use local DB only."""
# #     if not docs:
# #         return ""

# #     ctx = "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs[:4]))

# #     prompt = f"""
# # Use ONLY the numbered CONTEXT below to answer.

# # Every sentence MUST end with a citation like [1], [2].

# # If no answer is found, return "NOINFO".

# # Question: {query}

# # CONTEXT:
# # {ctx}
# # """

# #     ans = ask_gemini(prompt)
# #     if ans.startswith("NOINFO") or len(ans) < 40:
# #         return ""

# #     return ans


# # def scholarly_lookup(query: str, max_results=3):
# #     """Add scholarly citations."""
# #     try:
# #         r = requests.get(
# #             f"https://api.crossref.org/works?rows={max_results}&query={quote(query)}",
# #             timeout=8,
# #         ).json()
# #         items = r.get("message", {}).get("items", [])

# #         out = []
# #         for it in items:
# #             title = it.get("title", ["Untitled"])[0]
# #             year = it.get("issued", {}).get("date-parts", [[None]])[0][0]
# #             doi = it.get("DOI", "")
# #             link = f"https://doi.org/{doi}" if doi else ""
# #             out.append(f"{title} ({year}). {link}")

# #         return out or ["(No scholarly reference found)"]

# #     except:
# #         return ["(No scholarly reference found)"]


# # # ======================================================
# # # 🌉 GRAPH STATE
# # # ======================================================
# # class GraphState(TypedDict):
# #     query: str
# #     answer: str
# #     context: str
# #     citations: List[str]
# #     chat_history: List[Dict[str, str]]


# # # ======================================================
# # # 🔵 NODES (DB → Google → Wiki → Final)
# # # ======================================================
# # def db1_node(state: GraphState):
# #     q = clean_query(state["query"])
# #     docs = retriever1.invoke(q)
# #     ans = extractive_answer(q, docs)

# #     if ans:
# #         return {**state, "answer": ans, "context": "db1"}

# #     return {**state, "answer": "", "context": "no_db1"}


# # def google_node(state: GraphState):
# #     try:
# #         res = google_tool.invoke({"query": state["query"]})
# #         ans = ask_gemini(f"Summarize this Google result:\n{res}")
# #         if ans:
# #             return {**state, "answer": ans, "context": "google"}
# #         return {**state, "context": "no_google"}
# #     except:
# #         return {**state, "context": "no_google"}


# # def wiki_node(state: GraphState):
# #     try:
# #         res = wiki_tool.invoke({"query": state["query"]})
# #         ans = ask_gemini(f"Summarize this Wikipedia text:\n{res}")
# #         return {**state, "answer": ans, "context": "wiki"}
# #     except:
# #         return {**state, "context": "wiki"}


# # def final_node(state: GraphState):
# #     q = clean_query(state["query"])
# #     answer = state["answer"] or ask_gemini(q)
# #     refs = scholarly_lookup(q)

# #     state["answer"] = f"{answer}\n\n**References:**\n" + "\n".join(refs)
# #     state["citations"] = refs
# #     return state


# # # ======================================================
# # # 🔀 BUILD SHALLOW WORKFLOW (NO DB2)
# # # ======================================================
# # workflow = StateGraph(GraphState)

# # workflow.add_node("db1", db1_node)
# # workflow.add_node("google", google_node)
# # workflow.add_node("wiki", wiki_node)
# # workflow.add_node("final", final_node)

# # workflow.add_edge(START, "db1")

# # workflow.add_conditional_edges(
# #     "db1",
# #     lambda s: s["context"],
# #     {
# #         "db1": "final",
# #         "no_db1": "google",
# #     },
# # )

# # workflow.add_conditional_edges(
# #     "google",
# #     lambda s: s["context"],
# #     {
# #         "google": "final",
# #         "no_google": "wiki",
# #     },
# # )

# # workflow.add_edge("wiki", "final")
# # workflow.add_edge("final", END)

# # graph = workflow.compile()


# # # ======================================================
# # # 🎨 STREAMLIT UI
# # # ======================================================
# # st.set_page_config(page_title="Hybrid RAG Chatbot", page_icon="🤖", layout="wide")
# # st.title("🤖 Small-VectorDB + Google + Wikipedia Chatbot")

# # if "chat" not in st.session_state:
# #     st.session_state.chat = normalize_chat(load_memory())

# # user_input = st.text_input("Ask me anything:")

# # if st.button("Submit"):
# #     if user_input.strip():
# #         mem = st.session_state.chat

# #         result = graph.invoke({
# #             "query": user_input,
# #             "answer": "",
# #             "context": "",
# #             "citations": [],
# #             "chat_history": mem,
# #         })

# #         st.markdown("### 🧠 Response")
# #         st.write(result["answer"])
# #         st.write(f"**Source:** `{result['context']}`")

# #         mem.append({"query": user_input, "answer": result["answer"]})
# #         save_memory(mem)
# #         st.session_state.chat = mem


# # st.write("---")
# # st.write("### Recent Chat History")
# # for c in st.session_state.chat[-10:]:
# #     st.markdown(f"**You:** {c.get('query','')}")
# #     st.markdown(f"**Bot:** {c.get('answer','')}")






















