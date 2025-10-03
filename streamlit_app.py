# HWs/HW4.py
# ---------------------------------------------
# HW4: RAG chatbot over provided HTML docs
# - Builds a persistent Chroma vector DB ONCE from ./html folder
# - Chunks each HTML file into exactly two mini-docs (see method notes below)
# - Chat UI with conversation memory (last 5 Q&A pairs)
# - Sidebar lets you pick: gpt-4o-mini, gpt-4o, gpt-4.1
# - Evaluation panel runs 5 test questions across all 3 models

import os
import re
import uuid
import json
from typing import List, Tuple

import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv




# --- Chroma setup (sqlite on some hosts needs pysqlite3 shim)
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from chromadb.errors import IDAlreadyExistsError

load_dotenv()

# -------- Configuration --------
HTML_FOLDER = "./files_rl"              # <- unzip your provided HTML files here
CHROMA_PATH = "./Chroma_HW4_new"        # persistent db folder
COLLECTION_NAME = "HW4_HTML_Collection"
EMBED_MODEL = "text-embedding-3-small"

import os
import streamlit as st

# --- Add your 3 streamers ---
from openai import OpenAI as OpenAIClient
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
try:
    import cohere
except ImportError:
    cohere = None

def stream_openai(messages, model_name):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        yield "[OpenAI] Missing API key."
        return
    client = OpenAIClient(api_key=api_key)
    model_id = "gpt-4o-mini" if "mini" in model_name.lower() else "gpt-4o"
    stream = client.chat.completions.create(
        model=model_id,
        messages=messages,
        stream=True,
        temperature=0.3,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and getattr(delta, "content", None):
            yield delta.content

def stream_anthropic(messages, model_name):
    api_key = os.getenv("CLAUDE_API_KEY")
    if not api_key or not HAS_ANTHROPIC:
        yield "[Anthropic] Missing API key or package."
        return
    client = anthropic.Anthropic(api_key=api_key)
    # Separate system from user/assistant
    system_text = "\n".join(m["content"] for m in messages if m["role"] == "system")
    conversation = [m for m in messages if m["role"] != "system"]

    if "haiku" in model_name.lower():
        model_id = "claude-3-haiku-20240307"
    elif "opus" in model_name.lower():
        model_id = "claude-3-opus-20240229"
    else:
        model_id = "claude-sonnet-4-20250514"

    try:
        with client.messages.stream(
            model=model_id,
            max_tokens=4000,
            temperature=0.3,
            system=system_text,
            messages=conversation
        ) as stream:
            for event in stream:
                if event.type == "content_block_delta" and hasattr(event.delta, 'text'):
                    yield event.delta.text
                elif event.type == "message_stop":
                    break
    except Exception as e:
        yield f"[Anthropic] Error: {str(e)}"

def stream_cohere(messages, model_name):
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key or cohere is None:
        yield "[Cohere] Missing COHERE_API_KEY or package."
        return
    co = cohere.Client(api_key)
    sys_text = "\n".join(m["content"] for m in messages if m["role"] == "system")
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in messages if m["role"] != "system")
    prompt = sys_text + "\n" + convo

    if "flagship" in model_name.lower():
        model_id = "command-a-03-2025"
    else:
        model_id = "command-a-vision-07-2025"

    try:
        stream = co.chat_stream(model=model_id, message=prompt)
        for event in stream:
            if event.event_type == "text-generation":
                yield event.text
            elif event.event_type == "error":
                yield f"[Cohere Error] {event.error}"
            elif event.event_type == "stream-end":
                break
    except Exception as e:
        yield f"[Cohere Error] {str(e)}"


MODEL_CHOICES = {
    "OpenAI (gpt-4o / 4o-mini)": stream_openai,
    "Claude (Haiku / Sonnet / Opus)": stream_anthropic,
    "Cohere (Flagship / Vision)": stream_cohere,
}

# 5 evaluation questions (edit if you want)
EVAL_QUESTIONS = [
    "What is the overall purpose of the website described by these HTML pages?",
    "List three key features or topics covered in the documents.",
    "Where would a new user find instructions or a getting-started section?",
    "Summarize one page’s content in 3 bullet points.",
    "Cite which page best answers a question about configuration or setup."
]

# ----------------- Utilities -----------------
def _safe_html_to_text(html: str) -> str:
    """
    Convert HTML to plain text. Try BeautifulSoup if available; else fall back to regex.
    """
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        # get text with reasonable spacing
        text = soup.get_text(separator="\n")
        # collapse extra blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
    except Exception:
        # fallback: strip tags, keep some whitespace
        text = re.sub(r"<(script|style)[\s\S]*?</\1>", " ", html, flags=re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

def _list_html_files(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".html")]

def _chunk_into_two(text: str) -> Tuple[str, str]:
    """
    CHUNKING METHOD (explained):
    - We create EXACTLY TWO mini-docs per file by splitting near the midpoint,
      *but* we prefer to break on paragraph boundaries for better semantic coherence.
    - Steps:
        1) Split on blank lines (paragraphs).
        2) Accumulate paragraphs until we reach ~half of total characters.
        3) First chunk = paragraphs up to that point, second chunk = remainder.
    WHY THIS METHOD:
    - It preserves local context and keeps each chunk reasonably self-contained.
    - It also guarantees exactly two chunks (as required) with minimal fragmentation.
    """
    if not text:
        return "", ""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) <= 1:
        # trivial path: hard midpoint split
        mid = max(1, len(text)//2)
        return text[:mid].strip(), text[mid:].strip()

    total_len = sum(len(p) for p in paras)
    target = total_len // 2
    acc, chunk1 = 0, []
    for p in paras:
        if acc < target:
            chunk1.append(p)
            acc += len(p)
        else:
            break
    chunk2_start_idx = len(chunk1)
    chunk1_text = "\n\n".join(chunk1).strip()
    chunk2_text = "\n\n".join(paras[chunk2_start_idx:]).strip()
    return chunk1_text, chunk2_text

def _ensure_vector_db(openai_client: OpenAI):
    """
    Create the persistent Chroma collection if empty.
    Only builds embeddings from HTML files once; subsequent runs reuse the DB.
    """
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    # If collection already has vectors, skip building
    try:
        existing_count = collection.count()
    except Exception:
        existing_count = 0

    if existing_count > 0:
        return collection

    html_files = _list_html_files(HTML_FOLDER)
    if not html_files:
        st.warning(f"No HTML files found in {HTML_FOLDER}. Please unzip your HTML set there.")
        return collection

    # Build once
    with st.status("Building vector DB from HTML files…", expanded=False):
        for path in html_files:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    html = f.read()
                text = _safe_html_to_text(html)
                c1, c2 = _chunk_into_two(text)

                # Embed the two chunks
                for i, chunk in enumerate([c1, c2], start=1):
                    if not chunk:
                        continue
                    emb = openai_client.embeddings.create(
                        model=EMBED_MODEL,
                        input=chunk
                    ).data[0].embedding

                    doc_id = f"{os.path.basename(path)}::part{i}"  # stable ID (prevents dup inserts)
                    meta = {"filename": os.path.basename(path), "part": i}

                    try:
                        collection.add(documents=[chunk], embeddings=[emb], ids=[doc_id], metadatas=[meta])
                    except IDAlreadyExistsError:
                        # Safe-upsert behavior: if someone re-runs a partial build, just skip duplicates
                        pass

            except Exception as e:
                st.write(f"⚠️ Skipped {os.path.basename(path)} due to error: {e}")

    st.success("Vector DB created ✅")
    return collection

def _retrieve_context(collection, openai_client: OpenAI, query: str, k: int = 4) -> List[dict]:
    """
    Get top-k chunks for the query.
    """
    q_emb = openai_client.embeddings.create(model=EMBED_MODEL, input=query).data[0].embedding
    result = collection.query(query_embeddings=[q_emb], n_results=k)
    hits = []
    if result and result.get("documents"):
        for i in range(len(result["documents"][0])):
            hits.append({
                "text": result["documents"][0][i],
                "metadata": result["metadatas"][0][i],
                "id": result["ids"][0][i],
                "distance": result.get("distances", [[None]])[0][i]
            })
    return hits

def _call_llm(openai_client: OpenAI, model: str, system_prompt: str, messages: List[dict]) -> str:
    stream = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}] + messages,
        stream=True,
    )
    # stream to a string (Streamlit will render tokens live via write_stream if you prefer)
    text = ""
    for chunk in stream:
        delta = getattr(chunk.choices[0].delta, "content", None)
        if delta:
            text += delta
    return text

# ----------------- Streamlit Page -----------------
def hw4_run():
    st.title("🧠 HW4 – RAG Chatbot (HTML → Vector DB → LLM)")

    # --- API key + OpenAI client ---
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("No OpenAI API key found. Add OPENAI_API_KEY to your .env.", icon="🗝️")
        st.stop()
    if "openai_client" not in st.session_state:
        st.session_state.openai_client = OpenAI(api_key=api_key)
    client = st.session_state.openai_client

    # --- Vector DB (create once, then reuse) ---
    collection = _ensure_vector_db(client)
    if collection is None:
        st.stop()

    # --- Sidebar settings ---
    st.sidebar.header("Settings")
    model_label = st.sidebar.radio("LLM Backend", list(MODEL_CHOICES.keys()))
    streamer = MODEL_CHOICES[model_label]
    top_k = st.sidebar.slider("Top-k retrieved chunks", min_value=2, max_value=8, value=4, step=1)
    st.sidebar.caption("Vector DB: created once; subsequent runs will reuse it.")

    # --- Conversation memory ---
    if "messages_hw4" not in st.session_state:
        st.session_state.messages_hw4 = [
            {"role": "assistant", "content": "Hi! Ask me anything about the provided HTML docs."}
        ]

    # --- Display previous chat ---
    for m in st.session_state.messages_hw4:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # --- User input ---
    user_q = st.chat_input("Type your question…")

    if user_q:
        # Save user message
        st.session_state.messages_hw4.append({"role": "user", "content": user_q})

        # Trim memory to last 5 user turns (plus assistant replies)
        user_msgs = [m for m in st.session_state.messages_hw4 if m["role"] == "user"]
        while len(user_msgs) > 5:
            idx = next(i for i, m in enumerate(st.session_state.messages_hw4) if m["role"] == "user")
            st.session_state.messages_hw4.pop(idx)
            if idx < len(st.session_state.messages_hw4) and st.session_state.messages_hw4[idx]["role"] == "assistant":
                st.session_state.messages_hw4.pop(idx)
            user_msgs = [m for m in st.session_state.messages_hw4 if m["role"] == "user"]

        # --- Retrieve context for this question ---
        hits = _retrieve_context(collection, client, user_q, k=top_k)
        context_blocks = [h["text"] for h in hits]
        context_text = "\n\n---\n\n".join(context_blocks)

        system_prompt = (
            "You are a helpful RAG assistant. Use the supplied CONTEXT first. "
            "If the answer is not fully in context, say what’s missing. "
            "Cite the filename and part in parentheses when using retrieved chunks."
        )
        rag_message = (
            f"CONTEXT:\n{context_text}\n\n"
            f"USER QUESTION:\n{user_q}\n\n"
            f"Remember to cite like (filename.html part 1) when relevant."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": rag_message}
        ]

        # --- Call chosen model ---
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                output = ""
                for token in streamer(messages, model_label):
                    output += token
                    st.write(token, end="")
                st.markdown(output)
        answer = output

        # Save assistant reply
        st.session_state.messages_hw4.append({"role": "assistant", "content": answer})




    # ---------- Evaluation Panel ----------
    st.divider()
    st.subheader("🔍 Model Evaluation")

    test_questions = [
        "What is the overall purpose of the website described by these HTML pages?",
        "List three key features or topics covered in the documents.",
	    "Where would a new user find instructions or a getting-started section?",
	    "Summarize one page’s content in 3 bullet points.",
	    "Cite which page best answers a question about configuration or setup."

    ]

    if st.button("Run evaluation on sample questions"):
        streamer = MODEL_CHOICES[model_label]   # <-- use the model selected in sidebar

        for q in test_questions:
            st.markdown(f"**Q:** {q}")
            # Retrieve context for each test question
            hits = _retrieve_context(collection, client, q, k=top_k)
            ctx = "\n\n---\n\n".join(h["text"] for h in hits)

            sys_p = (
                "You are a helpful RAG assistant. Use the supplied CONTEXT first. "
                "If the answer is not fully in context, say what’s missing. "
                "Cite the filename and part in parentheses when using retrieved chunks."
            )
            user_msg = f"CONTEXT:\n{ctx}\n\nQUESTION:\n{q}\n\nRemember to cite like (filename.html part 1) when relevant."
            messages = [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": user_msg}
            ]

            # 🔹 Only call the selected model
            st.write(f"**{model_label}:**")
            output = "".join(chunk for chunk in streamer(messages, model_label))
            st.markdown(output)
            st.markdown("---")





pg = st.navigation(
    {
        "HW4": st.Page(hw4_run, title="HW4"),
    }
)

pg.run()
        