"""Streamlit RAG app using Strands Agents and AWS Bedrock.

This version removes LangChain and instead uses the lightweight
`strands-agents` package.  Embeddings are generated with Amazon Titan
via AWS Bedrock and stored in a persistent ChromaDB collection.  The
application allows users to upload documents, build a vector index and
ask questions with answers grounded in retrieved chunks.

AWS credentials are handled the same way as the previous implementation:
they are sourced from environment variables or a local `.env` file and
two regions are used – one for embeddings and one for the LLM.
"""

import os
import json
import time
import tempfile
import threading
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv
import PyPDF2

import boto3
from botocore.config import Config

import chromadb
from chromadb.config import Settings

from strands import Agent, tool
from strands.models import BedrockModel


# ---------------------------------------------------------------------------
# Environment & configuration
# ---------------------------------------------------------------------------
st.set_page_config(page_title="🧠 Knowledgebase (Strands + Bedrock)", layout="wide")

load_dotenv()

AWS_EMBEDDING_REGION = os.getenv("AWS_EMBEDDING_REGION", "us-east-1")
AWS_LLM_REGION = os.getenv("AWS_LLM_REGION", "us-east-2")

AWS_BEDROCK_EMBEDDING_MODEL_ID = os.getenv(
    "AWS_BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v1"
)
AWS_BEDROCK_LLM_MODEL_ID = os.getenv(
    "AWS_BEDROCK_LLM_MODEL_ID", "us.anthropic.claude-3-haiku-20240307-v1:0"
)

DATA_DIR = "data"
TEMP_DIR = tempfile.mkdtemp(prefix="kb_chroma_")
PERSIST_ROOT = os.path.join(TEMP_DIR, "chroma_db")

os.makedirs(DATA_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Globals used by the retrieval tool
# ---------------------------------------------------------------------------
CHROMA_CLIENT = None
COLLECTION = None
COLLECTION_NAME = "kb_main"
K_RETRIEVE = 3

_LAST_SOURCES: List[Dict[str, Any]] = []
_LAST_SOURCES_LOCK = threading.Lock()


def _set_last_sources(items: List[Dict[str, Any]]) -> None:
    """Save retrieved chunks in a thread-safe buffer."""
    global _LAST_SOURCES
    with _LAST_SOURCES_LOCK:
        _LAST_SOURCES = items


def _get_last_sources() -> List[Dict[str, Any]]:
    """Return a copy of the last retrieved chunks."""
    with _LAST_SOURCES_LOCK:
        return list(_LAST_SOURCES)


# ---------------------------------------------------------------------------
# AWS Bedrock helpers
# ---------------------------------------------------------------------------
def bedrock_client_embedding():
    """Client for embedding model in the embedding region."""
    return boto3.client(
        "bedrock-runtime",
        region_name=AWS_EMBEDDING_REGION,
        config=Config(retries={"max_attempts": 5, "mode": "standard"}),
    )


def titan_embed(text: str) -> List[float]:
    """Generate an embedding using Amazon Titan via Bedrock."""
    if not text or not text.strip():
        return []

    body = {"inputText": text}

    resp = bedrock_client_embedding().invoke_model(
        modelId=AWS_BEDROCK_EMBEDDING_MODEL_ID,
        body=json.dumps(body).encode("utf-8"),
        accept="application/json",
        contentType="application/json",
    )
    payload = json.loads(resp["body"].read())
    return payload.get("embedding", [])


# ---------------------------------------------------------------------------
# Document utilities
# ---------------------------------------------------------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract plain text from a PDF file using PyPDF2."""
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            t = page.extract_text() or ""
            if t:
                text += t + "\n\n"
    return text.strip()


def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Naive character-based splitter with overlap."""
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks


# ---------------------------------------------------------------------------
# ChromaDB helpers
# ---------------------------------------------------------------------------
def new_chroma_client(persist_dir: str):
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )


def reset_collection(client, name: str):
    existing = [c.name for c in client.list_collections()]
    if name in existing:
        client.delete_collection(name)
    return client.create_collection(name=name)


def load_collection_from_persist() -> bool:
    global CHROMA_CLIENT, COLLECTION
    persist_dir = st.session_state.get("persist_dir")
    if not persist_dir:
        return False
    try:
        CHROMA_CLIENT = new_chroma_client(persist_dir)
        COLLECTION = CHROMA_CLIENT.get_or_create_collection(COLLECTION_NAME)
        return True
    except Exception as e:
        st.error(f"Failed to load collection: {e}")
        return False


# ---------------------------------------------------------------------------
# Indexing pipeline
# ---------------------------------------------------------------------------
def reindex_knowledgebase() -> bool:
    global CHROMA_CLIENT, COLLECTION

    docs: List[str] = []
    ids: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    files = [f for f in os.listdir(DATA_DIR) if f.endswith((".txt", ".md"))]
    if not files:
        st.error("No .txt or .md files found. Upload or convert PDFs first.")
        return False

    for fname in files:
        path = os.path.join(DATA_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            st.warning(f"Skipping {fname}: {e}")
            continue

        if not content.strip():
            st.warning(f"Skipping empty file: {fname}")
            continue

        chunks = split_text(content, chunk_size=500, overlap=50)
        for i, ch in enumerate(chunks):
            docs.append(ch)
            ids.append(f"{fname}-{i}")
            metadatas.append({"source": fname, "chunk": i})

    if not docs:
        st.error("No valid chunks produced. Check your files.")
        return False

    persist_dir = os.path.join(PERSIST_ROOT, f"run_{int(time.time())}")
    os.makedirs(persist_dir, exist_ok=True)

    CHROMA_CLIENT = new_chroma_client(persist_dir)
    COLLECTION = reset_collection(CHROMA_CLIENT, COLLECTION_NAME)

    batch_size = 32
    embeddings: List[List[float]] = []
    total = len(docs)

    with st.spinner("Embedding chunks with Titan..."):
        for i in range(0, total, batch_size):
            batch = docs[i : i + batch_size]
            for b in batch:
                embeddings.append(titan_embed(b))
            st.write(f"Embedded {min(i + batch_size, total)}/{total} chunks")

    COLLECTION.add(
        documents=docs, embeddings=embeddings, metadatas=metadatas, ids=ids
    )

    st.session_state["vectorstore_loaded"] = True
    st.session_state["persist_dir"] = persist_dir

    st.success(
        f"✅ Indexed {len(docs)} chunks from {len(set(m['source'] for m in metadatas))} file(s)."
    )
    try:
        st.info(f"🔢 Collection count: {COLLECTION.count()}")
    except Exception:
        pass
    return True


# ---------------------------------------------------------------------------
# Retrieval tool for Strands Agent
# ---------------------------------------------------------------------------
@tool
def retrieve_chunks(question: str) -> str:
    global COLLECTION, K_RETRIEVE
    if COLLECTION is None:
        return "[No index loaded]"

    q_vec = titan_embed(question)

    res = COLLECTION.query(
        query_embeddings=[q_vec],
        n_results=max(1, int(K_RETRIEVE)),
        include=["documents", "metadatas", "distances"],
    )

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    packed = []
    for d, m, dist in zip(docs, metas, dists):
        packed.append(
            {
                "text": d or "",
                "meta": m or {},
                "distance": float(dist) if dist is not None else None,
            }
        )
    _set_last_sources(packed)

    out_lines: List[str] = []
    for rank, item in enumerate(packed, start=1):
        src = item["meta"].get("source", "unknown")
        dist = item.get("distance")
        dist_str = f"{dist:.4f}" if isinstance(dist, (int, float)) else "NA"
        out_lines.append(
            f"[Source {rank} — {src} — dist:{dist_str}]\n{item['text']}"
        )
    return "\n\n".join(out_lines) if out_lines else "[No matching context]"


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.title("🧠 Knowledgebase (Strands + Bedrock) — No LangChain")

if "vectorstore_loaded" not in st.session_state:
    st.session_state["vectorstore_loaded"] = False

st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Go to", ["📤 Upload & Re-Index", "🗑️ Delete Files", "💬 Ask Questions"], index=0
)


# -------------------------- Upload & Re-Index ------------------------------
if page == "📤 Upload & Re-Index":
    st.header("📤 Upload Files")
    st.info(
        "Upload .pdf/.md/.txt files. After uploading, click **Re-index Knowledgebase** to build the vector index."
    )

    uploaded = st.file_uploader(
        "Upload .pdf, .md, .txt", type=["pdf", "md", "txt"], accept_multiple_files=True
    )
    if uploaded:
        for up in uploaded:
            original = up.name
            dest_path = os.path.join(DATA_DIR, original)

            try:
                if original.lower().endswith(".pdf"):
                    pdf_tmp = dest_path + ".tmp"
                    with open(pdf_tmp, "wb") as f:
                        f.write(up.read())
                    text = extract_text_from_pdf(pdf_tmp)
                    os.remove(pdf_tmp)

                    if text:
                        txt_path = os.path.join(DATA_DIR, original[:-4] + ".txt")
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(text)
                        st.success(f"Converted {original} → {os.path.basename(txt_path)}")
                    else:
                        st.error(f"Failed to extract text from {original}")
                else:
                    with open(dest_path, "wb") as f:
                        f.write(up.read())
                    st.success(f"Saved {original}")
            except Exception as e:
                st.error(f"Error handling {original}: {e}")

    if st.button("🔄 Re-index Knowledgebase", type="primary"):
        ok = reindex_knowledgebase()
        if not ok:
            st.warning("Indexing failed or no documents found.")

    existing = [f for f in os.listdir(DATA_DIR) if f.endswith((".txt", ".md"))]
    if existing:
        st.subheader("📋 Current Files")
        for f in existing:
            st.write("•", f)


# ------------------------------- Delete Files ------------------------------
elif page == "🗑️ Delete Files":
    st.header("🗑️ Delete Files")
    files = [f for f in os.listdir(DATA_DIR) if f.endswith((".txt", ".md"))]
    selected = st.multiselect("Select files to delete", files)
    if st.button("Delete Selected"):
        deleted = 0
        for f in selected:
            try:
                os.remove(os.path.join(DATA_DIR, f))
                deleted += 1
            except Exception as e:
                st.error(f"Error deleting {f}: {e}")
        if deleted:
            st.success(f"Deleted {deleted} file(s). Re-index to refresh the database.")


# ------------------------------- Ask Questions -----------------------------
elif page == "💬 Ask Questions":
    st.header("💬 Ask Questions")

    if COLLECTION is None:
        load_collection_from_persist()

    if COLLECTION is not None:
        try:
            st.info(f"📦 Collection: {COLLECTION_NAME} | 🔢 Chunks: {COLLECTION.count()}")
        except Exception as e:
            st.warning(f"Could not read collection count: {e}")

    if not st.session_state.get("vectorstore_loaded", False) and COLLECTION is None:
        st.warning("⚠️ No index loaded. Go to **Upload & Re-Index** and build the index first.")
    else:
        st.subheader("🔎 Retrieval Settings")
        k_value = st.slider("Top-K chunks to retrieve", 1, 10, 3, 1)
        K_RETRIEVE = int(k_value)

        st.subheader("🧪 Test Retrieval (No LLM)")
        debug_q = st.text_input("Test query", value="test")
        if st.button("Run Test Retrieval"):
            if COLLECTION is None:
                st.error("No index loaded.")
            else:
                q_vec = titan_embed(debug_q)
                res = COLLECTION.query(
                    query_embeddings=[q_vec],
                    n_results=K_RETRIEVE,
                    include=["documents", "metadatas", "distances"],
                )
                docs = (res.get("documents") or [[]])[0]
                metas = (res.get("metadatas") or [[]])[0]
                dists = (res.get("distances") or [[]])[0]

                if not docs:
                    st.warning("No chunks returned. Try a different query or re-index.")
                else:
                    import pandas as pd

                    table = pd.DataFrame(
                        [
                            {
                                "Rank": i + 1,
                                "Source": (metas[i] or {}).get("source", "unknown"),
                                "Chunk#": (metas[i] or {}).get("chunk", None),
                                "Distance": float(dists[i]) if dists and dists[i] is not None else None,
                                "Preview": (docs[i] or "")[:140].replace("\n", " ")
                                + ("…" if len(docs[i]) > 140 else ""),
                            }
                            for i in range(len(docs))
                        ]
                    )
                    st.dataframe(table, use_container_width=True)

                    for i, d in enumerate(docs, start=1):
                        src = (metas[i - 1] or {}).get("source", "unknown")
                        chk = (metas[i - 1] or {}).get("chunk", None)
                        dist = (
                            float(dists[i - 1])
                            if dists and dists[i - 1] is not None
                            else None
                        )
                        label = f"Result {i} — {src} — chunk {chk}"
                        if dist is not None:
                            label += f" — distance {dist:.4f}"
                        with st.expander(label):
                            st.write(d)

        st.subheader("💬 Ask via LLM")
        model_id = st.text_input(
            "Bedrock model ID", value=AWS_BEDROCK_LLM_MODEL_ID, help="e.g., us.amazon.nova-micro-v1:0"
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

        bedrock_model = BedrockModel(model_id=model_id, temperature=temperature, region=AWS_LLM_REGION)
        agent = Agent(model=bedrock_model, tools=[retrieve_chunks])

        question = st.text_input("Your question")
        if st.button("Generate Answer") and question:
            with st.spinner("Retrieving and generating answer..."):
                system_preamble = (
                    "You are a helpful assistant that answers using local documents. "
                    "First call the `retrieve_chunks` tool with the user's question, then answer concisely. "
                    "If you use any context, cite it inline as [Source 1], [Source 2], etc. "
                    "If no context is relevant, say so."
                )
                response = agent(
                    f"{system_preamble}\n\nUser question: {question}"
                )

                st.markdown("### 💬 Answer")
                st.write(str(response))

                st.markdown("### 📄 Retrieved Chunks (from Chroma)")
                rows = _get_last_sources()
                if not rows:
                    st.info("No chunks returned.")
                else:
                    import pandas as pd

                    table = pd.DataFrame(
                        [
                            {
                                "Rank": i + 1,
                                "Source": r["meta"].get("source", "unknown"),
                                "Chunk#": r["meta"].get("chunk", None),
                                "Distance": r.get("distance", None),
                                "Preview": (r["text"] or "")[:140].replace("\n", " ")
                                + ("…" if len(r["text"]) > 140 else ""),
                            }
                            for i, r in enumerate(rows)
                        ]
                    )
                    st.dataframe(table, use_container_width=True)

                    for i, r in enumerate(rows, start=1):
                        src = r["meta"].get("source", "unknown")
                        chk = r["meta"].get("chunk", None)
                        dist = r.get("distance", None)
                        title = f"Source {i} — {src} — chunk {chk}"
                        if isinstance(dist, (int, float)):
                            title += f" — distance {dist:.4f}"
                        with st.expander(title):
                            st.write(r["text"] or "")

