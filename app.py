import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
import streamlit as st # type: ignore
import time, json
from pathlib import Path

from src.document_ingestion import ingest_documents
from src.qwen_retriever import QwenRetriever
from src.traditional_rag import TraditionalRAGRetriever
from src.evaluator import compare_results

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Qwen Intranet Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400&display=swap');

:root {
    --ink: #0d0d0d;
    --paper: #f5f2eb;
    --accent: #c84b31;
    --muted: #8a8070;
    --border: #d4cfc5;
    --card: #faf8f3;
}

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: var(--paper);
    color: var(--ink);
}

.stApp { background-color: var(--paper); }

h1, h2, h3 { font-family: 'Syne', sans-serif; font-weight: 800; }

.result-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 4px;
    padding: 1.2rem 1.4rem;
    margin: 0.8rem 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
}
.score-badge {
    display: inline-block;
    background: var(--accent);
    color: white;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 2px;
    margin-bottom: 0.5rem;
}
.pipeline-tag {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    font-family: 'Syne', sans-serif;
    font-weight: 600;
}
.stButton > button {
    background-color: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em;
}
.metric-box {
    background: var(--card);
    border: 1px solid var(--border);
    padding: 1rem;
    border-radius: 4px;
    text-align: center;
}
.metric-val { font-size: 1.8rem; font-weight: 800; color: var(--accent); }
.metric-lbl { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }


/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--ink) !important;
    color: white !important;
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stSelectbox > div > div { background: #1a1a1a !important; }
            [data-testid="stBaseButton-secondary"] {color: red !important; border-color: red !important}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False
if "results" not in st.session_state:
    st.session_state.results = None
if "comparison" not in st.session_state:
    st.session_state.comparison = None

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🔍 Qwen Intranet Search")
st.markdown("**Multimodal RAG** — text + image retrieval over internal documents")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    model_variant = st.selectbox(
        "Qwen Model",
        ["Qwen-VL-Chat (7B)", "Qwen2-VL-2B-Instruct", "Qwen2-VL-7B-Instruct"],
        help="Choose the Qwen multimodal model variant to load locally"
    )

    device = st.selectbox("Device", ["cuda", "cpu", "mps"],
                          help="Inference device. Use 'cuda' for GPU acceleration.")

    top_k = st.slider("Top-K results", 1, 10, 5)
    enable_compare = st.checkbox("Compare with traditional RAG", value=True)

    st.markdown("---")
    st.markdown("### 📂 Document Ingestion")

    uploaded_files = st.file_uploader(
        "Upload PDFs / Images",
        type=["pdf", "png", "jpg", "jpeg", "docx"],
        accept_multiple_files=True
    )

    use_sample = st.checkbox("Use sample intranet docs", value=True)

    if st.button("🔄 Build Index", use_container_width=True):
        with st.spinner("Ingesting documents and building embeddings…"):
            docs = []
            if use_sample:
                docs += ingest_documents("data/sample_docs", source="sample")
            if uploaded_files:
                save_dir = Path("data/uploads")
                save_dir.mkdir(parents=True, exist_ok=True)
                for f in uploaded_files:
                    p = save_dir / f.name
                    p.write_bytes(f.read())
                docs += ingest_documents(str(save_dir), source="upload")

            if not docs:
                st.error("No documents found. Upload files or enable sample docs.")
            else:
                # Initialize retrievers
                st.session_state.qwen = QwenRetriever(model_variant, device)
                st.session_state.qwen.build_index(docs)

                if enable_compare:
                    st.session_state.trad = TraditionalRAGRetriever()
                    st.session_state.trad.build_index(docs)

                st.session_state.index_ready = True
                st.session_state.doc_count = len(docs)
                st.success(f"Indexed {len(docs)} document chunks ✓")

    if st.session_state.index_ready:
        st.markdown(f"**Index:** {st.session_state.doc_count} chunks ready")

# ── Main area ─────────────────────────────────────────────────────────────────
col_query, col_img = st.columns([3, 1])

with col_query:
    query_text = st.text_area(
        "📝 Text Query",
        placeholder="e.g. What is the HR leave policy? / Show diagrams about network architecture",
        height=80
    )

with col_img:
    query_image = st.file_uploader(
        "🖼 Image Query (optional)",
        type=["png", "jpg", "jpeg"],
        key="img_query"
    )

if query_image:
    st.image(query_image, caption="Query image", width=200)

run_btn = st.button("🔎 Search", disabled=not st.session_state.index_ready, use_container_width=False)

if not st.session_state.index_ready:
    st.info("👈 Build the index from the sidebar before searching.")

# ── Search ────────────────────────────────────────────────────────────────────
if run_btn and (query_text or query_image):
    img_bytes = query_image.read() if query_image else None

    with st.spinner("Running Qwen inference…"):
        t0 = time.time()
        qwen_results = st.session_state.qwen.retrieve(
            query_text=query_text,
            query_image=img_bytes,
            top_k=top_k
        )
        qwen_time = time.time() - t0

    trad_results, trad_time = None, None
    if enable_compare and hasattr(st.session_state, "trad"):
        with st.spinner("Running traditional RAG…"):
            t0 = time.time()
            trad_results = st.session_state.trad.retrieve(query_text or "", top_k=top_k)
            trad_time = time.time() - t0

    st.session_state.results = (qwen_results, qwen_time)
    if trad_results:
        st.session_state.comparison = compare_results(qwen_results, trad_results, qwen_time, trad_time)

# ── Results display ───────────────────────────────────────────────────────────
if st.session_state.results:
    qwen_results, qwen_time = st.session_state.results

    if enable_compare and st.session_state.comparison:
        tab1, tab2, tab3 = st.tabs(["🟠 Qwen Results", "📄 Traditional RAG", "📊 Comparison"])
    else:
        tab1, = st.tabs(["🟠 Qwen Results"])
        tab2 = tab3 = None

    with tab1:
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'<div class="metric-box"><div class="metric-val">{len(qwen_results)}</div><div class="metric-lbl">Results</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-box"><div class="metric-val">{qwen_time:.2f}s</div><div class="metric-lbl">Latency</div></div>', unsafe_allow_html=True)
        with m3:
            avg_score = sum(r["score"] for r in qwen_results) / max(len(qwen_results), 1)
            st.markdown(f'<div class="metric-box"><div class="metric-val">{avg_score:.3f}</div><div class="metric-lbl">Avg Score</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        for i, res in enumerate(qwen_results, 1):
            modality = "🖼 Image+Text" if res.get("has_image") else "📄 Text"
            st.markdown(f"""
            <div class="result-card">
                <span class="score-badge">#{i} · Score: {res['score']:.4f}</span>
                <span class="pipeline-tag" style="margin-left:12px">{modality} · {res['source']}</span>
                <p style="margin-top:0.6rem; font-size:0.85rem; font-family:'Syne',sans-serif">{res['text'][:400]}{'…' if len(res['text'])>400 else ''}</p>
            </div>
            """, unsafe_allow_html=True)
            if res.get("image_path"):
                try:
                    st.image(res["image_path"], width=300, caption=f"Visual chunk — {res['source']}")
                except:
                    pass

    if tab2 and st.session_state.comparison:
        comp = st.session_state.comparison
        with tab2:
            st.markdown(f"**Latency:** {comp['trad_time']:.2f}s")
            for i, res in enumerate(comp["trad_results"], 1):
                st.markdown(f"""
                <div class="result-card">
                    <span class="score-badge">#{i} · Score: {res['score']:.4f}</span>
                    <span class="pipeline-tag" style="margin-left:12px">📄 Text-only · {res['source']}</span>
                    <p style="margin-top:0.6rem; font-size:0.85rem; font-family:'Syne',sans-serif">{res['text'][:400]}{'…' if len(res['text'])>400 else ''}</p>
                </div>
                """, unsafe_allow_html=True)

    if tab3 and st.session_state.comparison:
        comp = st.session_state.comparison
        with tab3:
            st.markdown("### Pipeline Comparison")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Qwen Multimodal RAG**")
                st.json({
                    "avg_score": round(comp["qwen_avg_score"], 4),
                    "latency_s": round(comp["qwen_time"], 3),
                    "multimodal": True,
                    "top_k": top_k,
                })
            with c2:
                st.markdown("**Traditional Text RAG**")
                st.json({
                    "avg_score": round(comp["trad_avg_score"], 4),
                    "latency_s": round(comp["trad_time"], 3),
                    "multimodal": False,
                    "top_k": top_k,
                })

            st.markdown("---")
            delta_score = comp["qwen_avg_score"] - comp["trad_avg_score"]
            delta_lat   = comp["qwen_time"] - comp["trad_time"]
            verdict = "✅ Qwen scores higher" if delta_score > 0 else "❌ Traditional RAG scores higher"
            st.markdown(f"""
            | Metric | Qwen | Traditional RAG | Delta |
            |---|---|---|---|
            | Avg Score | {comp['qwen_avg_score']:.4f} | {comp['trad_avg_score']:.4f} | `{delta_score:+.4f}` |
            | Latency (s) | {comp['qwen_time']:.3f} | {comp['trad_time']:.3f} | `{delta_lat:+.3f}` |
            | Modalities | Text + Image | Text only | — |
            """)
            st.markdown(f"**Verdict:** {verdict}")
            st.caption("Note: Scores reflect cosine similarity from embedding spaces. "
                       "Qwen's multimodal embeddings cover visual + textual semantics jointly.")
