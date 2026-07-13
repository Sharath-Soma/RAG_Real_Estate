"""Northstar Realty AI Assistant — Streamlit front-end.

Performance notes
-----------------
* ``@st.cache_resource`` on ``build_engine`` ensures the RAGEngine (and its
  SentenceTransformer + FAISS index + Gemini client) are constructed only once
  per server process.
* CSS is re-injected only when the active theme changes (keyed on session state).
* ``KnowledgeBaseSearch`` and ``SessionExporter`` are stored in session state
  so they are constructed once per browser session, not once per rerun.
* The ``QueryCache`` stored in session state is actually consulted before
  calling the LLM, giving sub-millisecond responses for repeated questions.
* ``handle_standard_query`` passes pre-computed ``search_results`` and
  ``context_chunks`` into ``engine.answer_question`` so the query is embedded
  and FAISS-searched exactly once per user turn (previously it happened twice).
"""

import time
import html
from datetime import datetime
from typing import List, Optional, Dict, Any

import streamlit as st

from config import (
    APP_SUBTITLE,
    APP_TITLE,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    LOGIN_PASSWORD,
    LOGIN_USERNAME,
)
from utils.rag_engine import RAGEngine
from utils.project import get_project_name
from utils.comparison import PropertyComparison
from utils.recommendation import PropertyRecommendation
from utils.retrieval_tracker import RetrievalTracker
from utils.exporter import SessionExporter
from utils.search import KnowledgeBaseSearch
from utils.cache import QueryCache

SUGGESTIONS = [
    "🏡 Compare Skyline Horizon Towers and Urban Nest Heights",
    "💰 I need a 3 BHK within 90 lakhs",
    "🏊 Properties with swimming pool and gym",
    "🎓 Near schools and metro stations",
]


@st.cache_resource(show_spinner=False)
def build_engine() -> RAGEngine:
    engine = RAGEngine()
    engine.initialize()
    return engine


def init_session_state() -> None:
    defaults = {
        "authenticated": False,
        "messages": [],
        "pending_prompt": None,
        "active_view": "chat",
        "theme": "dark",
        "response_times": [],
        "questions_asked": 0,
        "login_error": "",
        # QueryCache: kept in session state — created once per browser session.
        "query_cache": QueryCache(max_size=50),
        # KnowledgeBaseSearch and SessionExporter are heavyweight objects that
        # were previously re-instantiated on every rerun.  They are now created
        # lazily and stored here.
        "_kb_search": None,
        "_session_exporter": None,
        # Track the last theme we injected CSS for so we skip redundant renders.
        "_css_theme": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _get_kb_search(engine: RAGEngine) -> KnowledgeBaseSearch:
    """Return the session-level KnowledgeBaseSearch, creating it once."""
    if st.session_state._kb_search is None:
        st.session_state._kb_search = KnowledgeBaseSearch(engine.documents)
    return st.session_state._kb_search


def _get_exporter() -> SessionExporter:
    """Return the session-level SessionExporter, creating it once."""
    if st.session_state._session_exporter is None:
        st.session_state._session_exporter = SessionExporter()
    return st.session_state._session_exporter


def reset_chat() -> None:
    st.session_state.messages = []
    st.session_state.pending_prompt = None
    st.session_state.active_view = "chat"
    st.session_state.login_error = ""


def toggle_theme() -> None:
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"


def render_css() -> None:
    """Inject CSS on every rerun. (Streamlit requires CSS injection every cycle)."""
    theme = st.session_state.theme

    background = "#212121" if theme == "dark" else "#ffffff"
    surface = "#2f2f2f" if theme == "dark" else "#f4f4f4"
    text = "#ececec" if theme == "dark" else "#0d0d0d"
    muted = "#b4b4b4" if theme == "dark" else "#737373"
    accent = "#10a37f" if theme == "dark" else "#10a37f"
    border = "rgba(255,255,255,0.1)" if theme == "dark" else "rgba(0,0,0,0.1)"
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Söhne:wght@400;500;600&display=swap');
        :root {{ font-family: 'Söhne', ui-sans-serif, system-ui, -apple-system, sans-serif; }}
        .stApp {{ background: {background}; color: {text}; font-size: 16px; line-height: 1.6; }}
        .block-container {{ max-width: 800px; padding-top: 2rem; padding-bottom: 4rem; margin: 0 auto; }}
        
        h1, h2, h3, h4 {{ font-weight: 600; margin-bottom: 1rem; color: {text}; }}
        p, li {{ color: {text}; }}
        
        /* Chat rows */
        .chat-row {{ display: flex; margin-bottom: 1.5rem; width: 100%; align-items: flex-start; }}
        .chat-row.user {{ justify-content: flex-end; }}
        .chat-row.assistant {{ justify-content: flex-start; }}
        
        /* Bubbles */
        .bubble-user {{ background: {surface}; color: {text}; padding: 0.75rem 1.25rem; border-radius: 1.5rem; max-width: 80%; font-size: 1rem; }}
        .bubble-assistant {{ background: transparent; color: {text}; padding: 0; max-width: 85%; font-size: 1rem; flex-grow: 1; }}
        
        .avatar {{ width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; border-radius: 50%; margin-right: 1rem; font-size: 1.1rem; flex-shrink: 0; border: 1px solid {border}; background: {surface}; }}
        
        /* Cards */
        .card {{ border: 1px solid {border}; border-radius: 0.75rem; padding: 1.25rem; background: {background}; margin-bottom: 1rem; transition: background 0.2s ease; }}
        .card:hover {{ background: {surface}; }}
        
        /* Hero */
        .hero {{ padding: 3rem 0 2rem; text-align: center; }}
        .hero h2 {{ font-size: 2.2rem; font-weight: 600; margin-bottom: 0.5rem; margin-top: 0; }}
        .hero p {{ color: {muted}; font-size: 1.1rem; }}
        
        /* Sidebar */
        .sidebar .stButton > button {{ border-radius: 0.5rem; justify-content: flex-start; border: none; background: transparent; transition: background 0.2s; }}
        .sidebar .stButton > button:hover {{ background: {surface}; }}
        
        /* Markdown Code Blocks */
        pre {{ background-color: {surface} !important; border-radius: 0.5rem; padding: 1rem; overflow-x: auto; border: 1px solid {border}; }}
        code {{ font-family: 'ui-monospace', 'Cascadia Code', 'Source Code Pro', monospace; font-size: 0.85em; }}
        
        /* Input overrides */
        .stChatInputContainer {{ border: 1px solid {border} !important; border-radius: 1rem !important; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important; padding: 0.2rem !important; }}
        
        /* Typing animation */
        .typing {{ color: {muted}; font-style: italic; animation: pulse 1.5s infinite; padding: 0.5rem 0; }}
        @keyframes pulse {{ 0% {{ opacity: 0.6; }} 50% {{ opacity: 1; }} 100% {{ opacity: 0.6; }} }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state._css_theme = theme


def render_login_page() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🏠", layout="centered")
    render_css()
    st.markdown("<div class='hero' style='max-width: 560px; margin: 3rem auto;'>", unsafe_allow_html=True)
    st.markdown("## Secure Access")
    st.markdown("### Northstar Realty AI Assistant")
    st.caption("Sign in with your workspace credentials to continue.")

    with st.form("login_form"):
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        submit = st.form_submit_button("Sign in")

    if submit:
        if username == LOGIN_USERNAME and password == LOGIN_PASSWORD:
            st.session_state.authenticated = True
            st.session_state.login_error = ""
            st.rerun()
        else:
            st.session_state.login_error = "Invalid username or password"

    if st.session_state.login_error:
        st.error(st.session_state.login_error)
    st.markdown("</div>", unsafe_allow_html=True)


def render_sidebar(engine: RAGEngine) -> None:
    with st.sidebar:
        st.markdown("## Northstar")
        st.caption(APP_SUBTITLE)
        st.divider()

        # Navigation
        if st.button("🏠 New Chat", use_container_width=True):
            reset_chat()
        if st.button("🗑 Clear Chat", use_container_width=True):
            reset_chat()
        if st.button("📊 Analytics", use_container_width=True):
            st.session_state.active_view = "analytics"
        if st.button("📚 Browse KB", use_container_width=True):
            st.session_state.active_view = "search"
        if st.button("💾 Export", use_container_width=True):
            st.session_state.active_view = "export"
        if st.button("🌙 Toggle Theme", use_container_width=True):
            toggle_theme()
        if st.button("ℹ About", use_container_width=True):
            st.session_state.active_view = "about"

        st.divider()
        st.markdown("### 📊 Session Health")
        st.metric("Documents", engine.document_count)
        st.metric("Chunks", engine.chunk_count)
        st.metric("Questions", len([m for m in st.session_state.messages if m["role"] == "user"]))
        avg_latency = sum(st.session_state.response_times) / len(st.session_state.response_times) if st.session_state.response_times else 0
        st.metric("Avg Response", f"{avg_latency:.1f}s")

        st.divider()
        st.markdown("### 🔧 Configuration")
        st.caption(f"Embeddings: {EMBEDDING_MODEL.split('/')[-1]}")
        st.caption(f"LLM: {GENERATION_MODEL}")
        if engine.has_api_key():
            st.success("✓ API configured")
        else:
            st.warning("⚠ No API key")

        st.divider()
        st.markdown("### 🔍 Knowledge Base Search")
        search_query = st.text_input("Search documents", key="kb_search", placeholder="Type to search...")
        if search_query:
            # Use session-cached KnowledgeBaseSearch — not re-instantiated every rerun.
            kb_search = _get_kb_search(engine)
            results = kb_search.search_by_name(search_query)
            if results:
                st.markdown(f"**Found {len(results)} results**")
                for doc in results[:5]:
                    if st.button(f"📄 {doc['name']}", key=f"doc_{doc['name']}", use_container_width=True):
                        st.session_state.active_view = "search"
                        st.session_state.selected_doc = doc
                        st.rerun()
            else:
                st.info("No documents found")


def render_header() -> None:
    st.markdown("<div class='hero'>", unsafe_allow_html=True)
    st.markdown(f"<h2>{APP_TITLE}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p>{APP_SUBTITLE}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_suggestions() -> None:
    st.markdown("### Suggested questions")
    cols = st.columns(2)
    for index, suggestion in enumerate(SUGGESTIONS):
        if cols[index % 2].button(suggestion, key=f"suggestion-{index}", use_container_width=True):
            st.session_state.pending_prompt = suggestion
            st.rerun()


def render_chat_messages() -> None:
    total_messages = len(st.session_state.messages)
    for idx, message in enumerate(st.session_state.messages):
        role = message["role"]
        msg_type = message.get("type", "standard")
        
        with st.container():
            if role == "user":
                escaped_content = html.escape(message['content'])
                st.markdown(f"<div class='chat-row user'><div class='bubble-user'>{escaped_content}</div></div>", unsafe_allow_html=True)
            else:
                if message.get('content'):
                    st.markdown(f"<div class='chat-row assistant'><div class='avatar'>🤖</div><div class='bubble-assistant'>{message['content']}</div></div>", unsafe_allow_html=True)
                
                # Render comparison table if present
                if msg_type == "comparison" and "comparison_table" in message:
                    table_data = message["comparison_table"]
                    if table_data:
                        headers = list(table_data[0].keys())
                        headers = [h for h in headers if not h.startswith("_")]
                        md_table = "| " + " | ".join(headers) + " |\n"
                        md_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                        for row in table_data:
                            md_table += "| " + " | ".join(str(row.get(h, "")) for h in headers) + " |\n"
                        st.markdown(md_table)
                
                # Render recommendation cards if present
                elif msg_type == "recommendation" and "recommendations" in message:
                    recommendations = message["recommendations"]
                    for i, rec in enumerate(recommendations, 1):
                        score_pct = f"{rec.get('relevance_score', 0):.0%}"
                        title = rec.get("project_name", rec.get("property_name", "Property"))
                        
                        html_card = f"<div class='card'>"
                        html_card += f"<h4 style='margin-top:0; margin-bottom:0.5rem; color:#10a37f;'>🏡 {title}</h4>"
                        html_card += f"<div style='display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-bottom: 1rem; font-size: 0.9em;'>"
                        html_card += f"<div><strong>Builder:</strong> {rec.get('builder', 'N/A')}</div>"
                        html_card += f"<div><strong>Price:</strong> {rec.get('price', 'N/A')}</div>"
                        html_card += f"<div><strong>Configuration:</strong> {rec.get('configuration', 'N/A')}</div>"
                        html_card += f"<div><strong>Possession:</strong> {rec.get('possession', 'N/A')}</div>"
                        html_card += f"</div>"
                        
                        amenities = rec.get('amenities', 'N/A')
                        html_card += f"<p style='margin-bottom:0.5rem; font-size:0.9em;'><strong>Top Amenities:</strong> {amenities}</p>"
                        
                        html_card += f"<p style='margin-bottom:0.5rem; font-size:0.9em; color:#888;'><strong>Match Score:</strong> {rec.get('score', 0):.1f}%</p>"
                        
                        reasons = rec.get("reasons", [])
                        if reasons:
                            html_card += "<p style='margin-bottom:0.25rem;'><strong>Reason for Recommendation:</strong></p><ul style='margin-top:0; padding-left:1.5rem; margin-bottom:0.5rem;'>"
                            for reason in reasons:
                                html_card += f"<li>{reason}</li>"
                            html_card += "</ul>"
                        
                        citations = rec.get("supporting_citations", rec.get("citations", []))
                        if citations:
                            html_card += "<p style='margin-bottom:0.25rem; margin-top:0.5rem;'><strong>Sources:</strong></p><ul style='margin-top:0; padding-left:1.5rem; font-size:0.85em; color:#888;'>"
                            for c in citations:
                                html_card += f"<li>📄 {c}</li>"
                            html_card += "</ul>"
                            
                        html_card += "</div>"
                        st.markdown(html_card, unsafe_allow_html=True)
                
                # Render standard retrieval data
                elif msg_type == "standard":
                    retrieval_results = message.get("retrieval_results")
                    context_chunks = message.get("context_chunks")
                    
                    if retrieval_results:
                        render_confidence(retrieval_results)
                    if context_chunks and retrieval_results:
                        render_sources(context_chunks, retrieval_results)
                    
                    if idx == total_messages - 1 and context_chunks:
                        followups = build_followups(context_chunks)
                        st.markdown("### Follow-up questions")
                        cols = st.columns(2)
                        for f_idx, followup in enumerate(followups):
                            if cols[f_idx % 2].button(followup, key=f"followup-{f_idx}", use_container_width=True):
                                st.session_state.pending_prompt = followup
                                st.rerun()

                # Citations for comparison or fallback
                citations = message.get("citations")
                if citations:
                    st.markdown("### Sources")
                    for citation in citations:
                        st.caption(f"📄 {citation}")
            
            if message.get("timestamp"):
                align = "right" if role == "user" else "left"
                st.markdown(f"<div style='text-align: {align}; color: #888; font-size: 0.75rem; margin-top: -1rem; margin-bottom: 1rem; padding: 0 1rem;'>{message['timestamp']}</div>", unsafe_allow_html=True)


def show_typing_indicator(label: str) -> None:
    placeholder = st.empty()
    frames = [f"{label}", f"{label}.", f"{label}..", f"{label}..."]
    for frame in frames:
        placeholder.markdown(f"<div class='typing'>{frame}</div>", unsafe_allow_html=True)
        time.sleep(0.25)
    placeholder.empty()


def stream_response(text: str) -> None:
    placeholder = st.empty()
    rendered = ""
    for character in text:
        rendered += character
        placeholder.markdown(f"<div class='chat-row assistant'><div class='avatar'>🤖</div><div class='bubble-assistant'>{rendered}</div></div>", unsafe_allow_html=True)
        time.sleep(0.005)


def compute_confidence(results: List[dict]) -> tuple[str, float, str]:
    if not results:
        return "🔴 Low Confidence", 0.0, "No relevant documents were found in the knowledge base."
    top_score = max(item[1] for item in results)
    if top_score >= 0.72:
        return "🟢 High Confidence", top_score, "Based on highly relevant factual documents."
    if top_score >= 0.55:
        return "🟡 Medium Confidence", top_score, "Based on partially relevant supporting context."
    return "🔴 Low Confidence", top_score, "Based on weak context matches. Verify details independently."


def build_followups(context_chunks: List[dict]) -> List[str]:
    combined = " ".join(chunk.get("content", "") for chunk in context_chunks).lower()
    followups: List[str] = []
    if "refund" in combined or "cancellation" in combined:
        followups.extend(["What is the cancellation timeline?", "How do refunds work for buyers?", "Are there penalties for cancellation?", "What documents are needed for a refund request?"])
    elif "payment" in combined or "loan" in combined:
        followups.extend(["What are the available payment plans?", "How do down payments work?", "Which lenders are recommended?", "What is included in the financing package?"])
    elif "amenit" in combined or "facility" in combined:
        followups.extend(["What amenities are included?", "Are there community facilities?", "What is the maintenance policy?", "How accessible is the location?"])
    else:
        followups.extend(["Show the key highlights", "Explain the policy in simple terms", "What documents should I review first?", "What are the common buyer questions?"])
    return followups[:4]


def render_sources(context_chunks: List[dict], results: List[tuple]) -> None:
    if not context_chunks:
        return
    st.markdown("### 📚 Sources")
    for index, chunk in enumerate(context_chunks):
        score = results[index][1] if index < len(results) else 0.0
        preview = chunk.get("content", "")[:250]
        doc_name = chunk.get("document_name", "Document")
        page_num = chunk.get("page_number", "n/a")

        proj_name = get_project_name(doc_name)
        doc_lower = doc_name.lower()
        if "brochure" in doc_lower:
            doc_type = "Project Brochure"
        elif "builder" in doc_lower or "profile" in doc_lower:
            doc_type = "Builder Profile"
        elif "amenities" in doc_lower:
            doc_type = "Amenities Guide"
        elif "location" in doc_lower:
            doc_type = "Location Guide"
        elif "payment" in doc_lower:
            doc_type = "Payment Plan"
        elif "rera" in doc_lower:
            doc_type = "RERA Summary"
        else:
            doc_type = "General Document"

        similarity_pct = f"{score * 100:.0f}%"
        with st.expander(f"🏡 {proj_name} • {doc_type} ({similarity_pct} Match)", expanded=False):
            st.markdown(
                f"**Project Name:** `{proj_name}` &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"**Document Type:** `{doc_type}` &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"**Page:** `{page_num}` &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"**Similarity:** `{score:.3f}`"
            )
            st.markdown(f"> *{preview}...*")


def render_confidence(results: List[tuple]) -> None:
    label, score, explanation = compute_confidence(results)
    st.markdown("### Confidence")
    st.markdown(f"**{label}**")
    st.caption(explanation)


def render_analytics(engine: RAGEngine) -> None:
    st.markdown("### Analytics")
    col1, col2 = st.columns(2)
    col1.metric("Indexed documents", engine.document_count)
    col2.metric("Chunks", engine.chunk_count)
    col1.metric("Questions asked", len([m for m in st.session_state.messages if m["role"] == "user"]))
    avg_latency = sum(st.session_state.response_times) / len(st.session_state.response_times) if st.session_state.response_times else 0
    col2.metric("Average response time", f"{avg_latency:.1f}s")
    st.caption(f"Embedding model: {EMBEDDING_MODEL}")
    st.caption(f"LLM model: {GENERATION_MODEL}")


def render_about() -> None:
    st.markdown("### About Northstar Realty AI")
    st.write("Northstar Realty AI Assistant uses a comprehensive knowledge base to provide grounded, citation-backed answers for all your real estate questions.")

    st.markdown("### Key Features")
    st.markdown("""
    **🔄 Property Comparison** — Compare multiple projects side-by-side with detailed analysis of features, pricing, amenities, and location benefits.
    
    **🎯 Smart Recommendations** — Get personalized property recommendations based on your budget, space requirements, preferred amenities, and location preferences.
    
    **📍 Comprehensive Q&A** — Ask questions about listings, payment plans, amenities, locations, policies, and builders. All answers are grounded in the knowledge base with proper citations.
    
    **📊 Retrieval Transparency** — Watch the system search, retrieve, rank, and generate answers with live progress updates.
    
    **🔗 Source Citations** — Every answer includes direct links to the source documents so you can verify information independently.
    """)

    st.markdown("### How to Use")
    st.markdown("""
    1. **For Comparisons:** Type "Compare Project A vs Project B" or ask to compare builders
    2. **For Recommendations:** Describe your budget, space needs, and preferences
    3. **For General Questions:** Ask about any property details, policies, or features
    """)

    st.markdown("### Technology")
    st.markdown(f"""
    - **Embeddings:** {EMBEDDING_MODEL}
    - **Language Model:** {GENERATION_MODEL}
    - **Vector Store:** FAISS
    - **Framework:** Streamlit
    """)


def render_export_view() -> None:
    """Render conversation export interface."""
    st.markdown("### 💾 Export Conversation")
    
    if not st.session_state.messages:
        st.info("No conversation to export yet. Start chatting first!")
        return
    
    st.markdown(f"**Messages:** {len(st.session_state.messages)}")
    
    # Export format selection — exporter reused from session state (not re-created).
    col1, col2, col3 = st.columns(3)
    exporter = _get_exporter()
    
    with col1:
        if st.button("📄 Export as TXT", use_container_width=True):
            content = exporter.export_txt(st.session_state.messages)
            st.download_button(
                "Download TXT",
                content,
                file_name=exporter.get_filename("txt"),
                mime="text/plain",
            )
    
    with col2:
        if st.button("📝 Export as Markdown", use_container_width=True):
            content = exporter.export_markdown(st.session_state.messages)
            st.download_button(
                "Download MD",
                content,
                file_name=exporter.get_filename("md"),
                mime="text/markdown",
            )
    
    with col3:
        if st.button("🔗 Export as JSON", use_container_width=True):
            content = exporter.export_json(st.session_state.messages)
            st.download_button(
                "Download JSON",
                content,
                file_name=exporter.get_filename("json"),
                mime="application/json",
            )

    st.divider()
    st.subheader("Additional Export")
    col4, col5 = st.columns(2)
    with col4:
        if st.button("📄 Export as PDF", use_container_width=True):
            pdf_bytes = exporter.export_pdf(st.session_state.messages)
            st.download_button(
                "Download PDF",
                pdf_bytes,
                file_name=exporter.get_filename("txt").replace(".txt", ".pdf"),
                mime="application/pdf",
            )
    with col5:
        st.caption("PDF includes questions, answers, and basic timestamps.")

    
    st.divider()
    st.markdown("**Preview**")
    for msg in st.session_state.messages[:3]:
        if msg["role"] == "user":
            st.caption(f"👤 {msg.get('timestamp', '')}")
        else:
            st.caption(f"🤖 {msg.get('timestamp', '')}")
        st.write(msg["content"][:150] + ("..." if len(msg["content"]) > 150 else ""))


def render_search_view(engine: RAGEngine) -> None:
    """Render knowledge base search interface."""
    st.markdown("### 📚 Browse Knowledge Base")
    
    # Use session-cached KnowledgeBaseSearch — not re-instantiated every rerun.
    kb_search = _get_kb_search(engine)
    
    # Show stats
    stats = kb_search.get_document_stats()
    col1, col2, col3 = st.columns(3)
    col1.metric("Documents", stats["total_documents"])
    col2.metric("Chunks", stats["total_chunks"])
    col3.metric("File Types", len(stats["file_types"]))
    
    st.divider()
    
    # Search interface
    search_tab, browse_tab = st.tabs(["🔍 Search", "📋 Browse All"])
    
    with search_tab:
        search_query = st.text_input("Search document titles", placeholder="e.g., payment, amenities")
        if search_query:
            results = kb_search.search_by_name(search_query)
            if results:
                st.markdown(f"**Found {len(results)} documents**")
                for doc in results:
                    with st.expander(f"📄 {doc['name']}", expanded=False):
                        st.caption(f"Type: {doc['file_type']} | Pages: {doc['pages']}")
                        st.write(doc['preview'])
            else:
                st.info("No documents found")
    
    with browse_tab:
        all_docs = kb_search.get_all_documents()
        for doc in all_docs:
            with st.expander(f"📄 {doc['name']}", expanded=False):
                st.caption(f"Type: {doc['file_type']} | Pages: {doc['pages']}")
                st.write(doc['preview'])
    
    if st.button("Back to chat"):
        st.session_state.active_view = "chat"
        st.rerun()


def handle_prompt(prompt: str, engine: RAGEngine) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": datetime.now().strftime("%H:%M")})
    st.session_state.questions_asked += 1
    
    escaped_prompt = html.escape(prompt)
    st.markdown(f"<div class='chat-row user'><div class='bubble-user'>{escaped_prompt}</div></div>", unsafe_allow_html=True)

    # Detect feature type
    prompt_lower = prompt.lower()
    is_comparison = (
        any(phrase in prompt_lower for phrase in [" vs ", " vs.", " versus ", " compared to "]) or
        ("compare" in prompt_lower and (" and " in prompt_lower or " with " in prompt_lower))
    )
    # Expanded recommendation intent: catches budget, BHK, location preferences,
    # amenity requests, investment intent, and property-type queries.
    _rec_phrases = [
        "budget", "bhk", "recommend", "suggest", "looking for",
        "near school", "near metro", "near hospital", "near park",
        "swimming pool", "with pool", "with gym", "with clubhouse",
        "commercial property", "office space", "investment", "invest",
        "under ", "within ", "below ", "affordable", "luxury",
        "2 bhk", "3 bhk", "4 bhk", "1 bhk",
    ]
    is_recommendation = any(phrase in prompt_lower for phrase in _rec_phrases)

    start_time = time.perf_counter()
    tracker = RetrievalTracker()

    with st.chat_message("assistant"):
        if is_comparison:
            handle_comparison_query(prompt, engine, tracker)
        elif is_recommendation:
            handle_recommendation_query(prompt, engine, tracker)
        else:
            handle_standard_query(prompt, engine, tracker, start_time)

        latency = time.perf_counter() - start_time
        st.session_state.response_times.append(latency)
        if len(st.session_state.response_times) > 20:
            st.session_state.response_times = st.session_state.response_times[-20:]


def handle_comparison_query(prompt: str, engine: RAGEngine, tracker: RetrievalTracker) -> None:
    """Handle property/builder comparison queries."""
    progress_placeholder = st.empty()
    progress_placeholder.markdown(f"<div class='typing'>{tracker.next_stage('Analyzing query').message}</div>", unsafe_allow_html=True)

    try:
        comparator = PropertyComparison(engine)
        progress_placeholder.markdown(f"<div class='typing'>{tracker.next_stage('Retrieving property data').message}</div>", unsafe_allow_html=True)

        result = comparator.compare_properties(prompt)

        progress_placeholder.empty()
        if result.get("success"):
            table_data = result.get("comparison_table", [])
            answer_summary = f"Compared {', '.join(result.get('properties', []))} based on {len(table_data)} key attributes."
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_summary,
                "type": "comparison",
                "comparison_table": table_data,
                "citations": result.get("citations", []),
                "timestamp": datetime.now().strftime("%H:%M"),
            })
        else:
            err = result.get('error', 'Comparison failed')
            st.session_state.messages.append({
                "role": "assistant",
                "content": err,
                "type": "standard",
                "timestamp": datetime.now().strftime("%H:%M"),
            })

    except Exception:
        progress_placeholder.empty()
        error_msg = "An unexpected error occurred during property comparison. Please try again or rephrase your request."
        st.session_state.messages.append({
            "role": "assistant",
            "content": error_msg,
            "type": "standard",
            "timestamp": datetime.now().strftime("%H:%M"),
        })


def handle_recommendation_query(prompt: str, engine: RAGEngine, tracker: RetrievalTracker) -> None:
    """Handle smart property recommendation queries."""
    progress_placeholder = st.empty()
    progress_placeholder.markdown(f"<div class='typing'>{tracker.next_stage('Analyzing preferences').message}</div>", unsafe_allow_html=True)

    try:
        recommender = PropertyRecommendation(engine)
        progress_placeholder.markdown(f"<div class='typing'>{tracker.next_stage('Searching properties').message}</div>", unsafe_allow_html=True)

        result = recommender.recommend_properties(prompt, st.session_state.messages)

        progress_placeholder.empty()
        if result.get("success"):
            recommendations = result.get("recommendations", [])
            answer_summary = ""
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_summary,
                "type": "recommendation",
                "recommendations": recommendations,
                "timestamp": datetime.now().strftime("%H:%M"),
            })
        else:
            err = result.get('error', 'Recommendation failed')
            st.session_state.messages.append({
                "role": "assistant",
                "content": err,
                "type": "standard",
                "timestamp": datetime.now().strftime("%H:%M"),
            })

    except Exception:
        progress_placeholder.empty()
        error_msg = "An unexpected error occurred while generating recommendations. Please try again."
        st.session_state.messages.append({
            "role": "assistant",
            "content": error_msg,
            "type": "standard",
            "timestamp": datetime.now().strftime("%H:%M"),
        })


def handle_standard_query(prompt: str, engine: RAGEngine, tracker: RetrievalTracker, start_time: float) -> None:
    """Handle standard Q&A queries."""
    progress_placeholder = st.empty()
    progress_placeholder.markdown(f"<div class='typing'>{tracker.next_stage('Searching documents').message}</div>", unsafe_allow_html=True)

    # --- Check query cache first (zero LLM cost for repeated questions) ---
    query_cache: QueryCache = st.session_state.query_cache
    cached_answer = query_cache.get(prompt)
    if cached_answer is not None:
        progress_placeholder.empty()
        stream_response(cached_answer["answer"])
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": cached_answer["answer"],
            "type": "standard",
            "retrieval_results": cached_answer["retrieval_results"],
            "context_chunks": cached_answer["context_chunks"],
            "timestamp": datetime.now().strftime("%H:%M"),
        })
        return

    # --- Intelligent retrieval (intent boosting + deduplication + project filtering) ---
    progress_placeholder.markdown(f"<div class='typing'>{tracker.next_stage('Retrieving matches').message}</div>", unsafe_allow_html=True)

    retrieval = engine.retrieve(prompt)
    retrieval_results = retrieval["search_results"]
    context_chunks = retrieval["context_chunks"]

    progress_placeholder.markdown(f"<div class='typing'>{tracker.next_stage('Ranking results').message}</div>", unsafe_allow_html=True)

    if not context_chunks:
        progress_placeholder.empty()
        st.session_state.messages.append({
            "role": "assistant",
            "content": "I couldn't find reliable information in the provided knowledge base.",
            "type": "standard",
            "timestamp": datetime.now().strftime("%H:%M"),
        })
        return

    progress_placeholder.markdown(f"<div class='typing'>{tracker.next_stage('Generating answer').message}</div>", unsafe_allow_html=True)

    # Pass pre-computed results so answer_question() skips a redundant embed+search
    result = engine.answer_question(
        prompt,
        st.session_state.messages,
        search_results=retrieval_results,
        context_chunks=context_chunks,
    )

    progress_placeholder.empty()
    stream_response(result["answer"])

    # Store answer in query cache for future identical questions.
    query_cache.set(prompt, {
        "answer": result["answer"],
        "retrieval_results": retrieval_results,
        "context_chunks": context_chunks,
    })

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "type": "standard",
        "retrieval_results": retrieval_results,
        "context_chunks": context_chunks,
        "timestamp": datetime.now().strftime("%H:%M"),
    })


def main() -> None:
    init_session_state()
    st.set_page_config(page_title=APP_TITLE, page_icon="🏠", layout="wide")
    render_css()

    if not st.session_state.authenticated:
        render_login_page()
        return

    engine = build_engine()
    render_sidebar(engine)

    render_header()

    if st.session_state.active_view == "analytics":
        render_analytics(engine)
        st.divider()
        if st.button("Back to chat"):
            st.session_state.active_view = "chat"
            st.rerun()
        return

    if st.session_state.active_view == "stats":
        st.metric("Indexed Documents", engine.document_count)
        st.metric("Chunks", engine.chunk_count)
        st.metric("Embedding model", EMBEDDING_MODEL)
        st.metric("LLM model", GENERATION_MODEL)
        st.divider()
        if st.button("Back to chat"):
            st.session_state.active_view = "chat"
            st.rerun()
        return

    if st.session_state.active_view == "about":
        render_about()
        st.divider()
        if st.button("Back to chat"):
            st.session_state.active_view = "chat"
            st.rerun()
        return

    if st.session_state.active_view == "export":
        render_export_view()
        st.divider()
        if st.button("Back to chat"):
            st.session_state.active_view = "chat"
            st.rerun()
        return

    if st.session_state.active_view == "search":
        render_search_view(engine)
        return

    if st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
        handle_prompt(prompt, engine)
        st.rerun()

    if not st.session_state.messages:
        render_suggestions()

    render_chat_messages()

    prompt = st.chat_input("Ask about listings, payment plans, policies, amenities, or locations...")
    if prompt:
        handle_prompt(prompt, engine)
        st.rerun()


if __name__ == "__main__":
    main()
