import os
# Prevent tokenizer deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import uuid
import time
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

import streamlit as st
from src import config
from src.rag.chain import get_rag_chain
from src.analytics.tracking import AnalyticsManager
from helper_function.prints import *

analytics = AnalyticsManager()
logo_path = ROOT_DIR / "visual" / "bmw-logo.png"

# -----------------------------------------------------------------------------
# 1. CACHING ENGINE
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="🧠 Starting up AI Engine...")
def load_cached_chain(top_k_value):
    """
    Loads the AI Model and Database into memory ONCE.
    """
    print("⚡ [APP] Cache Miss - Starting to load RAG Chain...")
    try:
        start_t = time.time()
        chain = get_rag_chain(top_k=top_k_value)
        end_t = time.time()
        print(f"[APP] RAG Chain loaded successfully in {end_t - start_t:.2f} seconds.")
        return chain
    except Exception as e:
        print(f"[APP] CRITICAL ERROR loading chain: {e}")
        raise e

# -----------------------------------------------------------------------------
# 2. UI SETUP
# -----------------------------------------------------------------------------
def setup_page():
    st.set_page_config(page_title="BMW AI", page_icon="🚗", layout="centered")
    st.title("🚗 BMW AI Assistant")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = set()
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

def render_sidebar():
    with st.sidebar:
        if logo_path.exists(): st.image(str(logo_path), width=100)
        st.header("⚙️ Settings")
        
        top_k = st.slider("Context Precision", 1, 10, getattr(config, 'top_k', 4))
        
        st.divider()
        st.info("**Model:** Llama-3.2-1B\n**Simulated Cost:** GPT-4o Pricing **Architecture:**\n"
            "- **Embedding:** BGE-M3 (Metal)\n"
            "- **Reranker:** FlashRank\n"
            "- **Vector DB:** ChromaDB")
        
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.feedback_given = set()
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()
    return top_k

# -----------------------------------------------------------------------------
# 3. DASHBOARD (UPDATED)
# -----------------------------------------------------------------------------
def render_dashboard():
    st.header("📊 Executive Analytics Dashboard")
    
    # Unpack values from tracking
    metrics, cat_counts, source_diversity, bad_sources = analytics.get_dashboard_metrics()
    
    if metrics is None:
        st.info("No data available yet. Chat with the bot to generate analytics!")
        return

    # KEY METRICS (Added Latency) ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Queries", metrics["total_queries"])
    c2.metric("Avg Latency", f"{metrics['avg_latency']:.2f}s") 
    c3.metric("Est. Cost (base: GPT-4o)", f"${metrics['est_cost']:.4f}")
    c4.metric("Avg Session Depth", f"{metrics['session_depth']:.1f} msgs")

    st.divider()

    # POPULAR SUBJECTS (User Intents) ---
    st.subheader("📌 Most Popular Subjects")
    if not cat_counts.empty:
        st.bar_chart(cat_counts.set_index("Category"))
    else:
        st.caption("No data yet.")

    st.divider()

    # KNOWLEDGE HEALTH & QUALITY 
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📚 Knowledge Coverage")
        if not source_diversity.empty:
            st.dataframe(
                source_diversity,
                column_config={
                    "Source": "Document Name",
                    "Share": st.column_config.ProgressColumn(
                        "Usage %",
                        help="Percentage of total citations coming from this source",
                        format="%.1f%%", 
                        min_value=0,
                        max_value=100,
                    ),
                    "Usage": "Citation Count"
                },
                hide_index=True,
                width='stretch'
            )
            # st.bar_chart(source_diversity.set_index("Source")["Share"])
            # st.caption("Which documents are used most often?")
        else:
            st.caption("No sources cited yet.")

    with col_right:
        st.subheader("⚠️ Problematic Sources")
        if not bad_sources.empty:
            # Filter to show only sources with negative feedback
            issues = bad_sources[bad_sources["Thumbs Down"] > 0].copy()
            
            if not issues.empty:
                st.dataframe(
                    issues[["Source File", "Thumbs Down", "Approval Rate"]].sort_values("Approval Rate").head(10),
                    hide_index=True,
                    width='stretch'
                )
                st.caption("Sources receiving negative feedback.")
            else:
                st.success("No negative feedback received yet!")
        else:
            st.info("No feedback data available.")

# -----------------------------------------------------------------------------
# 4. CHAT LOGIC
# -----------------------------------------------------------------------------
def process_chat(top_k):
    # Display History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Sources"):
                    for s in msg["sources"]: st.caption(s)
            
            # Feedback Buttons
            if msg["role"] == "assistant" and msg.get("msg_id"):
                msg_id = msg.get("msg_id")
                if msg_id not in st.session_state.feedback_given:
                    c1, c2, _ = st.columns([1, 1, 8])
                    with c1:
                        if st.button("👍", key=f"up_{msg_id}"):
                            analytics.log_feedback(msg_id, 1, "Like", msg["sources"])
                            st.session_state.feedback_given.add(msg_id)
                            st.rerun()
                    with c2:
                        if st.button("👎", key=f"down_{msg_id}"):
                            analytics.log_feedback(msg_id, 0, "Dislike", msg["sources"])
                            st.session_state.feedback_given.add(msg_id)
                            st.rerun()
                else:
                    st.caption("✅ Thank you for your feedback!")

    # Chat Input
    if prompt := st.chat_input("Ask about BMW..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            
            try:
                # 1. Load Chain (Cached)
                chain = load_cached_chain(top_k) 
                
                start = time.time()
                placeholder.markdown("🧠 *Thinking...*")
                
                # 2. Run Inference
                res = chain.invoke(prompt)
                latency = time.time() - start
                
                # 3. Process
                answer = res["answer"]
                context_text = "\n".join([d.page_content for d in res["docs"]])
                sources = list(set([d.metadata.get("source", "Unk").split("/")[-1] for d in res["docs"]]))

                msg_id = analytics.log_interaction(
                    st.session_state.session_id,
                    prompt, 
                    answer, 
                    sources, 
                    context_text,
                    latency
                )

                placeholder.markdown(answer)
                if sources:
                    with st.expander("📚 Sources"):
                        for s in sources: st.caption(s)

                st.session_state.messages.append({
                    "role": "assistant", "content": answer, "sources": sources, "msg_id": msg_id
                })
                st.rerun() 
                
            except Exception as e:
                placeholder.error(f"Error: {e}")

def main():
    setup_page()
    top_k = render_sidebar()
    
    if "rag_chain_loaded" not in st.session_state:
        load_cached_chain(top_k)
        st.session_state["rag_chain_loaded"] = True
    
    tab1, tab2 = st.tabs(["💬 Chat", "📈 Analytics"])
    
    with tab1:
        process_chat(top_k)
    with tab2:
        render_dashboard()

if __name__ == "__main__":
    main()