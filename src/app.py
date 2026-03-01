import sys
import time
import pandas as pd
from pathlib import Path

# Path Setup
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

import streamlit as st
from src import config
from src.rag.chain import get_rag_chain
from src.analytics.tracking import AnalyticsManager

# Initialize
analytics = AnalyticsManager()
logo_path = ROOT_DIR / "visual" / "bmw-logo.png"

# -----------------------------------------------------------------------------
# UI HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def setup_page():
    st.set_page_config(page_title="BMW AI", page_icon="🚗", layout="centered")
    st.title("🚗 BMW AI Assistant")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = set()

def render_sidebar():
    with st.sidebar:
        if logo_path.exists(): 
            st.image(str(logo_path), width=100)
        else:
            st.header("BMW AI")

        st.header("⚙️ Settings")
        top_k = st.slider("Context Precision", 1, 10, getattr(config, 'top_k', 4))
        
        st.divider()
        st.info(
            "**Architecture:**\n"
            "- **LLM:** Llama-3.2-1B (4-bit)\n"
            "- **Embedding:** BGE-M3 (Metal)\n"
            "- **Reranker:** FlashRank\n"
            "- **Vector DB:** ChromaDB"
        )

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.feedback_given = set()
            st.rerun()
            
    return top_k

def render_dashboard():
    st.header("📊 Executive Analytics Dashboard")
    metrics, cat_counts, bad_sources = analytics.get_dashboard_metrics()
    
    if metrics is None:
        st.info("No data available yet.")
        return

    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Queries", metrics["total_queries"])
    c2.metric("Avg Latency", f"{metrics['avg_latency']:.2f} s")
    c3.metric("No Answer Rate", f"{metrics['no_answer_rate']:.1f}%")

    # Charts
    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader("📌 User Intents")
        if not cat_counts.empty:
            st.bar_chart(cat_counts.set_index("Category"))
    with c_right:
        st.subheader("⚠️ Problematic Sources")
        if not bad_sources.empty:
            st.dataframe(
                bad_sources[["Source File", "Rejection Rate", "Negative Feedback"]], 
                hide_index=True, use_container_width=True
            )
            st.caption("Rejection Rate = % of times source was cited in a 'Thumbs Down' answer.")
        else:
            st.success("No negative feedback recorded yet.")

# -----------------------------------------------------------------------------
# MAIN CHAT LOGIC
# -----------------------------------------------------------------------------

def process_chat(top_k):
    # Render History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Sources"):
                    for s in msg["sources"]: st.caption(s)
            
            # CHECK: If this is an assistant message, render its feedback buttons immediately
            if msg["role"] == "assistant":
                msg_id = msg.get("msg_id")
                if msg_id:
                    if msg_id not in st.session_state.feedback_given:
                        col1, col2, _ = st.columns([1, 1, 8])
                        with col1:
                            if st.button("👍", key=f"up_{msg_id}"):
                                analytics.log_feedback(msg_id, 1, "Like", msg["sources"])
                                st.session_state.feedback_given.add(msg_id)
                                st.rerun()
                        with col2:
                            if st.button("👎", key=f"down_{msg_id}"):
                                analytics.log_feedback(msg_id, 0, "Dislike", msg["sources"])
                                st.session_state.feedback_given.add(msg_id)
                                st.rerun()
                    else:
                        st.caption("✅ Feedback recorded.")

    # Chat Input
    if prompt := st.chat_input("Ask about BMW..."):
        # User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        # Assistant Message
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("🧠 *Thinking...*")
            start = time.time()
            try:
                # Run Chain
                chain = get_rag_chain(top_k=top_k)
                res = chain.invoke(prompt)
                latency = time.time() - start
                
                answer = res["answer"]
                sources = list(set([d.metadata.get("source", "Unk").split("/")[-1] for d in res["docs"]]))

                # Log
                msg_id = analytics.log_interaction(prompt, answer, sources, latency)

                # Display Answer
                placeholder.markdown(answer)
                if sources:
                    with st.expander("📚 Sources"):
                        for s in sources: st.caption(s)

                # Save to History
                st.session_state.messages.append({
                    "role": "assistant", "content": answer, "sources": sources, "msg_id": msg_id
                })
                st.rerun() 
            except Exception as e:
                placeholder.error(f"Error: {e}")

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    setup_page()
    top_k = render_sidebar()
    
    tab1, tab2 = st.tabs(["💬 Chat", "📈 Analytics"])
    
    with tab1:
        process_chat(top_k)

    with tab2:
        render_dashboard()

if __name__ == "__main__":
    main()