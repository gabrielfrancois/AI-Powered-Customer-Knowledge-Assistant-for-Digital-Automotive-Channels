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

def setup_page():
    st.set_page_config(page_title="BMW AI", page_icon="🚗", layout="centered")
    st.title("🚗 BMW AI Assistant")
    
    # Initialize Session State
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
        
        # Info section
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

def process_chat(top_k):
    # Display History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Sources"):
                    for s in msg["sources"]: st.caption(s)

    # Input
    if prompt := st.chat_input("Ask about BMW..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

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

                # Display
                placeholder.markdown(answer)
                if sources:
                    with st.expander("📚 Sources"):
                        for s in sources: st.caption(s)

                st.session_state.messages.append({
                    "role": "assistant", "content": answer, "sources": sources, "msg_id": msg_id
                })
                st.rerun() # Force rerun to show buttons
            except Exception as e:
                placeholder.error(f"Error: {e}")

def main():
    setup_page()
    top_k = render_sidebar()
    
    tab1, tab2 = st.tabs(["💬 Chat", "📈 Analytics"])
    
    with tab1:
        process_chat(top_k)
        
        # Feedback Buttons
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            last_msg = st.session_state.messages[-1]
            msg_id = last_msg.get("msg_id")

            if msg_id:
                if msg_id not in st.session_state.feedback_given:
                    st.divider()
                    st.write("Rate this answer:")
                    c1, c2 = st.columns([1, 10])
                    with c1:
                        if st.button("👍", key=f"up_{msg_id}"):
                            analytics.log_feedback(msg_id, 1, "Like", last_msg["sources"])
                            st.session_state.feedback_given.add(msg_id)
                            st.rerun()
                    with c2:
                        if st.button("👎", key=f"down_{msg_id}"):
                            analytics.log_feedback(msg_id, 0, "Dislike", last_msg["sources"])
                            st.session_state.feedback_given.add(msg_id)
                            st.rerun()
                else:
                    st.caption("✅ Feedback recorded.")

    with tab2:
        render_dashboard()

if __name__ == "__main__":
    main()