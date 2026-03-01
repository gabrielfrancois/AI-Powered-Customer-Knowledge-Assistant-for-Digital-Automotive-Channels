"""
Run with: streamlit run src/app.py
"""

import sys
from pathlib import Path
import time

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

import streamlit as st

from src import config
from src.rag.chain import get_rag_chain


# -----------------------------------------------------------------------------
# 1. UI CONFIGURATION & SETUP
# -----------------------------------------------------------------------------

# Build the absolute path safely
logo_path = ROOT_DIR / "visual" / "bmw-logo.png"


def setup_page():
    """Configures the Streamlit page title, icon, and layout."""
    st.set_page_config(
    page_title="BMW AI Assistant",
    page_icon="🚗",
    layout="centered"
)
    st.title("🚗 BMW AI Assistant")
    st.markdown("**Your Intelligent BMW Product & Service Assistant**")

def render_sidebar() -> dict:
    """
    Renders the sidebar and returns user-configured settings.
    """
    with st.sidebar:
        if logo_path.exists():
            st.image(str(logo_path), width=100)
        else:
            # Fallback text if image is missing
            st.header("BMW AI")
        st.header("⚙️ Settings")
        
        # Retrieval Settings
        st.subheader("Retrieval Precision")
        top_k = st.slider(
            "Context Documents (Top-K)", 
            min_value=1, 
            max_value=10, 
            value=config.top_k,
            help="Number of documents to retrieve before re-ranking."
        )
        
        st.divider()
        
        # About Section
        st.info(
            "**Architecture:**\n"
            "- **LLM:** Llama-3.2-1B (4-bit)\n"
            "- **Embedding:** BGE-M3 (MPS/Metal)\n"
            "- **Reranker:** FlashRank (TinyBERT)\n"
            "- **Vector DB:** ChromaDB"
        )
        
        if st.button("Clear Chat History", type="secondary"):
            st.session_state.messages = []
            st.rerun()

    return {"top_k": top_k}

# -----------------------------------------------------------------------------
# 2. CHAT HISTORY MANAGEMENT
# -----------------------------------------------------------------------------

def initialize_session_state():
    """Initializes session state variables if they don't exist."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

def render_chat_history():
    """Iterates through session state and renders past messages."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Render sources if they exist (for assistant messages)
            if "sources" in message and message["sources"]:
                with st.expander("📚 Reference Documents Used"):
                    for src in message["sources"]:
                        st.markdown(f"- 📄 `{src}`")

# -----------------------------------------------------------------------------
# 3. CORE LOGIC (RAG PIPELINE)
# -----------------------------------------------------------------------------
def process_user_input(prompt: str, settings: dict):
    """
    Handles the user input, calls the RAG chain, and updates the UI.
    """
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate Assistant Response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("🧠 *Thinking... (Retrieving & Re-ranking)*")
        
        start_time = time.time()
        
        try:
            # Initialize the RAG Chain with user settings
            chain = get_rag_chain(top_k=settings["top_k"])
            
            # Execute Chain
            # Note: We pass the string directly because our chain handles it via RunnablePassthrough
            response = chain.invoke(prompt)
            
            answer = response["answer"]
            raw_docs = response["docs"]
            
            # Extract clean source filenames (deduplicated)
            # Logic: Get 'source' metadata -> split path -> take filename
            source_names = list(set(
                [doc.metadata.get("source", "Unknown").split("/")[-1] for doc in raw_docs]
            ))

            # Calculation time
            latency = time.time() - start_time
            
            # 3. Render Final Answer
            placeholder.markdown(answer)
            
            # 4. Render Sources
            if source_names:
                with st.expander(f"📚 Reference Documents ({len(source_names)})"):
                    for src in source_names:
                        st.markdown(f"- 📄 `{src}`")
            
            # 5. Save to History
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": source_names
            })
            
            # Optional: Show latency in sidebar or small caption
            # st.caption(f"⏱️ Response generated in {latency:.2f}s")

        except Exception as e:
            placeholder.error(f"❌ Error generating response: {str(e)}")
            # Print full trace to console for debugging
            import traceback
            traceback.print_exc()

# -----------------------------------------------------------------------------
# 4. MAIN APPLICATION ENTRY POINT
# -----------------------------------------------------------------------------
def main():
    setup_page()
    initialize_session_state()
    
    settings = render_sidebar()
    render_chat_history()

    # Chat Input Listener
    if prompt := st.chat_input("Ask about warranty, service, or vehicle features..."):
        process_user_input(prompt, settings)

if __name__ == "__main__":
    main()
