from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

from src.llm.chat_backend import MLXChatModel
from src.vectorstore.chroma_store import get_vectorstore
from src import config

def format_docs(docs):
    """Format retrieved documents into a string with citations."""
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown").split("/")[-1]
        content = doc.page_content.strip()
        formatted.append(f"SOURCE: {source}\nCONTENT: {content}")
    return "\n\n".join(formatted)

def get_rag_chain(top_k: int = config.top_k):
    """
    Builds the RAG chain: Retriever -> Prompt -> LLM
    """
    vectorstore = get_vectorstore(clean=False)
    
    # Search relevant information with HNSW (already built by src/rag/ingest.py)
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    llm = MLXChatModel(
        max_tokens=config.GENERATION_CONFIG["max_tokens"],
        temp=config.GENERATION_CONFIG["temp"]
    )
    
    # System prompt with chain of thought
    template = """You are a BMW Product Expert.
    
    TASK: Answer the user's question strictly based on the provided Reference Documents below.
    
    REFERENCE DOCUMENTS:
    {context}
    
    USER QUESTION: 
    {question}
    
    RESPONSE GUIDELINES:
    1. **Concept Matching:** The user may use different terminology than the documents. actively look for synonyms or related concepts (e.g., if asked about "high voltage", check sections on "electric" or "battery").
    2. **Precision:** If the documents provide specific figures (years, km, kW), cite them exactly.
    3. **Focus:** Answer only what is asked. Do not include irrelevant details from other sections (like extended warranties) unless requested.
    4. **Uncertainty:** If the answer is strictly not found in the documents, say "I do not have enough information to answer this based on the provided context."
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        RunnableParallel(
            {
                "context": retriever | format_docs, 
                "docs": retriever, # <--- Capture raw docs here
                "question": RunnablePassthrough()
            }
        )
        .assign(answer= prompt | llm | StrOutputParser()) # Generate answer
        .pick(["answer", "docs"]) # Return only answer and sources
    )
    
    return chain

# Helper for the UI to see sources
def get_retriever(top_k: int = config.top_k):
    vectorstore = get_vectorstore(clean=False)
    return vectorstore.as_retriever(search_kwargs={"k": top_k})