from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from flashrank import Ranker, RerankRequest # To allow Re-Ranking strategy 

from src.llm.chat_backend import MLXChatModel
from src.vectorstore.chroma_store import get_vectorstore
from src import config

def rerank_docs(docs, query):
    """
    Re-ranks retrieved documents using a cross-encoder (FlashRank).
    1. Retrieval gets 'broad' matches (Top-15).
    2. Reranker selects 'precise' matches (Top-3).
    Args:
        docs (List[Document]): 
            List of LangChain `Document` objects retrieved from the vector store.
            Each document must contain:
                - page_content (str)
                - metadata (dict)
        query (str): 
            The user query used to re-score the retrieved documents.
    Returns:
        List[Document]:
            A list of re-ranked LangChain `Document` objects,
            sorted by decreasing relevance score,
            truncated to `config.TOP_K`.

            Returns an empty list if `docs` is empty.
    """
    if not docs:
        return []
    
    # Initialize Ranker (uses a tiny model ~40MB, runs on CPU/MPS fast) TODO: in production maybe take a better model if we've computational capacities 
    ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir=config.MODEL_DIR)
    
    # Convert LangChain Docs to FlashRank format
    passages = [
        {"id": str(i), "text": doc.page_content, "meta": doc.metadata} 
        for i, doc in enumerate(docs)
    ]
    
    rerank_request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(rerank_request)
    
    final_docs = []
    for res in results[:config.top_k]: # Pick only the best ones (the top_k)
        final_docs.append(Document(page_content=res["text"], metadata=res["meta"]))
        
    return final_docs

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
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k*3}) # 3x the documents we actually need, then we'll keep only top_k (re_rank)

    llm = MLXChatModel(
        max_tokens=config.GENERATION_CONFIG["max_tokens"],
        temp=config.GENERATION_CONFIG["temp"]
    )
    
    def smart_retrieval(query:str):
        initial_docs = base_retriever.invoke(query) # first search (classical HNSW)
        ranked_docs = rerank_docs(initial_docs, query) # Re-ranking
        return ranked_docs
    
    # System prompt (+user_prompt) with chain of thought
    template = """You are a BMW Product Expert.
    
    TASK: Answer the user's question strictly based on the provided Reference Documents below.
    
    REFERENCE DOCUMENTS:
    {context}
    
    RESPONSE GUIDELINES:
    1. **Concept Matching:** The user may use different terminology than the documents. actively look for synonyms or related concepts (e.g., if asked about "high voltage", check sections on "electric" or "battery").
    2. **Precision:** If the documents provide specific figures (years, km, kW), cite them exactly.
    3. **Focus:** Answer only what is asked. Do not include irrelevant details from other sections (like extended warranties) unless requested.
    4. **Uncertainty:** If the answer is strictly not found in the documents, say "I do not have enough information to answer this based on the provided context."
    
    USER QUESTION: 
    {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        RunnableParallel(
            {
                "docs": RunnablePassthrough() | smart_retrieval, # <--- Capture raw docs here
                "question": RunnablePassthrough()
            }
        )
        .assign(context=lambda x:format_docs(x["docs"])) # format re-ranked docs
        .assign(answer= prompt | llm | StrOutputParser())
        .pick(["answer", "docs"]) # Return only answer and sources
    )
    
    return chain

# Helper for the UI to see sources
def get_retriever(top_k: int = config.top_k):
    vectorstore = get_vectorstore(clean=False)
    return vectorstore.as_retriever(search_kwargs={"k": top_k})