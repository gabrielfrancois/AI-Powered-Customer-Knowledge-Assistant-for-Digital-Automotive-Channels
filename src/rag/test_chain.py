import time
from src.rag.chain import get_rag_chain
from helper_function.prints import *

def main():
    print("🔗 Initializing RAG Chain...")
    
    # 1. Load the chain
    chain = get_rag_chain()
    print("✅ Chain initialized successfully.") 

    # 2. Define a test query
    query = "What is the warranty coverage for the high-voltage battery?"
    print(f"\n❓ Question: {query}")
    print("⏳ Generating answer..")

    # 3. Measure latency
    start_time = time.time()
    result = chain.invoke(query)
    end_time = time.time()
    
    print("\n🔎 RETRIEVAL DEBUG (What the LLM saw):")
    print("-" * 40)
    found_answer = False
    for i, doc in enumerate(result["docs"]):
        print(f"[{i+1}] {doc.metadata.get('source')} (Length: {len(doc.page_content)})")
        # Check if the text actually contains the battery info
        if "160,000" in doc.page_content or "8 years" in doc.page_content:
            print(f"    ✅ FOUND TARGET INFO HERE: {doc.page_content[:100]}...")
            found_answer = True
        else:
            print(f"    ❌ Irrelevant chunk: {doc.page_content[:50]}...")
    print("-" * 40)

    print("\n🤖 LLM Answer:")
    print(result["answer"])
    print(f"\n⏱️ Time taken: {end_time - start_time:.2f} s")

    if not found_answer:
        print("\n⚠️ CRITICAL: The retriever did NOT find the battery info.")
        print("Suggestion: Increase Chunk Overlap or check ingest.py")

if __name__ == "__main__":
    main()