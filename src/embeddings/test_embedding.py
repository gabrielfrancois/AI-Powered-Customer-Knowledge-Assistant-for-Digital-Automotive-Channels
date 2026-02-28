import time

from src.embeddings.bge_m3 import get_embedding_model

def main():
    print("--- 1. Initializing Embedding Model ---")
    start_load = time.time()
    embeddings = get_embedding_model()
    print(f"✅ Model loaded in {time.time() - start_load:.2f} seconds")

    print("\n--- 2. Testing Embedding Generation ---")
    text = "The BMW i4 has a range of up to 590 km."

    start_embed = time.time()
    vector = embeddings.embed_query(text)
    duration = time.time() - start_embed

    print(f"✅ Input text: '{text}'")
    print(f"✅ Vector length: {len(vector)}")
    # print(f"✅ vector : {vector}")
    print(f"✅ Time to embed: {duration:.4f} seconds")

    # Sanity check for dimensions
    if len(vector) == 1024:
        print("\n🎉 SUCCESS: Embeddings are working correctly!")
    else:
        print(f"\n⚠️ WARNING: Unexpected vector length: {len(vector)}")

if __name__ == "__main__":
    main()
