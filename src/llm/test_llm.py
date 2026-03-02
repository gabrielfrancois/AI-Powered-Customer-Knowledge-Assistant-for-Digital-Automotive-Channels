# test_llm.py
from src.llm.chat_backend import MLXChatModel

def main():
    print("Initializing MLX Model...")
    llm = MLXChatModel()
    
    print("\n--- Testing Generation ---")
    response = llm.invoke("Why is a BMW M3 a good car? Answer in one sentence.")
    
    print("\n--- Response ---")
    print(response)

if __name__ == "__main__": 
    main()
    