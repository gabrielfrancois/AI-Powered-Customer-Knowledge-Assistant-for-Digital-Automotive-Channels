# BMW AI Assistant 🚗

A high-performance, privacy-focused RAG (Retrieval-Augmented Generation) solution designed to assist BMW customer support and sales teams. This application runs entirely locally on **Apple Silicon M-series** chips, leveraging **MLX** for optimized inference and **FlashRank** for precision retrieval.

---

## 🚀 How to Run (Apple Silicon Only)

This project is strictly optimized for **macOS (M-series chips)** using the `uv` package manager for blistering fast environment setup.

### Prerequisites
* **Hardware:** Mac with Apple Silicon (M1/M2/M3/M4).
* **Software:** Python 3.10+ installed.

### Quick Start
1.  **Install uv** (if not installed):
    ```bash
    brew install uv
    ```

2.  **Run the Application:**
    Navigate to the project root and run:
    ```bash
    uv run main.py
    ```
    *The launcher will automatically handle dependency installation, check the knowledge base status, and launch the dashboard at `http://localhost:8501`.*

### Advanced Commands
* **Force Re-ingestion:** If you have added new text files to `data/knowledge_base`, force a database rebuild:
    ```bash
    uv run main.py --restart-ingestion
    ```

### ✨ One-Click Launch (The "Magic" Way)
For a seamless experience, use the included launcher script.

1.  **Make the script executable** (run this once):
    ```bash
    chmod +x BMW_get_started.command
    ```
2.  **Launch:**
    Simply **double-click** the `BMW_get_started.command` file in Finder. It will open a terminal, set up the environment, and launch the dashboard automatically.

---

## ⚙️ Architecture & Pipeline

This system uses a **Level 2 RAG Architecture** designed for high precision and low latency on edge devices.

### 1. Ingestion Layer (Adaptive HNSW)
* **Loader:** `TextLoader` is used to ingest raw technical manuals (.txt).
* **Chunking:** `RecursiveCharacterTextSplitter` ensures semantic integrity of technical specs.
* **Adaptive Indexing:** The system automatically configures the **HNSW (Hierarchical Navigable Small World)** parameters based on dataset size:
    * **Small Datasets (<1k chunks):** Uses high-fidelity settings (`M=64`, `ef=400`) to simulate Brute Force accuracy.
    * **Large Datasets:** Switches to balanced settings (`M=32`, `ef=200`) to maintain retrieval speed without sacrificing recall.

### 2. Retrieval Layer (The "Brain")
* **Vector Database:** **ChromaDB** stores embeddings locally for sub-millisecond similarity search.
* **Re-Ranking:** **FlashRank (TinyBERT)** acts as a critical second-stage filter.
    * **The "Screw-up" Scenario:** If a user asks about *"Charging faults"*, a standard vector search might blindly return documents about *Turbocharging* (engine air intake) because the word "charging" matches mathematically. The LLM would then hallucinate an answer about engine repair for an electric vehicle problem.
    * **The Fix:** FlashRank analyzes the context, realizes the intent is "Electrical/Battery", and forces the Turbocharger documents to the bottom, ensuring the LLM only sees relevant data.

### Why FlashRank (Cross-Encoder)?
Standard Vector Search (Bi-Encoder) calculates the "meaning" of the user query and the document **independently**. It compresses a whole paragraph into a single point in space, often losing fine details (e.g., confusing "battery charge" with "turbocharger").

**FlashRank (Cross-Encoder)** acts as a second-stage "reader."
* **The Funnel Strategy:** We use Vector Search to cast a wide net (Top 15 docs), then use FlashRank to deeply analyze the specific relationship between the User Question and those 15 candidates.
* **The Mechanism:** Unlike vector search, FlashRank inputs the Question and Document **together** into a neural network, allowing the model to see how words in the query directly interact with words in the document. This provides "human-level" relevance scoring for the final context window.

### 3. Generation Layer
* **LLM:** **Llama-3.2-1B-Instruct** (4-bit quantized).
* **Inference Engine:** **MLX** (Apple's Machine Learning framework) is used instead of PyTorch for unified memory access, providing ~3x faster generation on Mac devices.

### 4. Analytics & Feedback Loop
* **Tracking:** Every interaction logs latency, token usage, and user feedback.
* **Visualization:** A real-time Executive Dashboard monitors "Knowledge Health" (which documents are actually being used) and "Session Depth".

---

## ⚖️ Design Decisions & Trade-offs

We deliberately prioritized **speed** and **user experience** over complex retrieval pipelines for this iteration.

### Why no Query Decomposition?
Complex user questions often contain multiple intents (e.g., *"How do I check tire pressure and what is the warranty on the battery?"*).
* **The Ideal Solution:** Use an LLM agent to split this into two separate queries.
* **The Constraint:** Running multiple serial LLM inference calls on a base model **8GB computer** would cause unacceptable latency (10s+ wait times). We deliberately avoided curbing the pipeline speed, prioritizing a snappy, single-shot response over multi-intent handling for this edge prototype.

### Why no Query Expansion?
Query expansion (generating synonyms via LLM before searching) typically improves recall but **triples the latency**, as it requires multiple LLM calls per user query. To maintain a "real-time" chat feel (sub-second latency), we relied on **Dense Retrieval + Reranking** instead.

### Why no Hybrid Search (BM25)?
While BM25 is excellent for exact keyword matching (like error codes), it requires maintaining a separate sparse index. Given the high semantic density of the technical manuals, the **BGE-M3** embedding model combined with **FlashRank** provided sufficient precision without the added architectural complexity of a hybrid system.

### Future Improvements (Cross-Platform)
Currently, `src.llm.chat_backend` defaults to Apple `mlx` for maximum performance on Mac. To deploy this on **Linux or Windows** (NVIDIA GPUs):
* **The Fix:** Simply open `src/llm/chat_backend.py`, comment out the **MLX** class, and uncomment the **Ollama** class provided in the file. This instantly switches the inference engine to a universally compatible backend.

---

## 💼 Business Value & Why Local?

Why should a company deploy this specific architecture instead of using a generic Cloud LLM wrapper?

### 1. Data Privacy & Security (Local Execution)
* **Zero Data Leakage:** No data leaves the device. Customer PII and proprietary technical manuals never touch OpenAI or Google servers. This is critical for **GDPR compliance** and protecting trade secrets.

### 2. Strategic Data Mining (Free Market Research)
This system doesn't just answer questions, it collects a **goldmine of relevant data** that usually requires expensive surveys to acquire.
* **Identify Pain Points:** You can see exactly what users are struggling with most frequently (e.g., *"Why does everyone ask about Bluetooth pairing on the X5?"*).
* **Detect Outdated/Irrelevant Docs:** By tracking the "No Answer" rate and Thumbs Down feedback, the system flags documents that are **irrelevant, outdated, or confusing**.
* **Cost Savings:** Instead of paying for external audits or customer surveys, you get direct, organic feedback from your actual users every day.

### 3. Cost Control
* **Zero API Costs:** Running Llama-3 locally costs **$0.00** in token fees.
* **Scalability:** Deploying this on 100 customer support laptops requires no cloud infrastructure scaling, as the compute is distributed on the edge devices.

### 4. Uncluttering Support Centers
* By answering Tier 1 questions ("What is the tire pressure for X?" or "How to pair Bluetooth?") instantly and accurately, human agents are freed to handle complex, high-value relationship tasks. This reduces ticket volume and increases Customer Satisfaction (CSAT).