import os
from typing import Any, List, Optional, Mapping
from pathlib import Path
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from huggingface_hub import snapshot_download
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

from src import config
from helper_function.prints import * 

GLOBAL_MODEL = None
GLOBAL_TOKENIZER = None

class MLXChatModel(LLM):
    """
    A custom LangChain wrapper for running LLMs locally via MLX.
    """

    model_id: str = config.MODEL_NAME 
    model_dir: Path = config.MODEL_DIR
    gen_config: dict = config.GENERATION_CONFIG

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if GLOBAL_MODEL is not None:
            pass 

    def _load_model(self):
        """
        Downloads (if necessary) and loads the model into the GLOBAL cache.
        """
        global GLOBAL_MODEL, GLOBAL_TOKENIZER

        if GLOBAL_MODEL is not None:
            return

        if not self.model_dir.exists():
            print(green(f"Creating model directory at: {self.model_dir}"))
            self.model_dir.mkdir(parents=True, exist_ok=True)

        print(f"Checking for model: {self.model_id}...")
        try:
            model_path = snapshot_download(
                repo_id=self.model_id,
                local_dir=self.model_dir,
            )
        except Exception as e:
            raise RuntimeError(red(f"Failed to download model: {e}") )

        print(f"\n ⚡ Loading model into memory from {model_path}...")
        try:
            # Load locally first
            model, tokenizer = load(model_path)
            
            print("Warming up inference engine...")
            # from mlx_lm import generate
            
            warmup_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": "hi"}], 
                tokenize=False, 
                add_generation_prompt=True
            )
            # Warmup generation
            _ = generate(model, tokenizer, prompt=warmup_prompt, max_tokens=2, verbose=False)
            
            # SAVE TO GLOBAL CACHE
            GLOBAL_MODEL = model
            GLOBAL_TOKENIZER = tokenizer
            
            print(green("Model loaded successfully!"))
        except Exception as e:
            raise RuntimeError(red(f"Failed to load model: {e}"))

    @property
    def _llm_type(self) -> str:
        return "mlx_lm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        The core function that runs the generation.
        """
        global GLOBAL_MODEL, GLOBAL_TOKENIZER

        if GLOBAL_MODEL is None:
            self._load_model()
            
        print(yellow(f"\n[DEBUG] FINAL PROMPT SENT TO LLM:\n{'-'*20}\n{prompt}\n{'-'*20}\n"))

        params = self.gen_config.copy()
        params.update(kwargs)
        
        messages = [{"role": "user", "content": prompt}]
        
        # Use GLOBAL_TOKENIZER
        formatted_prompt = GLOBAL_TOKENIZER.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        sampler = make_sampler(params.get("temp", config.TEMPERATURE))

        # Use GLOBAL_MODEL
        response = generate(
            GLOBAL_MODEL,
            GLOBAL_TOKENIZER,
            prompt=formatted_prompt, 
            max_tokens=params.get("max_tokens", 512),
            sampler=sampler,
            verbose=params.get("verbose", False)
        )

        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_id": self.model_id}


"""
dummy class (uncomment to get a Linux/windows compatible model)
"""


# from langchain_community.llms import Ollama

# class MLXChatModel(LLM):
#     """
#     A wrapper that *looks* like the old MLX class but runs Ollama.
#     """

#     model_id: str = "llama3.2" # Default to a standard Ollama tag
#     client: Any = None
#     gen_config: dict = config.GENERATION_CONFIG

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self._load_model()

#     def _load_model(self):
#         """
#         Checks if Ollama is running and pulls the model if needed.
#         """
#         print(f"[Ollama] Checking for model: {self.model_id}...")
        
#         try:
#             # Check if ollama is accessible, we assume it's running on localhost:11434
#             pass 
#         except Exception:
#             print(red("Ollama is not running! Please run 'ollama serve' in a terminal."))
#             return

#         # Attempt to pull the model (automatic setup): mimics the 'snapshot_download' 
#         try:
#             print(f"⚡ [Ollama] Ensuring '{self.model_id}' is pulled...")
#             # We run this to ensure the model exists. If it exists, it's fast.
#             subprocess.run(["ollama", "pull", self.model_id], check=True)
#             print(green(f"Model '{self.model_id}' is ready."))
#         except FileNotFoundError:
#             print(red("Ollama CLI not found. Please install Ollama from ollama.com"))
#         except Exception as e:
#             print(red(f"Could not pull model automatically: {e}"))

#         # Initialize the LangChain Ollama client
#         self.client = Ollama(
#             model=self.model_id,
#             temperature=self.gen_config.get("temp", 0.1),
#             # Keep model in RAM for 1 hour for speed
#             keep_alive="1h" 
#         )

#     @property
#     def _llm_type(self) -> str:
#         return "ollama_wrapper"

#     def _call(
#         self,
#         prompt: str,
#         stop: Optional[List[str]] = None,
#         run_manager: Optional[CallbackManagerForLLMRun] = None,
#         **kwargs: Any,
#     ) -> str:
#         """
#         The core function that runs the generation via Ollama.
#         """
        
#         print(yellow(f"\n[DEBUG] FINAL PROMPT SENT TO OLLAMA:\n{'-'*20}\n{prompt}\n{'-'*20}\n"))
        
#         if self.client is None:
#             self._load_model()
#         # Ollama uses 'num_predict' instead of 'max_tokens'
#         if "max_tokens" in kwargs:
#             self.client.num_predict = kwargs["max_tokens"]

#         # Generate answer
#         try:
#             response = self.client.invoke(prompt)
#             return response
#         except Exception as e:
#             return f"Error communicating with Ollama: {e}"

#     @property
#     def _identifying_params(self) -> Mapping[str, Any]:
        # return {"model_id": self.model_id}