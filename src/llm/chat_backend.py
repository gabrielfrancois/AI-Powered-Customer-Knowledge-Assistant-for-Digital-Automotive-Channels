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

class ModelRegistry:
    _model = None
    _tokenizer = None

    @classmethod
    def get_model(cls):
        """Returns (model, tokenizer). Loads them if they aren't ready."""
        if cls._model is not None:
            return cls._model, cls._tokenizer
        
        cls._load()
        return cls._model, cls._tokenizer

    @classmethod
    def _load(cls):
        """Internal loading logic"""
        model_id = config.MODEL_NAME
        model_dir = config.MODEL_DIR

        if not model_dir.exists():
            print(green(f"Creating model directory at: {model_dir}"))
            model_dir.mkdir(parents=True, exist_ok=True)

        print(f"🔄 Checking for model: {model_id}...")
        try:
            model_path = snapshot_download(
                repo_id=model_id,
                local_dir=model_dir,
            )
        except Exception as e:
            raise RuntimeError(red(f"Failed to download model: {e}"))

        print(f"\n⚡ Loading model into memory from {model_path}...")
        try:
            model, tokenizer = load(model_path)
            
            print("Warming up inference engine...")
            warmup_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": "hi"}], 
                tokenize=False, 
                add_generation_prompt=True
            )
            _ = generate(model, tokenizer, prompt=warmup_prompt, max_tokens=1, verbose=False)
            
            cls._model = model
            cls._tokenizer = tokenizer
            
            print(green("Model loaded and warmed up!"))
            
        except Exception as e:
            raise RuntimeError(red(f"Failed to load model: {e}"))

class MLXChatModel(LLM):
    """
    A custom LangChain wrapper for running LLMs locally via MLX.
    """

    model_id: str = config.MODEL_NAME 
    gen_config: dict = config.GENERATION_CONFIG

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ModelRegistry.get_model()

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
        model, tokenizer = ModelRegistry.get_model()

        params = self.gen_config.copy()
        params.update(kwargs)
        
        messages = [{"role": "user", "content": prompt}]
        
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        sampler = make_sampler(params.get("temp", config.TEMPERATURE))

        print(orange("Generate answer..."))
        response = generate(
            model,
            tokenizer,
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