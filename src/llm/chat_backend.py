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

class MLXChatModel(LLM):
    """
    A custom LangChain wrapper for running LLMs locally via MLX.
    """

    model_id: str = config.MODEL_NAME
    model_dir: Path = config.MODEL_DIR
    model: Any = None
    tokenizer: Any = None
    gen_config: dict = config.GENERATION_CONFIG

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_model()

    def _load_model(self):
        """
        Downloads (if necessary) and loads the model into memory.
        """
        # Ensure directory exists
        if not self.model_dir.exists():
            print(green(f"Creating model directory at: {self.model_dir}"))
            self.model_dir.mkdir(parents=True, exist_ok=True)

        print(f"🔄 Checking for model: {self.model_id}...")
        model_path = snapshot_download(# Ensure it downloads to our custom path
            repo_id=self.model_id,
            local_dir=self.model_dir,
        ) 

        print(f"\n ⚡ Loading model into memory from {model_path}...")
        self.model, self.tokenizer = load(model_path)
        print(green("Model loaded successfully!"))

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
        
        print(yellow(f"\n[DEBUG] FINAL PROMPT SENT TO LLM:\n{'-'*20}\n{prompt}\n{'-'*20}\n"))
        
        if self.model is None:
            self._load_model()

        params = self.gen_config.copy()
        params.update(kwargs)
        
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        sampler = make_sampler(params.get("temp", config.TEMPERATURE))

        # Generate answer
        response = generate(
            self.model,
            self.tokenizer,
            prompt=formatted_prompt, # use the formatted prompt
            max_tokens=params.get("max_tokens", 512),
            sampler=sampler,
            verbose=params.get("verbose", False)
        )

        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_id": self.model_id}
