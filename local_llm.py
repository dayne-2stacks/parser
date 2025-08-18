import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from dotenv import load_dotenv
import os
from gpu_logging_utils import (
    log_gpu_memory_nvidia_smi,
    log_cuda_memory_pytorch,
    flush_cuda_cache,
)

from pathlib import Path

load_dotenv()
AUTH_TOKEN = os.getenv("AUTH_TOKEN")
HOME_DIR = os.getenv("HOME_DIR")
print(HOME_DIR)


PRETRAINED_TOKENIZERS = {
    "llama3-70b": "meta-llama/Meta-Llama-3-70B-Instruct",
    "llama3_1-70b": "meta-llama/Llama-3.1-70B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral-nemo": "mistralai/Mistral-Nemo-Instruct-2407",
    "gemma2-27b": "google/gemma-2-27b-it",
    "gpt-2": "openai-community/gpt2",
}

HF_MODEL_NAME_MAP = {
    "llama3_1-70b": "meta-llama/Llama-3.1-70B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma2-27b": "google/gemma-2-27b-it",
    "gpt-2": "openai-community/gpt2",
}

SAVE_LOCAL_MODELS_DIR = Path(f"{HOME_DIR}/local_llms/models/")
SAVE_LOCAL_TOKENIZERS_DIR = Path(f"{HOME_DIR}/local_llms/tokenizer/")


class LocalLLM:
    def __init__(self, model_name, *, quantization: str = "4bit", device_map="auto"):
        """Load a local language model with optional quantisation and offloading."""
        log_cuda_memory_pytorch("initialization_start")
        

        self.model_name = model_name
        self.hf_model_path = HF_MODEL_NAME_MAP[model_name]

        self.save_local_model_dir = SAVE_LOCAL_MODELS_DIR / model_name
        self.save_local_tokenizers_dir = SAVE_LOCAL_TOKENIZERS_DIR / model_name

        self.device_map = device_map

        quantization_config = None
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True
            )
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        if self.save_local_model_dir.exists():
            self.model = AutoModelForCausalLM.from_pretrained(
                self.save_local_model_dir,
                device_map=self.device_map,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.hf_model_path,
                quantization_config=quantization_config,
                device_map=self.device_map,
                token=AUTH_TOKEN,
                torch_dtype=torch.bfloat16,
            )
            self.model.save_pretrained(self.save_local_model_dir)

        if self.save_local_tokenizers_dir.exists():
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.save_local_tokenizers_dir,
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hf_model_path, token=AUTH_TOKEN
            )
            self.tokenizer.save_pretrained(self.save_local_tokenizers_dir)

        # Ensure the model stays in inference mode to avoid unnecessary
        # gradient allocations when parsing.
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def get_responses(self, chats, max_new_tokens=512, batch_size=1):
        """Generate responses for multiple chats with adjustable batch size."""
        responses = []
        for i in range(0, len(chats), batch_size):
            batch_chats = chats[i : i + batch_size]
            formatted_chats = [
                self.tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True
                )
                for chat in batch_chats
            ]
            inputs = self.tokenizer(
                formatted_chats,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            input_lengths = (
                inputs["input_ids"]
                .ne(self.tokenizer.pad_token_id)
                .sum(dim=1)
                .tolist()
            )
            inputs = {
                key: tensor.to(self.model.device) for key, tensor in inputs.items()
            }

            stop_token_ids = self.tokenizer.encode("}", add_special_tokens=False)[-1:]
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                eos_token_id=stop_token_ids[0],
                pad_token_id=self.tokenizer.eos_token_id,
            )

            for output, input_len in zip(outputs, input_lengths):
                decoded_output = self.tokenizer.decode(
                    output[input_len:], skip_special_tokens=True
                )
                if not decoded_output.strip().endswith("}"):
                    decoded_output = decoded_output.strip() + "}"
                responses.append(decoded_output)

        return responses

    def get_response(self, chat, max_new_tokens=1024):
        """Generate response with better JSON handling"""
        return self.get_responses([chat], max_new_tokens=max_new_tokens)[0]
    
    def get_token_cnt(self, prompt):
        tokens = self.tokenizer.tokenize(prompt)
        return len(tokens)
