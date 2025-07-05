import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from dotenv import load_dotenv
import os

from pathlib import Path

load_dotenv()
AUTH_TOKEN = os.getenv('AUTH_TOKEN')
HOME_DIR = os.getenv('HOME_DIR')
print(HOME_DIR)


PRETRAINED_TOKENIZERS = {
    "llama3-70b": "meta-llama/Meta-Llama-3-70B-Instruct",
    "llama3_1-70b": "meta-llama/Llama-3.1-70B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral-nemo": "mistralai/Mistral-Nemo-Instruct-2407",
    "gemma2-27b": "google/gemma-2-27b-it",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}

HF_MODEL_NAME_MAP = {
    "llama3_1-70b": "meta-llama/Llama-3.1-70B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma2-27b": "google/gemma-2-27b-it",
}

SAVE_LOCAL_MODELS_DIR = Path(f"{HOME_DIR}/local_llms/models/")
SAVE_LOCAL_TOKENIZERS_DIR = Path(f"{HOME_DIR}/local_llms/tokenizer/")


class LocalLLM:
    def __init__(self, model_name):
        self.model_name = model_name
        self.hf_model_path = HF_MODEL_NAME_MAP[model_name]

        self.save_local_model_dir = SAVE_LOCAL_MODELS_DIR / model_name
        self.save_local_tokenizers_dir = SAVE_LOCAL_TOKENIZERS_DIR / model_name

        if self.save_local_model_dir.exists():
            self.model = AutoModelForCausalLM.from_pretrained(
                self.save_local_model_dir,
                device_map="auto"
            )
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            self.model = AutoModelForCausalLM.from_pretrained(self.hf_model_path, quantization_config=quantization_config, device_map="auto", token=AUTH_TOKEN)
            self.model.save_pretrained(self.save_local_model_dir)

        if self.save_local_tokenizers_dir.exists():
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.save_local_tokenizers_dir,
                padding_size="left"
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_path, padding_size="left", token=AUTH_TOKEN)
            self.tokenizer.save_pretrained(self.save_local_tokenizers_dir)


    def get_response(self, chat, max_new_tokens=1024):
        """Generate response with better JSON handling"""
        formatted_chat = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)
        inputs = {key: tensor.to(self.model.device) for key, tensor in inputs.items()}
        
        # Configure generation with stopping at closing brace
        stop_token_ids = self.tokenizer.encode("}", add_special_tokens=False)[-1:]
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            eos_token_id=stop_token_ids[0],  # Stop at closing brace
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        decoded_output = self.tokenizer.decode(outputs[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
        
        # Ensure the JSON has a closing brace
        if not decoded_output.strip().endswith('}'):
            decoded_output = decoded_output.strip() + '}'
            
        return decoded_output


    def get_token_cnt(self, prompt):
        tokens = self.tokenizer.tokenize(prompt)
        return len(tokens)
