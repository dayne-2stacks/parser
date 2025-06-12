from __future__ import annotations
from typing import List, Dict, Tuple, Set
import logging
from nltk.grammar import Nonterminal
import torch
from torch.nn import functional as F

from local_llm import LocalLLM    

# Logger for provider events
provider_logger = logging.getLogger("provider")
provider_logger.setLevel(logging.INFO)
provider_handler = logging.FileHandler("logs/provider.log")
provider_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
provider_logger.addHandler(provider_handler)

# Logger for logits events
logits_logger = logging.getLogger("logits")
logits_logger.setLevel(logging.INFO)
logits_handler = logging.FileHandler("logs/logits.log")
logits_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
logits_logger.addHandler(logits_handler)


    
class TokenLevelProbabilityProvider:
    """
    Uses the LLM's token-level probabilities to calculate grammar rule probabilities.
    """
    
    def __init__(
        self, 
        llm: LocalLLM,
        nonterminals: Set[str],
        *,
        cache_size: int = 2048
        ):

        self._llm = llm
        self._nonterminals = nonterminals
        
        # Map non terminals to their token IDs in the LLM's vocabulary
        self._nt_token_ids = self._map_nonterminals_to_token_ids() 
        
    def _map_nonterminals_to_token_ids(self) -> Dict[str, int]:
        """
        Map non-terminals to their token IDs in the LLM's vocabulary.
        """
        nt_token_ids = {}
        for nt in self._nonterminals:
            # Add a space before the symbol
            token_ids = self._llm.tokenizer.encode(f" {nt}", add_special_tokens=False)
            # Use the first token ID if the symbol gets tokenized into multiple tokens
            # generative nonterminals will have the same root token ID
            nt_token_ids[nt] = token_ids[0] if token_ids else None
            provider_logger.info(f"Non-terminal '{nt}' mapped to token ID {nt_token_ids[nt]}")
        return nt_token_ids
    
    def get_span_probabilities(
        self, text: str, spans: List[Tuple[int, int]]
    ) -> Dict[Tuple[int, int], Dict[Nonterminal, float]]:
        """
        Get probabilities of nonterminals for each text span.
        """
        result = {}
        
        for start, end in spans:
            span_text = text.split()[start:end]
            span_str = " ".join(span_text)
            
            # Create prompt asking for the syntactic category
            prompt = (
                # f"Given the possible nonterminal symbols {', '.join(self._nonterminals)}, "
                "Using the nonterminal symbols of the Penn Treebank corpus, "
                f"the phrase '{span_str}' in the text '{text}' forms the "
                f"syntactic category of"
            )
            # print("Prompt: ",prompt)
            # Get logits from the LLM
            tokens = self._llm.tokenizer.encode(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self._llm.model(tokens.to(self._llm.model.device))
                logits = outputs.logits[0, -1, :]  
            
            # Create a mask for nonterminal tokens
            mask = torch.zeros_like(logits)
            for nt, token_id in self._nt_token_ids.items():
                logits_logger.info(f"Non-terminal '{nt}' token ID: {token_id}, logits: {logits[token_id] if token_id is not None else 'N/A'}")
                if token_id is not None:
                    mask[token_id] = 1.0
            
            # Apply mask and softmax
            masked_logits = logits * mask
            masked_logits[mask == 0] = float('-inf')  # Set non-NT tokens to -inf
            probs = F.softmax(masked_logits, dim=0)
            
            # Get top 5 candidates and their probabilities
            topk = torch.topk(probs, 5)
            topk_indices = topk.indices.tolist()
            topk_probs = topk.values.tolist()
            topk_tokens = [self._llm.tokenizer.decode([idx]) for idx in topk_indices]
            topk_results = list(zip(topk_tokens, topk_probs))
            print(f"Top 5 candidates for span '{span_str}': {topk_results}")
            
            # Convert to dictionary of probabilities
            span_probs = {}
            for nt, token_id in self._nt_token_ids.items():
                if token_id is not None:
                    prob = probs[token_id].item()
                    span_probs[Nonterminal(nt)] = prob
                    logits_logger.info(f"Probability for non-terminal '{nt}' with id {token_id}: {prob:.4f}")

            result[(start, end)] = span_probs
            provider_logger.info(
                f"Span: {span_text} probabilities: {span_probs}"
            )
        return result
    
    
    def set_text_and_precompute(self, tokens):
        """
        Set the current text being parsed and precompute span probabilities.
        This should be called before parsing.
        """
        self._text = " ".join(tokens)
        n = len(tokens)
        
        # Generate all possible spans
        spans = [(i, j) for i in range(n) for j in range(i+1, n+1)]
        
        # Get span probabilities from LLM
        self._span_probs = self.get_span_probabilities(self._text, spans)
        
        return self._span_probs
    


if __name__ == "__main__":
   
   llm = LocalLLM(model_name="llama3_1-70b")
   tokenizers = llm.tokenizer
   print(tokenizers.encode(" RBS", add_special_tokens=False)[0] == tokenizers.encode(" RRB", add_special_tokens=False)[0])

