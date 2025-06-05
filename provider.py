from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Sequence, Set
from nltk.grammar import Nonterminal, PCFG


import json
import math
import threading
from functools import lru_cache
from typing import Dict, Iterable, Sequence, Tuple

from nltk.grammar import Nonterminal
from local_llm import LocalLLM    
import torch
from torch.nn import functional as F
# noqa: your class

# external probability
class ExternalProbabilityProvider(ABC):
    @abstractmethod
    def probability_mass(
        self, lhs: Nonterminal, rhs_options: Sequence[Tuple]
    ) -> Dict[Tuple, float]:
        raise NotImplementedError
    


class LLMProbabilityProvider(ExternalProbabilityProvider):
    """
    Query a *local* HF model (wrapped by ``LocalLLM``) for a probability
    mass function over the RHSs of each grammar non-terminal.

    • The provider is **thread-safe** (one global lock around model-calls).  
    • Results are **LRU-cached** – repeated LHS queries cost zero compute.  
    • All outputs are validated & re-normalised before they reach the parser.
    """

    _MODEL_LOCK = threading.Lock()  

    def __init__(
        self,
        llm: LocalLLM,
        *,
        cache_size: int = 2048,
        temperature: float = 0.0,
    ):
        self._llm = llm
        self._temperature = temperature


    def probability_mass(
        self,
        lhs: Nonterminal,
        rhs_options: Sequence[Tuple],
    ) -> Dict[Tuple, float]:
        """Public wrapper that normalises args → hashable, then hits the cache."""
        rhs_key = tuple(rhs_options)       # ensure immutability
        return self._probability_mass_impl(lhs, rhs_key)


    def _probability_mass_impl(
        self,
        lhs: Nonterminal,
        rhs_options: Sequence[Tuple],
    ) -> Dict[Tuple, float]:

        rhs_strings = [" ".join(map(str, rhs)) for rhs in rhs_options]
        numbered = "\n".join(f"{i+1}. {txt}" for i, txt in enumerate(rhs_strings))

        user_msg = (
            "Given the context‐free grammar non-terminal below, you must assign\n"
            "a probability to every candidate right-hand side so that the\n"
            "values form a valid probability distribution (sum exactly to 1).\n\n"
            "it is important to consider the likelihood of each candidate so \n"
            "that it reflects real world probabilities\n\n"
            f"Non-terminal: {lhs}\n"
            "Candidates:\n"
            f"{numbered}\n\n"
            "Return only a JSON object mapping the integer index to the\n"
            "probability.  Example: {\"1\":0.5, \"2\":0.5}"
        )
        
        print(f"LLM prompt for {lhs!s}:\n{user_msg}\n")

        chat = [
            {
                "role": "system",
                "content": (
                    "You are a terse probability engine.  You never explain or add "
                    "text outside the JSON result."
                ),
            },
            {"role": "user", "content": user_msg},
        ]

        with LLMProbabilityProvider._MODEL_LOCK:
            max_tokens = min(1024, 512 + len(rhs_options) * 15)  
            raw = self._llm.get_response(chat, max_new_tokens=max_tokens)


        try:
            js = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"LLM returned invalid JSON for LHS={lhs!s}: {raw}"
            ) from e

        if not isinstance(js, dict) or not js:
            raise ValueError(
                f"LLM response must be a non-empty JSON object, got: {raw}"
            )


        probs: Dict[Tuple, float] = {}
        for i, rhs in enumerate(rhs_options, start=1):
            key = str(i)
            p = float(js.get(key, 0.0))
            if p < 0.0:
                raise ValueError(f"Negative probability for key {key}: {p}")
            probs[rhs] = p

        S = sum(probs.values())
        if math.isclose(S, 0.0):
            uniform = 1.0 / len(rhs_options)
            probs = {rhs: uniform for rhs in rhs_options}
            S = 1.0

        for rhs in probs:
            probs[rhs] /= S

        return probs

class TokenLevelProbabilityProvider(ExternalProbabilityProvider):
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
            # Add a space before the symbol as most tokenizers include the space
            token_ids = self._llm.tokenizer.encode(f" {nt}", add_special_tokens=False)
            # Use the last token ID if the symbol gets tokenized into multiple tokens
            nt_token_ids[nt] = token_ids[-1] if token_ids else None
            print(f"Non-terminal '{nt}' mapped to token ID {nt_token_ids[nt]}")
        return nt_token_ids
    
    def get_span_probabilities(
        self, text: str, spans: List[Tuple[int, int]]
    ) -> Dict[Tuple[int, int], Dict[Nonterminal, float]]:
        """
        Get probabilities of nonterminals for each text span.
        
        Args:
            text: The full text being parsed
            spans: List of (start, end) tuples indicating spans to analyze
        
        Returns:
            Dictionary mapping spans to probability distributions over nonterminals
        """
        result = {}
        
        for start, end in spans:
            span_text = text.split()[start:end]
            span_str = " ".join(span_text)
            
            # Create prompt asking for the syntactic category
            prompt = (
                f"The phrase '{span_str}' in the text '{text}' forms the "
                f"syntactic category of"
            )
            
            # Get logits from the LLM
            tokens = self._llm.tokenizer.encode(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self._llm.model(tokens.to(self._llm.model.device))
                logits = outputs.logits[0, -1, :]  # Get logits for last position
                
            # Create a mask for nonterminal tokens
            mask = torch.zeros_like(logits)
            for nt, token_id in self._nt_token_ids.items():
                if token_id is not None:
                    mask[token_id] = 1.0
            
            # Apply mask and softmax
            masked_logits = logits * mask
            masked_logits[mask == 0] = float('-inf')  # Set non-NT tokens to -inf
            probs = F.softmax(masked_logits, dim=0)
            
            # Convert to dictionary of probabilities
            span_probs = {}
            for nt, token_id in self._nt_token_ids.items():
                if token_id is not None:
                    prob = probs[token_id].item()
                    span_probs[Nonterminal(nt)] = prob
            
            result[(start, end)] = span_probs
            
        return result
    
    def probability_mass(
        self,
        lhs: Nonterminal,
        rhs_options: Sequence[Tuple],
    ) -> Dict[Tuple, float]:
        """
        Compute probabilities for each right-hand side option using token-level LLM probabilities.
        This is called by the parser to evaluate P(rhs|lhs) for rules.
        
        For token-level approach, we use the cached span probabilities computed by get_span_probabilities.
        """
        # We need span probabilities to be pre-computed
        if not hasattr(self, "_span_probs"):
            raise ValueError("Span probabilities must be pre-computed with get_span_probabilities before calling probability_mass")
        
        probs = {}
        for rhs in rhs_options:
            # This will be implemented by the TokenLevelViterbiParser that has context of spans
            # Here we provide a placeholder implementation
            probs[rhs] = 1.0 / len(rhs_options)
            
        return probs
    
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
    # Example usage
    llm = LocalLLM(model_name="llama3_1-70b")  # Replace with your model name
    provider = TokenLevelProbabilityProvider(llm, nonterminals={"S", "A", "B", "C"})

    lhs = Nonterminal("S")
    rhs_options = [(Nonterminal("A"), Nonterminal("B")), (Nonterminal("C"),)]
    
