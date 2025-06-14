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
        
        # Map non-terminals to the sequence of token ids that represent them in
        # the LLM vocabulary.  Some symbols (e.g. "NP-SBJ") are tokenised into
        # multiple tokens, so we keep the full sequence to compute probabilities
        # correctly when querying the model.
        self._nt_token_seqs = self._map_nonterminals_to_token_seqs()

    def _map_nonterminals_to_token_seqs(self) -> Dict[str, List[int]]:
        """Map non-terminals to their corresponding token id sequences."""
        nt_token_ids: Dict[str, List[int]] = {}
        for nt in self._nonterminals:
            # Add a space before the symbol to mimic generation from the model
            token_ids = self._llm.tokenizer.encode(f" {nt}", add_special_tokens=False)
            nt_token_ids[nt] = token_ids
            provider_logger.info(
                f"Non-terminal '{nt}' mapped to token IDs {token_ids}"
            )
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
                f"Given the possible nonterminal symbols {', '.join(self._nonterminals)}, "
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
            
            # Create a mask for the first token of every non-terminal symbol
            mask = torch.zeros_like(logits)
            for nt, token_seq in self._nt_token_seqs.items():
                first_token = token_seq[0] if token_seq else None
                logits_logger.info(
                    f"Non-terminal '{nt}' first token ID: {first_token}, logits: {logits[first_token] if first_token is not None else 'N/A'}"
                )
                if first_token is not None:
                    mask[first_token] = 1.0
            
            # Apply mask and softmax
            masked_logits = logits * mask
            masked_logits[mask == 0] = float('-inf')  # Set non-NT tokens to -inf
            probs = F.softmax(masked_logits, dim=0)

            # Convert to dictionary of probabilities using multi-token sequences
            span_probs: Dict[Nonterminal, float] = {}
            device = self._llm.model.device
            for nt, token_seq in self._nt_token_seqs.items():
                if not token_seq:
                    continue

                prob = probs[token_seq[0]].item()

                # If the symbol is represented by multiple tokens, generate the
                # follow-up tokens one by one and multiply their probabilities.
                if len(token_seq) > 1:
                    context = torch.cat(
                        [tokens.to(device), torch.tensor([[token_seq[0]]], device=device)],
                        dim=1,
                    )
                    for next_id in token_seq[1:]:
                        with torch.no_grad():
                            out = self._llm.model(context)
                            next_logits = out.logits[0, -1, :]
                        next_probs = F.softmax(next_logits, dim=0)
                        prob *= next_probs[next_id].item()
                        context = torch.cat(
                            [context, torch.tensor([[next_id]], device=device)],
                            dim=1,
                        )

                span_probs[Nonterminal(nt)] = prob
                logits_logger.info(
                    f"Probability for non-terminal '{nt}' with tokens {token_seq} in the span '{span_str}': {prob:.8f}"
                )

            # Optional: log top 5 candidates for debugging
            sorted_probs = sorted(span_probs.items(), key=lambda x: x[1], reverse=True)[:5]
            topk_results = [(nt.symbol(), p) for nt, p in sorted_probs]
            print(f"Top 5 candidates for span '{span_str}': {topk_results}")

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

