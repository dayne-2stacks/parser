from __future__ import annotations
from typing import List, Dict, Tuple, Set, Optional
import logging
from nltk.grammar import Nonterminal
import torch
from torch.nn import functional as F

from local_llm import LocalLLM
from dataclasses import dataclass, field

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


@dataclass
class _TrieNode:
    children: Dict[int, "_TrieNode"] = field(default_factory=dict)
    nts: List[str] = field(default_factory=list)


    
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
        self._trie = self._build_trie()

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

    def _build_trie(self) -> _TrieNode:
        """Build a prefix trie of non-terminal token sequences."""
        root = _TrieNode()
        for nt, seq in self._nt_token_seqs.items():
            node = root
            for tok in seq:
                node = node.children.setdefault(tok, _TrieNode())
            node.nts.append(nt)
        return root

    def _predict_recursive(
        self,
        context: torch.Tensor,
        node: _TrieNode,
        prob: float,
    ) -> Dict[str, float]:
        device = self._llm.model.device
        results: Dict[str, float] = {}

        if not node.children:
            for nt in node.nts:
                results[nt] = results.get(nt, 0.0) + prob
            return results

        with torch.no_grad():
            out = self._llm.model(context)
            logits = out.logits[0, -1, :]
        token_probs = F.softmax(logits, dim=0)

        total_child_prob = 0.0
        for tok, child in node.children.items():
            p = token_probs[tok].item()
            if p <= 0:
                continue
            total_child_prob += p
            new_ctx = torch.cat(
                [context, torch.tensor([[tok]], device=device)], dim=1
            )
            sub = self._predict_recursive(new_ctx, child, prob * p)
            for nt, v in sub.items():
                results[nt] = results.get(nt, 0.0) + v

        leftover = max(1.0 - total_child_prob, 0.0)
        for nt in node.nts:
            results[nt] = results.get(nt, 0.0) + prob * leftover

        return results
    
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
            tokens = self._llm.tokenizer.encode(prompt, return_tensors="pt")
            device = self._llm.model.device
            span_probs_raw = self._predict_recursive(tokens.to(device), self._trie, 1.0)
            span_probs: Dict[Nonterminal, float] = {Nonterminal(nt): p for nt, p in span_probs_raw.items()}
            for nt, p in span_probs.items():
                logits_logger.info(
                    f"Probability for non-terminal '{nt.symbol()}' in the span '{span_str}': {p:.8f}"
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

