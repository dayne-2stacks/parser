from __future__ import annotations
from typing import List, Dict, Tuple, Set
import logging
import os
from nltk.grammar import Nonterminal
import torch
from torch.nn import functional as F

from local_llm import LocalLLM
from dataclasses import dataclass, field

os.makedirs("logs", exist_ok=True)

# Logger for provider events
provider_logger = logging.getLogger("provider")
provider_logger.setLevel(logging.INFO)
provider_handler = logging.FileHandler("logs/provider.log")
provider_handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s %(message)s")
)
provider_logger.addHandler(provider_handler)

# Logger for logits events
logits_logger = logging.getLogger("logits")
logits_logger.setLevel(logging.INFO)
logits_handler = logging.FileHandler("logs/logits.log")
logits_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
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
        self, llm: LocalLLM, nonterminals: Set[str], *, cache_size: int = 2048
    ):

        self._llm = llm
        self._nonterminals = nonterminals

        # Map non-terminals to the sequence of token ids that represent them in
        # the LLM vocabulary.  Some symbols (e.g. "NP-SBJ") are tokenised into
        # multiple tokens, so we keep the full sequence to compute probabilities
        # correctly when querying the model.
        self._nt_token_seqs = self._map_nonterminals_to_token_seqs()
        self._trie = self._build_trie()
        try:
            self._space_token_id = self._llm.tokenizer.encode(
                " ", add_special_tokens=False
            )[0]
        except Exception:
            self._space_token_id = None

    def _map_nonterminals_to_token_seqs(self) -> Dict[str, List[int]]:
        """Map non-terminals to their corresponding token id sequences."""
        nt_token_ids: Dict[str, List[int]] = {}
        for nt in self._nonterminals:
            # Add a space before the symbol to mimic generation from the model
            token_ids = self._llm.tokenizer.encode(f" {nt}", add_special_tokens=False)
            nt_token_ids[nt] = token_ids
            provider_logger.info(f"Non-terminal '{nt}' mapped to token IDs {token_ids}")
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

        # During parsing we only need forward activations without gradient
        # tracking. ``torch.inference_mode`` is slightly more efficient than
        # ``torch.no_grad`` and avoids allocating autograd buffers.
        with torch.inference_mode():
            out = self._llm.model(context)
            logits = out.logits[0, -1, :]
        token_probs = F.softmax(logits, dim=0)

        total_child_prob = 0.0
        for tok, child in node.children.items():
            p = token_probs[tok].item()
            if p <= 0:
                continue
            total_child_prob += p
            new_ctx = torch.cat([context, torch.tensor([[tok]], device=device)], dim=1)
            sub_nodes = self._predict_recursive(new_ctx, child, prob * p)
            for nt, v in sub_nodes.items():
                results[nt] = results.get(nt, 0.0) + v

        if self._space_token_id is not None:
            leftover = token_probs[self._space_token_id].item()
        else:
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
        history: Dict[Tuple[int, int], str] = {}

        for start, end in spans:
            span_text = text.split()[start:end]
            span_str = " ".join(span_text)

            context_parts = [f"{s}:{e}->{cat}" for (s, e), cat in history.items()]
            context_str = "; ".join(context_parts)

            prompt = (
                f"Given the possible nonterminal symbols {', '.join(self._nonterminals)}, "
                f"previously recognised spans: {context_str}. "
                "Using the nonterminal symbols of the Penn Treebank corpus, "
                f"the phrase '{span_str}' in the text '{text}' forms the "
                f"syntactic category of"
            )
            tokens = self._llm.tokenizer.encode(prompt, return_tensors="pt")
            device = self._llm.model.device
            span_probs_raw = self._predict_recursive(tokens.to(device), self._trie, 1.0)
            span_probs: Dict[Nonterminal, float] = {
                Nonterminal(nt): p for nt, p in span_probs_raw.items()
            }
            for nt, p in span_probs.items():
                logits_logger.info(
                    f"Probability for non-terminal '{nt.symbol()}' in the span '{span_str}': {p:.8f}"
                )

            sorted_probs = sorted(span_probs.items(), key=lambda x: x[1], reverse=True)[
                :5
            ]
            topk_results = [(nt.symbol(), p) for nt, p in sorted_probs]
            print(f"Top 5 candidates for span '{span_str}': {topk_results}")
            provider_logger.info(
                f"Top candidates for span '{span_str}': {topk_results}"
            )

            result[(start, end)] = span_probs
            if span_probs:
                best_nt = max(span_probs.items(), key=lambda x: x[1])[0]
                history[(start, end)] = best_nt.symbol()
            provider_logger.info(f"Span: {span_text} probabilities: {span_probs}")
        return result

    def set_text_and_precompute(self, tokens):
        """
        Set the current text being parsed and precompute span probabilities.
        This should be called before parsing.
        """
        self._text = " ".join(tokens)
        n = len(tokens)

        # Generate all possible spans
        spans: List[Tuple[int, int]] = [
            (start, start + length)
            for length in range(1, n + 1)
            for start in range(n - length + 1)
        ]
        spans.sort(key=lambda s: (s[1] - s[0], s[0]))

        # Get span probabilities from LLM
        self._span_probs = self.get_span_probabilities(self._text, spans)

        return self._span_probs

    def print_trie(self):
        """Print the structure of the trie for debugging purposes."""

        def _print_node(node, prefix="", token_id=None, depth=0):

            indent = "  " * depth
            token_text = ""
            if token_id is not None:
                try:
                    token_text = f" → '{self._llm.tokenizer.decode([token_id])}'"
                except Exception:
                    pass

            print(f"{indent}├── Token: {token_id}{token_text}")

            if node.nts:
                nt_indent = "  " * (depth + 1)
                print(f"{nt_indent}└── Non-terminals: {', '.join(node.nts)}")

            for token_id, child in sorted(node.children.items()):
                _print_node(child, prefix + "  ", token_id, depth + 1)

        print("Trie Structure:")
        print("Root")
        for token_id, child in sorted(self._trie.children.items()):
            _print_node(child, "", token_id, 1)


if __name__ == "__main__":
    from nltk.grammar import PCFG

    with open("grammar/induced_grammar.cfg", "r") as f:
        grammar_str = f.read()
    grammar = PCFG.fromstring(grammar_str)

    # Extract all nonterminals from the grammar
    nonterminals = {str(prod.lhs()) for prod in grammar.productions()}

    llm = LocalLLM(model_name="llama3_1-70b")

    # logger.info(f"Creating token level probability provider with nonterminals: {nonterminals}")
    provider = TokenLevelProbabilityProvider(llm, nonterminals, cache_size=2048)

    provider.print_trie()
