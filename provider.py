from __future__ import annotations
from typing import List, Dict, Tuple, Set, Optional
import logging
import os
from gpu_logging_utils import (
    log_gpu_memory_nvidia_smi,
    log_cuda_memory_pytorch,
    flush_cuda_cache,
)
from nltk.grammar import Nonterminal, PCFG
import torch
from torch.nn import functional as F

from local_llm import LocalLLM
from dataclasses import dataclass, field

os.makedirs("logs", exist_ok=True)

# Logger for provider events
provider_logger = logging.getLogger("provider")
provider_logger.setLevel(logging.WARNING)
provider_handler = logging.FileHandler("logs/provider.log")
provider_handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s %(message)s")
)
provider_logger.addHandler(provider_handler)

# Logger for logits events
logits_logger = logging.getLogger("logits")
logits_logger.setLevel(logging.WARNING)
logits_handler = logging.FileHandler("logs/logits.log")
logits_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logits_logger.addHandler(logits_handler)

# Logger for performance metrics
performance_logger = logging.getLogger("performance")
performance_logger.setLevel(logging.INFO)
performance_handler = logging.FileHandler("logs/performance.log")
performance_handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s %(message)s")
)
performance_logger.addHandler(performance_handler)


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
        cache_size: int = 2048,
        precompute_batch_size: int = 5,
        batch_initial_step_only: bool = True,
        # Pruning controls
        filter_top_k: int | None = 10,
        filter_top_p: float | None = 0.95,
        prob_floor: float = 1e-4,
        # Router controls
        grammar: Optional[PCFG] = None,
        router_enabled: bool = False,
        router_lex_min_prob: float = 0.9,
        router_lex_ratio_threshold: float = 10.0,
    ):

        self._llm = llm
        self._nonterminals = nonterminals
        log_gpu_memory_nvidia_smi("provider init")
        log_cuda_memory_pytorch("provider init")

        # Map non-terminals to the sequence of token ids that represent them in
        # the LLM vocabulary.  Some symbols (e.g. "NP-SBJ") are tokenised into
        # multiple tokens, so we keep the full sequence to compute probabilities
        # correctly when querying the model.
        self._nt_token_seqs = self._map_nonterminals_to_token_seqs()
        self._trie = self._build_trie()
        flush_cuda_cache()
        try:
            self._space_token_id = self._llm.tokenizer.encode(
                " ", add_special_tokens=False
            )[0]
        except Exception:
            self._space_token_id = None

        # Metrics for performance logging
        self._token_generation_count = 0
        self._final_nt_token_count = 0
        self._prediction_count = 0

        # Batching configuration
        self._precompute_batch_size = max(1, precompute_batch_size)
        self._batch_initial_step_only = batch_initial_step_only

        # Pruning configuration
        self._filter_top_k = filter_top_k
        self._filter_top_p = filter_top_p
        self._prob_floor = max(0.0, prob_floor)

        # Router configuration and lexical index
        self._router_enabled = bool(router_enabled)
        self._router_lex_min = float(router_lex_min_prob)
        self._router_lex_ratio = float(router_lex_ratio_threshold)
        self._grammar: Optional[PCFG] = grammar
        self._lex_index: Dict[str, Dict[str, float]] = {}
        if self._grammar is not None:
            self._build_lex_index()

    def _build_lex_index(self) -> None:
        """Build token -> {lhs_symbol: prob(A -> 'token')} mapping from the PCFG."""
        lex_index: Dict[str, Dict[str, float]] = {}
        for prod in self._grammar.productions():
            rhs = prod.rhs()
            if len(rhs) == 1 and isinstance(rhs[0], str):
                word = rhs[0]
                lhs_sym = prod.lhs().symbol()
                lex_index.setdefault(word, {})[lhs_sym] = (
                    lex_index.get(word, {}).get(lhs_sym, 0.0) + float(prod.prob())
                )
        self._lex_index = lex_index

    def _router_span_probs_or_none(
        self, sentence_text: str, start: int, end: int
    ) -> Optional[Dict[Nonterminal, float]]:
        """
        Router heuristic: for single-token spans with a dominant grammar tag,
        return a grammar-only distribution to avoid LLM; otherwise None.
        """
        if not self._router_enabled or self._grammar is None or not self._lex_index:
            return None
        if end - start != 1:
            return None
        tokens = sentence_text.split()
        if not (0 <= start < len(tokens)):
            return None
        word = tokens[start]
        by_nt = self._lex_index.get(word)
        if not by_nt:
            return None
        total = sum(by_nt.values())
        if total <= 0:
            return None
        items = sorted(((nt, p / total) for nt, p in by_nt.items()), key=lambda x: x[1], reverse=True)
        top_nt, top_p = items[0]
        second_p = items[1][1] if len(items) > 1 else 0.0
        ratio = (top_p / second_p) if second_p > 0 else float("inf")
        if top_p >= self._router_lex_min or ratio >= self._router_lex_ratio:
            return {Nonterminal(nt): p for nt, p in items}
        return None

    def _map_nonterminals_to_token_seqs(self) -> Dict[str, List[int]]:
        """Map non-terminals to their corresponding token id sequences."""
        nt_token_ids: Dict[str, List[int]] = {}
        log_gpu_memory_nvidia_smi("map_nonterminals_start")
        log_cuda_memory_pytorch("map_nonterminals_start")
        for nt in self._nonterminals:
            # Add a space before the symbol to mimic generation from the model
            token_ids = self._llm.tokenizer.encode(f" {nt}", add_special_tokens=False)
            nt_token_ids[nt] = token_ids
            provider_logger.info(f"Non-terminal '{nt}' mapped to token IDs {token_ids}")
            # log_cuda_memory_pytorch(f"mapped {nt}")
        return nt_token_ids

    def _build_trie(self) -> _TrieNode:
        """Build a prefix trie of non-terminal token sequences."""
        # log_gpu_memory_nvidia_smi("build_trie_start")
        # log_cuda_memory_pytorch("build_trie_start")
        root = _TrieNode()
        for nt, seq in self._nt_token_seqs.items():
            node = root
            for tok in seq:
                node = node.children.setdefault(tok, _TrieNode())
            node.nts.append(nt)
            # log_cuda_memory_pytorch(f"trie_nt_{nt}")
        flush_cuda_cache()
        return root

    def _predict_recursive(
        self,
        context: torch.Tensor,
        node: _TrieNode,
        prob: float,
        depth: int = 0,
        result_cache: Dict[Tuple[int, Tuple[int, ...]], Dict[str, float]] = None,
        cached_token_probs: torch.Tensor | None = None,
    ) -> Dict[str, float]:
        if result_cache is None:
            result_cache = {}
        
        # Create a cache key from the last token ID and current node's children keys
        cache_key = (context[0, -1].item(), tuple(sorted(node.children.keys())))
        if cache_key in result_cache:
            return {k: v * prob for k, v in result_cache[cache_key].items()}
        
        log_cuda_memory_pytorch("predict_recursive_start")
        device = self._llm.model.device
        results: Dict[str, float] = {}

        if not node.children:
            for nt in node.nts:
                results[nt] = results.get(nt, 0.0) + prob
            return results
        
        log_cuda_memory_pytorch("before_model_call")
        # print(
        #     f"ctx shape: {context.shape}, "
        #     f"memory: {context.numel() * context.element_size() / (1024 * 1024):.4f} MB"
        # )

        # Count tokens generated by each model call
        if cached_token_probs is not None and depth == 0:
            token_probs = cached_token_probs
        else:
            self._token_generation_count += 1
            out = self._llm.model(context)
            logits = out.logits[0, -1, :]
            log_cuda_memory_pytorch("after_model_call")
            token_probs = F.softmax(logits, dim=0)
            # Free memory after getting token probabilities
            del out
        if depth % 3 == 0:  # Flush cache every 3 levels of recursion
            flush_cuda_cache()

        # Prepare children and their probabilities
        child_items = list(node.children.items())
        if child_items:
            child_tokens = torch.tensor([tok for tok, _ in child_items], device=token_probs.device, dtype=torch.long)
            child_probs = token_probs[child_tokens]
            # Sort children by probability descending
            sorted_probs, sorted_idx = torch.sort(child_probs, descending=True)

            # Determine how many to keep using top-p (preferred) or top-k
            keep_n = len(sorted_probs)
            if self._filter_top_p is not None:
                cumsum = torch.cumsum(sorted_probs, dim=0)
                # Number needed to reach top_p mass; ensure at least 1
                needed = int((cumsum < float(self._filter_top_p)).sum().item()) + 1
                keep_n = max(1, needed)
            elif self._filter_top_k is not None:
                keep_n = min(keep_n, int(self._filter_top_k))
            # Apply floor mask within the top region
            top_mask = torch.arange(len(sorted_probs), device=sorted_probs.device) < keep_n
            floor_mask = sorted_probs >= self._prob_floor
            combined_mask = top_mask & floor_mask
            if combined_mask.sum().item() == 0:
                # Ensure we keep at least the single highest-prob child
                combined_mask[0] = True

            kept_sorted_idx = sorted_idx[combined_mask]
            pruned_sorted_idx = sorted_idx[~combined_mask]

            kept_probs = child_probs[kept_sorted_idx]
            pruned_probs = child_probs[pruned_sorted_idx]

            kept_sum = float(kept_probs.sum().item())
            pruned_sum = float(pruned_probs.sum().item())

            # Iterate over kept children only, in small batches to limit memory
            kept_indices = kept_sorted_idx.tolist()
            batch_size = 5  # Adjust based on GPU memory
            for batch_idx in range(0, len(kept_indices), batch_size):
                batch_kept = kept_indices[batch_idx:batch_idx + batch_size]
                for idx in batch_kept:
                    tok, child = child_items[idx]
                    p = float(token_probs[tok].item())
                    if p <= 0:
                        continue

                    # Create new context tensor
                    new_ctx = torch.cat([context, torch.tensor([[tok]], device=device)], dim=1)

                    # Recursive call with depth tracking
                    sub_nodes = self._predict_recursive(
                        new_ctx,
                        child,
                        prob * p,
                        depth + 1,
                        result_cache,
                        cached_token_probs=None,
                    )

                    # Update results
                    for nt, v in sub_nodes.items():
                        results[nt] = results.get(nt, 0.0) + v

                    # Free memory
                    del new_ctx

                # Flush cache after each batch
                flush_cuda_cache()

        else:
            kept_sum = 0.0
            pruned_sum = 0.0

        # After pruning, absorb all remaining mass (including pruned and non-child tokens)
        # into the end-of-label path to preserve probability mass for shorter labels.
        leftover = max(1.0 - kept_sum, 0.0)

        for nt in node.nts:
            results[nt] = results.get(nt, 0.0) + prob * leftover
        
        # Cache the normalized results (without the prob multiplier)
        if prob > 0:
            result_cache[cache_key] = {k: v/prob for k, v in results.items()}
        
        return results

    def get_span_probabilities(
        self, text: str, spans: List[Tuple[int, int]]
    ) -> Dict[Tuple[int, int], Dict[Nonterminal, float]]:
        """
        Get probabilities of nonterminals for each text span.
        """
        log_gpu_memory_nvidia_smi("get_span_probabilities_start")
        log_cuda_memory_pytorch("get_span_probabilities_start")
        result = {}
        history: Dict[Tuple[int, int], str] = {}

        for start, end in spans:
            # Count a prediction for each span
            self._prediction_count += 1
            span_text = text.split()[start:end]
            span_str = " ".join(span_text)

            context_parts = [f"{s}:{e}->{cat}" for (s, e), cat in history.items()]
            context_str = "; ".join(context_parts)

            # Router: try grammar-only for confident single-token spans
            routed = self._router_span_probs_or_none(text, start, end)
            if routed is not None:
                span_probs: Dict[Nonterminal, float] = routed
            else:
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
            log_cuda_memory_pytorch(f"span_{start}_{end}")

            result[(start, end)] = span_probs
            if span_probs:
                best_nt = max(span_probs.items(), key=lambda x: x[1])[0]
                history[(start, end)] = best_nt.symbol()
                # Track tokens used in the final selected non-terminal
                self._final_nt_token_count += len(
                    self._nt_token_seqs.get(best_nt.symbol(), [])
                )
            provider_logger.info(f"Span: {span_text} probabilities: {span_probs}")
        log_gpu_memory_nvidia_smi("get_span_probabilities_end")
        log_cuda_memory_pytorch("get_span_probabilities_end")
        return result

    def _batched_last_token_probs(self, contexts: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Compute the softmax over vocabulary for the last non-padding token
        position for each context, using a single batched forward pass.

        Returns a list of 1D tensors (vocab-size) on the model device.
        """
        if not contexts:
            return []

        device = self._llm.model.device
        pad_id = getattr(self._llm.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self._llm.tokenizer, "eos_token_id", 0) or 0

        lengths = [c.shape[1] for c in contexts]
        max_len = max(lengths)

        batch_input = torch.full(
            (len(contexts), max_len),
            fill_value=pad_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros((len(contexts), max_len), dtype=torch.long, device=device)

        for i, c in enumerate(contexts):
            L = c.shape[1]
            batch_input[i, :L] = c.to(device)[0]
            attention_mask[i, :L] = 1

        # Single batched forward
        out = self._llm.model(input_ids=batch_input, attention_mask=attention_mask)
        logits = out.logits  # [B, T, V]

        probs_list: List[torch.Tensor] = []
        for i, L in enumerate(lengths):
            last_index = max(L - 1, 0)
            token_probs = F.softmax(logits[i, last_index, :], dim=0)
            probs_list.append(token_probs)

        # Clean up large tensors
        del out, logits, batch_input, attention_mask
        flush_cuda_cache()

        return probs_list

    def get_span_probabilities_batch(
        self, text: str, spans: List[Tuple[int, int]], *, batch_size: int | None = None
    ) -> Dict[Tuple[int, int], Dict[Nonterminal, float]]:
        """
        Batched version that reduces model calls by batching the first
        step (distribution over next token) across multiple span prompts.

        Note: For spans within the same batch, the prompt context for
        "previously recognised spans" is frozen to the history as of the
        start of the batch. History is then updated sequentially after
        computing the batch results.
        """
        log_gpu_memory_nvidia_smi("get_span_probabilities_batch_start")
        log_cuda_memory_pytorch("get_span_probabilities_batch_start")

        if batch_size is None:
            batch_size = self._precompute_batch_size

        result: Dict[Tuple[int, int], Dict[Nonterminal, float]] = {}
        history: Dict[Tuple[int, int], str] = {}
        device = self._llm.model.device

        for i in range(0, len(spans), batch_size):
            batch_spans = spans[i : i + batch_size]

            # Freeze history snapshot for this batch's prompt context
            context_parts = [f"{s}:{e}->{cat}" for (s, e), cat in history.items()]
            context_str = "; ".join(context_parts)

            prompts: List[str] = []
            contexts: List[torch.Tensor] = []
            span_strs: List[str] = []
            needs_llm_mask: List[bool] = []

            for start, end in batch_spans:
                span_text = text.split()[start:end]
                span_str = " ".join(span_text)
                span_strs.append(span_str)
                routed = self._router_span_probs_or_none(text, start, end)
                if routed is not None:
                    result[(start, end)] = routed
                    needs_llm_mask.append(False)
                else:
                    prompt = (
                        f"Given the possible nonterminal symbols {', '.join(self._nonterminals)}, "
                        f"previously recognised spans: {context_str}. "
                        "Using the nonterminal symbols of the Penn Treebank corpus, "
                        f"the phrase '{span_str}' in the text '{text}' forms the "
                        f"syntactic category of"
                    )
                    prompts.append(prompt)
                    tokens = self._llm.tokenizer.encode(prompt, return_tensors="pt")
                    contexts.append(tokens.to(device))
                    needs_llm_mask.append(True)

            # Batched first-step probabilities only for spans requiring LLM
            first_step_probs: List[torch.Tensor] = []
            if any(needs_llm_mask):
                first_step_probs = self._batched_last_token_probs(contexts)

            # Compute full distribution per span using cached first step
            llm_idx = 0
            for (start, end), span_str, need_llm in zip(batch_spans, span_strs, needs_llm_mask):
                # Count a prediction for each span (for metrics)
                self._prediction_count += 1
                if need_llm:
                    ctx = contexts[llm_idx]
                    tok_probs = first_step_probs[llm_idx]
                    llm_idx += 1
                    span_probs_raw = self._predict_recursive(
                        ctx, self._trie, 1.0, depth=0, result_cache=None, cached_token_probs=tok_probs
                    )
                    span_probs: Dict[Nonterminal, float] = {
                        Nonterminal(nt): p for nt, p in span_probs_raw.items()
                    }
                else:
                    span_probs = result[(start, end)]

                for nt, p in span_probs.items():
                    logits_logger.info(
                        f"Probability for non-terminal '{nt.symbol()}' in the span '{span_str}': {p:.8f}"
                    )

                sorted_probs = sorted(span_probs.items(), key=lambda x: x[1], reverse=True)[:5]
                topk_results = [(nt.symbol(), p) for nt, p in sorted_probs]
                print(f"Top 5 candidates for span '{span_str}': {topk_results}")
                provider_logger.info(
                    f"Top candidates for span '{span_str}': {topk_results}"
                )
                log_cuda_memory_pytorch(f"span_{start}_{end}")

                result[(start, end)] = span_probs

            # Update history in order using batch results
            for start, end in batch_spans:
                span_probs = result.get((start, end), {})
                if span_probs:
                    best_nt = max(span_probs.items(), key=lambda x: x[1])[0]
                    history[(start, end)] = best_nt.symbol()
                    self._final_nt_token_count += len(
                        self._nt_token_seqs.get(best_nt.symbol(), [])
                    )

            # Flush between batches
            flush_cuda_cache()

        log_gpu_memory_nvidia_smi("get_span_probabilities_batch_end")
        log_cuda_memory_pytorch("get_span_probabilities_batch_end")
        return result

    def set_text_and_precompute(self, tokens):
        """
        Set the current text being parsed and precompute span probabilities.
        This should be called before parsing.
        """
        log_gpu_memory_nvidia_smi("set_text_start")
        log_cuda_memory_pytorch("set_text_start")
        self._text = " ".join(tokens)
        n = len(tokens)

        # Reset metrics for this sentence
        self._token_generation_count = 0
        self._final_nt_token_count = 0
        self._prediction_count = 0

        # Generate all possible spans
        spans: List[Tuple[int, int]] = [
            (start, start + length)
            for length in range(1, n + 1)
            for start in range(n - length + 1)
        ]
        spans.sort(key=lambda s: (s[1] - s[0], s[0]))

        # Get span probabilities from LLM (batched first-step default)
        if self._precompute_batch_size > 1:
            self._span_probs = self.get_span_probabilities_batch(
                self._text, spans, batch_size=self._precompute_batch_size
            )
        else:
            self._span_probs = self.get_span_probabilities(self._text, spans)
        log_gpu_memory_nvidia_smi("set_text_end")
        log_cuda_memory_pytorch("set_text_end")

        # Log performance metrics
        if self._prediction_count:
            avg_generated = self._token_generation_count / self._prediction_count
            avg_final = self._final_nt_token_count / self._prediction_count
            ratio = (
                self._token_generation_count / self._final_nt_token_count
                if self._final_nt_token_count
                else 0.0
            )
        else:
            avg_generated = avg_final = ratio = 0.0
        performance_logger.info(
            f"Avg tokens generated per prediction: {avg_generated:.2f}; "
            f"Avg tokens in final non-terminal: {avg_final:.2f}; "
            f"Generation/Final ratio: {ratio:.2f}; "
            f"Total predictions: {self._prediction_count}"
        )

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
