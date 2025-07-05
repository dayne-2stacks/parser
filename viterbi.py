import logging

from typing import List, Dict, Any, Tuple, Sequence, Iterable
from nltk.grammar import Nonterminal, PCFG, ProbabilisticProduction as Production
from collections import defaultdict
from provider import  TokenLevelProbabilityProvider
from nltk.parse.viterbi import ViterbiParser
from nltk.tree import Tree, ProbabilisticTree
import math
from functools import reduce

# Logging
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

# Logger for provider events
constituent_logger = logging.getLogger("constituent")
constituent_logger.setLevel(logging.INFO)
constituent_handler = logging.FileHandler("logs/constituent.log")
constituent_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
constituent_logger.addHandler(constituent_handler)

# Logger for logits events
# logits_logger = logging.getLogger("logits")
# logits_logger.setLevel(logging.INFO)
# logits_handler = logging.FileHandler("logits.log")
# logits_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
# logits_logger.addHandler(logits_handler)
    

class _Interpolator:
    def __init__(
        self,
        grammar:PCFG,
        provider: TokenLevelProbabilityProvider,
        *,
        theta: float = 0.8,
        prob_floor: float = 1e-12,
        ) -> None:
        if not 0.0 <= theta <= 1.0:
            raise ValueError("θ must be in [0, 1]")
        self._g = grammar
        self._provider = provider
        self._theta = theta
        self._floor = prob_floor
    
    def build(self) -> PCFG:
        """
        Build a new PCFG with interpolated probabilities.
        """
        by_lhs: Dict[Nonterminal, List[Production]] = defaultdict(list)
        for prod in self._g.productions():
            by_lhs[prod.lhs()].append(prod)
        new_prods: List[Production] = []

        for lhs, prods in by_lhs.items():
            rhs_options = [tuple(prod.rhs()) for prod in prods]
            ext_prob = self._provider.probability_mass(lhs, rhs_options)
            
            combined_probs: List[Tuple[Production, float]] = []
            for p in prods:
                pg = p.prob()
                pll = ext_prob.get(tuple(p.rhs()), 0.0)
                print(f"{lhs} -> {p.rhs()}: grammar prob = {pg}, external prob = {pll}")
                pc = self._theta * pg + (1.0 - self._theta) * pll
                if pc >= self._floor:
                    combined_probs.append((p, pc))
                    
                
            if not combined_probs:
                LOGGER.warning(
                    "All rules for %s fell below prob_floor; "
                    "reverting to original grammar weights.",
                    lhs,
                )
                new_prods.extend(prods)
                continue
            
            
            Z = sum(m for _, m in combined_probs)
            for prod, m in combined_probs:
                new_prods.append(Production(lhs, prod.rhs(), prob=m / Z))

        return PCFG(self._g.start(), new_prods)

class InterpolatingPhraseViterbiParser(ViterbiParser):

    def __init__(
        self,
        grammar: PCFG,
        ext_provider: TokenLevelProbabilityProvider,
        *,
        theta: float = 0.8,
        prob_floor: float = 1e-12,
        trace: int = 0,
    ):
        interpolated = _Interpolator(
            grammar,
            ext_provider,
            theta=theta,
            prob_floor=prob_floor,
        ).build()
        super().__init__(interpolated, trace=trace)


    def parse(self, tokens: Iterable[str]):

        for tree in super().parse(tokens):
            phrases = [" ".join(t.leaves()) for t in tree.subtrees()]
            yield tree, phrases
            
            
class TokenLevelViterbiParser(ViterbiParser):
    
    def __init__(
        self,
        grammar: PCFG,
        token_provider: TokenLevelProbabilityProvider,
        *,
        theta: float = 0.8,
        trace: int = 0,
    ):
        super().__init__(grammar, trace=trace)
        self._token_provider = token_provider
        self._theta = theta
        if not 0.0 <= theta <= 1.0:
            raise ValueError("Theta must be in [0, 1]")

        # Index lexical productions for quick lookup
        self._lexical_index: Dict[str, List[Production]] = defaultdict(list)
        for prod in grammar.productions():
            if len(prod.rhs()) == 1 and isinstance(prod.rhs()[0], str):
                self._lexical_index[prod.rhs()[0]].append(prod)
        
    def parse(self, tokens):
        
        tokens = list(tokens)

        # Only enforce full coverage if theta==1.0 (pure grammar)
        if self._theta == 1.0:
            self._grammar.check_coverage(tokens)

        span_probs = self._token_provider.set_text_and_precompute(tokens)

        self.current_tokens = tokens

        constituents = {}

        if self._trace:
            print("Inserting tookens into the most likely constituents table")

        for index in range(len(tokens)):
            token = tokens[index]
            if self._trace:
                print(f"Token: {token}")
            constituents[index, index + 1, token] = token
            if self._trace > 1:
                self._trace_lexical_insertion(token, index, len(tokens))

            # Grammar lexical productions for this token
            grammar_prods = self._lexical_index.get(token, [])
            llm_probs = span_probs.get((index, index + 1), {})

            for prod in grammar_prods:
                g_prob = prod.prob()
                llm_prob = llm_probs.get(prod.lhs(), 0.0)
                prob = self._theta * g_prob + (1.0 - self._theta) * llm_prob
                tree = ProbabilisticTree(prod.lhs().symbol(), [token], prob=prob)
                constituents[index, index + 1, prod.lhs()] = tree

            # Add LLM-only categories for tokens unseen in grammar
            if not grammar_prods:
                for nt, lprob in llm_probs.items():
                    prob = (1.0 - self._theta) * lprob
                    tree = ProbabilisticTree(nt.symbol(), [token], prob=prob)
                    constituents[index, index + 1, nt] = tree
                
        
        for length in range(1, len(tokens) +1):
            if self._trace:
                print(
                    "Finding the most likely constituents"
                    + " spanning %d text elements..." % length
                )
            for start in range(len(tokens) - length + 1):
                span = (start, start + length)
                self._add_constituents_spanning(span, constituents, tokens)
                
        tree = constituents.get((0, len(tokens), self._grammar.start()))
        if tree is not None:
            yield tree
            
    def _add_constituents_spanning(self, span, constituents, tokens):
        """
         Find constituents that might cover a span, using interpolated probabilities.
        """
        
        changed = True
        while changed:
            changed = False
            
            # Find all ways to instantiate grammar productions that cover span
            instantiations = self._find_instantiations(span, constituents)
            
            for production, children in instantiations:
                subtrees = [c for c in children if isinstance(c, Tree)]
                grammar_prob = reduce(lambda pr, t: pr*t.prob(), subtrees, production.prob())
                llm_prob = self._get_llm_probability(production, children, span)
                interpolated_prob = self._theta * grammar_prob + (1.0 - self._theta) * llm_prob
                constituent_logger.info(
                    f"Production '{production}' with children {children} has \n \
                    \t LLM probability {llm_prob:.8f} \n \
                    \t grammar probability {grammar_prob:.8f} \n \
                    \t interpolated probability {interpolated_prob:.8f} \n \
                    for span {span}."
                )
                
                node = production.lhs().symbol()
                tree = ProbabilisticTree(node, children, prob=interpolated_prob)
                
                c = constituents.get((span[0], span[1], production.lhs()))
                
                if self._trace > 1:
                    if c is None or c!= tree:
                        if c is None or c.prob() < tree.prob():
                            print("   Insert:", end=" ")
                        else:
                            print("  Discard:", end=" ")
                        self._trace_production(production, interpolated_prob, span, len(tokens))
                        print(f"  (Grammar: {grammar_prob:.8f}, LLM: {llm_prob:.8f})")
                        
                if c is None or c.prob() < tree.prob():
                    constituents[span[0], span[1], production.lhs()] = tree
                    constituent_logger.info(f"Inserting production '{production}' with probability {interpolated_prob:.8f} for span {span} into constituents table.")
                    changed = True
    
    def _get_llm_probability(self, production, children, span):
        """
        Get the LLM probability of a production application
        """
        lhs = production.lhs()
        start, end = span
        
        # get the probability of the LHS for this span
        lhs_prob = self._token_provider._span_probs.get((start, end), {}).get(Nonterminal(lhs.symbol()), 0)

        child_probs = 1.0
        for child in children:
            if isinstance(child, Tree):
                # If the child is a tree, use its probability
                child_span = self._get_tree_span(child, span[0])
                child_lhs = Nonterminal(child.label())
                constituent_logger.warning("Child label: %s, span: %s, child lhs: %s", child.label(), child_span, child_lhs)
                child_prob = self._token_provider._span_probs.get(child_span, {}).get(child_lhs, 0)
                child_probs *= child_prob

        return lhs_prob * child_probs

    def _get_tree_span(self, tree, start_offset):
        """
        Calculate the span covered by a subtree given the start offset.
        """
        # Calculate the span based on the number of leaf nodes
        length = len(tree.leaves())
        return (start_offset, start_offset + length)
        
