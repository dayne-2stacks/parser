import logging

from typing import List, Dict, Any, Tuple, Sequence, Iterable
from nltk.grammar import Nonterminal, PCFG, ProbabilisticProduction as Production
from collections import defaultdict
from provider import ExternalProbabilityProvider, TokenLevelProbabilityProvider
from nltk.parse.viterbi import ViterbiParser
from nltk.tree import Tree, ProbabilisticTree
import math
from functools import reduce

# Logging
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())
    

class _Interpolator:
    def __init__(
        self,
        grammar:PCFG,
        provider: ExternalProbabilityProvider,
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
        ext_provider: ExternalProbabilityProvider,
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
        
    def parse(self, tokens):
        
        tokens = list(tokens)
        self._grammar.check_coverage(tokens)
        
        LOGGER.info("Precomputing span probabilities using llm")
        span_probs = self._token_provider.set_text_and_precompute(tokens)
        
        self.current_tokens = tokens
        
        constituents = {}
        
        if self._trace:
            print("Inserting tookens into the most likely constituents table")
            
        for index in range(len(tokens)):
            token = tokens[index]
            if self._trace:
                print(f"Token: {token}")
            constituents[index, index+1, token] = token
            if self._trace > 1:
                self._trace_lexical_insertion(token, index, len(tokens))
                
        
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
                    changed = True
    
    def _get_llm_probability(self, production, children, span):
        """
        Get the LLM probability of a production application
        """
        lhs = production.lhs()
        start, end = span
        
        # get the probability of the LHS for this span
        lhs_prob = self._token_provider._span_probs.get((start, end), {}).get(lhs.symbol(), 0.001)
        
        # Check for redundancy
        redundancy_penalty = 1.0
        if len(children) == 1 and isinstance(children[0], Tree) and lhs.symbol() == children[0].label():
            redundancy_penalty = 1e-7  # Heavily penalize unary redundancy
        
        if all(not isinstance(c, Tree) for c in children):
            return lhs_prob * redundancy_penalty
        
        child_probs = 1.0
        for child in children:
            if isinstance(child, Tree):
                # If the child is a tree, use its probability
                child_span = self._get_tree_span(child, span[0])
                child_lhs = Nonterminal(child.label())
                child_prob = self._token_provider._span_probs.get(child_span, {}).get(child_lhs, 0.001)
                child_probs *= child_prob
                
        # Avoid zero probabilities
        if lhs_prob == 0 or child_probs == 0:
            return 10e-12
            
        return lhs_prob * child_probs
    
    def _get_tree_span(self, tree, start_offset):
        """
        Calculate the span covered by a subtree given the start offset.
        
        This is a helper method to identify the span of text covered by a subtree.
        """
        # Calculate the span based on the number of leaf nodes
        length = len(tree.leaves())
        return (start_offset, start_offset + length)
        
