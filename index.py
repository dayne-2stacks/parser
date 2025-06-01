import nltk
import spacy
# from cfg import CustomCFG
from nltk.parse.chart import ChartParser
from nltk.parse.pchart import BottomUpProbabilisticChartParser
from nltk.parse import ViterbiParser
from nltk.grammar import Nonterminal, read_grammar, PCFG
from nltk.tree import ProbabilisticTree, Tree
from nltk.tokenize import TreebankWordTokenizer


import os
import re


class PhraseChartParser(ChartParser):
    def parse(self, tokens):
        for tree in super().parse(tokens):
            phrases = [" ".join(sub.leaves()) for sub in tree.subtrees()]
            yield tree, phrases
            
class PhrasePChartParser(BottomUpProbabilisticChartParser):
    def parse(self, tokens):
        for tree in super().parse(tokens):
            phrases = [" ".join(sub.leaves()) for sub in tree.subtrees()]
            yield tree, phrases

class PhraseViterbiParser(ViterbiParser):
    
    def parse(self, tokens):
        tokens = list(tokens)
        n = len(tokens)
        self._grammar.check_coverage(tokens)

        # 1) Build the chart exactly as ViterbiParser does:
        constituents = {}
        if self._trace:
            print("Inserting tokens into the most likely constituents table...")
        for i, tok in enumerate(tokens):
            constituents[i, i+1, tok] = tok
            if self._trace > 1:
                self._trace_lexical_insertion(tok, i, n)

        for length in range(1, n+1):
            if self._trace:
                print(f"Finding constituents spanning length={length}...")
            for start in range(n-length+1):
                self._add_constituents_spanning((start, start+length), constituents, tokens)

        # 2) Check for a full-span parse
        S = self._grammar.start()
        full_tree = constituents.get((0, n, S))

        # 3) If there's no S spanning (0,n), collect diagnostics:
        if full_tree is None:
            print("\n⚠️  No complete parse for the full sentence with start symbol:", S)
            
            # a) Which nonterminals *did* cover the full span?
            full_symbols = {
                X for (s,e,X) in constituents 
                  if s==0 and e==n and isinstance(X, type(S))
            }
            if full_symbols:
                print("  But I *did* find constituents for full span (0→n) with nonterminals:")
                for X in full_symbols:
                    tree = constituents[0, n, X]
                    prob = getattr(tree, "prob", lambda: None)()
                    print(f"    • {X}  (prob={prob:.6g})")
            else:
                print("  No constituent at all covers the entire span 0→n.")

            # b) Now find the *longest* subsequence that *does* get recognized as an S
            spans = [
                (s,e) 
                for (s,e,X) in constituents 
                  if X == S and isinstance(constituents[s,e,X], ProbabilisticTree)
            ]
            if spans:
                # pick the one with largest (e−s)
                best_span = max(spans, key=lambda se: se[1]-se[0])
                best_tree = constituents[best_span[0], best_span[1], S]
                print(f"\n  Longest partial S-constituent spans tokens[{best_span[0]}:{best_span[1]}], "
                      f"prob={best_tree.prob():.6g}")
                print("    covered text →", " ".join(tokens[best_span[0]:best_span[1]]))
            else:
                print("\n  The start symbol S never appeared on any smaller span either.")
            
            # since there is no full parse, bail out
            return

        # 4) If we do have a full-span tree, just yield it (and phrases)
        phrases = [" ".join(t.leaves()) for t in full_tree.subtrees()]
        yield full_tree, phrases
            
    


def test_phrase_chart_parser():
    with open("induced_gramma.cfg", "r") as f:
        grammar_str = f.read()
    grammar = PCFG.fromstring(grammar_str)
    # grammar = PCFG(Nonterminal('ROOT'), grammar.productions())
    # print(grammar.start())
    parser = PhraseViterbiParser(grammar)
    # parser.trace(2)
    # parser = PhrasePChartParser(grammar)
    
    # tokens = "Pierre Vinken , 61 years old , will join the board as a nonexecutive director Nov. 29 .".split()
    # tokens = "Pierre Vinken bans taxes .".split()
    tokens = TreebankWordTokenizer().tokenize("Pierre bans taxes in America, but left home because he was scared.")
    # tokens = nltk.corpus.treebank.sents()[0]
    
    parsed = list(parser.parse(tokens))
    
    tree, phrases = parsed[0]
    
    print(tree) 
    
def transform_sentences_to_terminals(sentences):
    terminals = []
    for sent in sentences:
        tokens = nltk.word_tokenize(sent)
        pos_tags = nltk.pos_tag(tokens)
        sent_terminals = [f"{pos} -> {word}" for (word, pos) in pos_tags]
        terminals.append(sent_terminals)
    return terminals

if __name__ == "__main__":
    test_phrase_chart_parser()
    # nltk.download('punkt_tab')
    # nltk.download('averaged_perceptron_tagger_eng')
    #ans = transform_sentences_to_terminals(["The cat chased the dog", "The dog barked at the cat"])
    #print(ans)
    # path = os.path.dirname(os.path.abspath(__file__))
    # path = os.path.join(path, "test.txt")
    # with open(path, "r") as f:
    #     grammar_str = f.read()
    #     print(grammar_str)
        
    # _STANDARD_NONTERM_RE = re.compile(r"( -?[\w/][\w/^<>-]* ) \s*", re.VERBOSE)
    
    # lines =grammar_str.splitlines()
    # for line in lines:
    #     # print(line)
    #     ans = _STANDARD_NONTERM_RE.match(line)
    #     if ans:
    #         print(ans[0])
    #     else:
    #         continue
    
    # ans = _STANDARD_NONTERM_RE.match(grammar_str)
    # grammar = PCFG.fromstring(grammar_str)
    # print(ans[0]) 
    # print(grammar)
    # print(grammar.productions())
