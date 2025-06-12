from nltk.grammar import Nonterminal, read_grammar, CFG
import re

_STANDARD_NONTERM_RE = re.compile(r"( [-?`*?\w/][\w/^<>-]* ) [\s\$?`?]*", re.VERBOSE)


def nonterm_parser(string, pos):
    m = _STANDARD_NONTERM_RE.match(string, pos)
    # print(m.group(1))
    if not m:
        raise ValueError("Expected a nonterminal, found: " + string[pos:])
    return (Nonterminal(m.group(1)), m.end())

class CustomCFG(CFG):
    def __init__(self, start, productions, calculate_leftcorner=True):
        super().__init__(start, productions, calculate_leftcorner=calculate_leftcorner)
        
    @classmethod
    def fromstring(cls, input, encoding=None):
        """
        Return the grammar instance corresponding to the input string(s).
        """
        start, productions = read_grammar(
            input, nonterm_parser, encoding=encoding
        )
        return cls(start, productions)
        
    
    