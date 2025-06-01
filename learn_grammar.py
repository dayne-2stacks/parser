import argparse
import nltk
from nltk.grammar import Nonterminal
from nltk import induce_pcfg
from nltk.corpus import treebank

import re

_re_score   = re.compile(r"\s*\[[^\]]+\]$")
_re_lhs_punct = re.compile(r"^\s*['\"]?[.,?!]['\"]?\s*->")
_punct = {".", ",", "?", "!"}

def clean_grammar(inp: str, out: str) -> None:
    with open(inp) as fin, open(out, "w") as fout:
        next(fin)  # drop header
        for line in fin:
            if _re_lhs_punct.match(line):
                continue
            line = _re_score.sub("", line).strip()
            if "->" not in line:
                continue
            lhs, rhs = line.split("->", 1)
            lhs = lhs.strip()
            toks = re.findall(r"'[^']*'|\S+", rhs)
            toks = [t for t in toks if t.strip("'\"") not in _punct]
            fout.write(f"{lhs} -> {' '.join(toks)}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Induce a PCFG from the NLTK Treebank."
    )
    parser.add_argument(
        "--output",
        "-o",
        default="learned_grammar.cfg",
        help="Path to write the induced grammar"
    )
    parser.add_argument(
        "--num-trees",
        "-n",
        type=int,
        default=None,
        help="Limit to the first N trees (default: all)"
    )
    args = parser.parse_args()

    nltk.download("treebank", quiet=True)

    # gather productions
    productions = []
    parsed_sentences = treebank.parsed_sents()
    if args.num_trees:
        parsed_sentences = parsed_sentences[: args.num_trees]

    for t in parsed_sentences:
        productions.extend(t.productions())

    start = Nonterminal("S")
    grammar = induce_pcfg(start, productions)

    # write to file
    with open(args.output, "w") as out:
        out.write(str(grammar))
        
        clean_grammar(args.output, args.output + ".cleaned")


if __name__ == "__main__":
    main()
