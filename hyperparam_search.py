import itertools
import json
import time
from typing import Dict, Any, List
import torch
import yaml
import nltk
from collections import defaultdict
from nltk import Nonterminal, induce_pcfg, ProbabilisticProduction, Production
from nltk.tree import Tree
from nltk.parse import ViterbiParser
from viterbi import TokenLevelViterbiParser
from provider import TokenLevelProbabilityProvider
from nltk.tokenize import TreebankWordTokenizer
from local_llm import LocalLLM
import matplotlib.pyplot as plt
import os


def get_constituents(tree: Tree) -> set:

    constituents = set()
    leaves_pos = {}

    leaves = tree.leaves()
    for i, leaf in enumerate(leaves):
        for pos in tree.leaf_treepositions(i):
            leaves_pos[pos] = i

    for position in tree.treepositions():
        if isinstance(tree[position], Tree):
            if len(tree[position].leaves()) > 1:
                subtree = tree[position]
                start = leaves_pos[subtree.leaf_treeposition(0)]
                end = leaves_pos[subtree.leaf_treeposition(len(subtree.leaves()) - 1)] + 1
                label = subtree.label()
                constituents.add((label, start, end))
    return constituents


def load_config(path: str) -> Dict[str, List[Any]]:
    """Load a config file in either JSON or YAML"""
    with open(path, "r") as f:
        if path.endswith(".json"):
            data = json.load(f)
        else:
            data = yaml.safe_load(f)
    return data


def get_treebank_splits(dev_ratio: float = 0.1, test_ratio: float = 0.1) -> (List[Tree], List[Tree], List[Tree]):
    """Get train, dev and test splits of the NLTK Treebank corpus"""
    # Get all the trees from the NLTK Treebank corpus
    trees = list(nltk.corpus.treebank.parsed_sents())
    split_dev = int(len(trees) * (1 - dev_ratio - test_ratio))
    train_trees = trees[:split_dev]
    dev_trees = trees[split_dev : -int(len(trees) * test_ratio)] if test_ratio else trees[split_dev:]
    test_trees = trees[-int(len(trees) * test_ratio) :] if test_ratio else []
    return train_trees, dev_trees, test_trees


def build_pcfg(train_trees: List[Tree]):
    """Induce a PCFG from training trees with sanitized non-terminals."""

    with open(os.path.join("grammar", "changes.json"), "r") as f:
        special_nt_map = json.load(f)

    def _format_nonterminal(nt: Nonterminal) -> Nonterminal:
        sym = nt.symbol()

        if sym.startswith("-") and sym.endswith("-") and len(sym) > 1:
            sym = sym[1:-1]

        if sym in special_nt_map:
            sym = special_nt_map[sym]

        if not (sym[0].isalnum() or sym[0] == "_"):
            sym = "X" + sym

        for char, replacement in special_nt_map.items():
            if len(char) == 1:
                sym = sym.replace(char, replacement)

        return Nonterminal(sym)

    productions = []
    root_counts = defaultdict(int)

    for tree in train_trees:
        root_counts[tree.label()] += 1
        tree.chomsky_normal_form(horzMarkov=2)
        for prod in tree.productions():
            lhs = _format_nonterminal(prod.lhs())
            rhs = [_format_nonterminal(sym) if isinstance(sym, Nonterminal) else sym for sym in prod.rhs()]
            productions.append(Production(lhs, rhs))

    if len(root_counts) > 1:
        start = Nonterminal("ROOT")
        for lbl, cnt in root_counts.items():
            rhs_nt = _format_nonterminal(Nonterminal(lbl))
            for _ in range(cnt):
                productions.append(Production(start, [rhs_nt]))
    else:
        start = _format_nonterminal(Nonterminal(next(iter(root_counts))))

    return induce_pcfg(start, productions)


def evaluate(parser: ViterbiParser, dev_trees: List[Tree], *, theta: float, vocab: set):
    correct = 0
    total_time = 0.0
    total_gold_constituents = 0
    total_pred_constituents = 0
    total_correct_constituents = 0

    for gold in dev_trees:
        sent = gold.leaves()
        tokens = TreebankWordTokenizer().tokenize(" ".join(sent))
        print(f"Evaluating sentence: {' '.join(sent)}")

        if theta == 1.0 and any(w not in vocab for w in tokens):
            total_time += 0.0
            continue

        t0 = time.perf_counter()
        try:
            parsed = next(parser.parse(tokens))
            print(f"Parsed: {parsed}")
        except:
            print("Parsing failed")
            parsed = None
        total_time += time.perf_counter() - t0

        # Exact match
        if parsed and parsed == gold:
            correct += 1

        # F1 calculation
        if parsed:
            gold_constituents = get_constituents(gold)
            pred_constituents = get_constituents(parsed)

            correct_constituents = gold_constituents.intersection(pred_constituents)

            total_gold_constituents += len(gold_constituents)
            total_pred_constituents += len(pred_constituents)
            total_correct_constituents += len(correct_constituents)

    # Calculate metrics
    acc = correct / len(dev_trees)
    avg_time = total_time / len(dev_trees)

    precision = total_correct_constituents / total_pred_constituents if total_pred_constituents > 0 else 0
    recall = total_correct_constituents / total_gold_constituents if total_gold_constituents > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return acc, avg_time, precision, recall, f1


def main(cfg_path: str):
    config = load_config(cfg_path)
    train_trees, dev_trees, test_trees = get_treebank_splits()
    grammar = build_pcfg(train_trees)

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    lexical_vocab = {
        prod.rhs()[0] for prod in grammar.productions() if len(prod.rhs()) == 1 and isinstance(prod.rhs()[0], str)
    }

    results = []
    current_model = None
    llm = None
    provider = None

    nts = {str(p.lhs()) for p in grammar.productions()}

    for theta, model_name in itertools.product(config["data_score"], config["model"]):
        print(theta)

        if theta == 1.0:
            parser = ViterbiParser(grammar)
        else:
            # Check if model has changed
            if model_name != current_model:
                if llm is not None:
                    del llm
                    torch.cuda.empty_cache()
                # Create new LLM instance and provider
                llm = LocalLLM(model_name=model_name)
                provider = TokenLevelProbabilityProvider(llm, nts)
                current_model = model_name

            # Use existing provider when model hasn't changed
            parser = TokenLevelViterbiParser(grammar, provider, theta=theta)

        acc, avg_time, precision, recall, f1 = evaluate(parser, dev_trees, theta=theta, vocab=lexical_vocab)
        results.append(
            {
                "data_score": theta,
                "model": model_name,
                "accuracy": acc,
                "avg_inference_time": avg_time,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
        print(f"Results for theta={theta}, model={model_name}:")
        print(
            f"Accuracy: {acc:.4f}, Avg Inference Time: {avg_time:.4f}s, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
        )

        # Add a second plot for F1 scores
        plt.figure()
        for model in set(r["model"] for r in results):
            vals = sorted([r for r in results if r["model"] == model], key=lambda x: x["data_score"])
            thetas = [r["data_score"] for r in vals]
            f1_scores = [r["f1"] for r in vals]
            plt.plot(thetas, f1_scores, marker="o", label=model)

        plt.xlabel("theta")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.savefig("results/f1_scores.png")


if __name__ == "__main__":
    import argparse

    nltk.download("treebank")
    parser = argparse.ArgumentParser(description="Hyperparameter search for parsing")
    parser.add_argument("--config", help="Path to YAML config file", default="configs/sample.yaml")
    args = parser.parse_args()
    main(args.config)
