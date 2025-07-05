import itertools
import json
import time
from typing import Dict, Any, List

import yaml
import nltk
from collections import defaultdict
from nltk import Nonterminal, induce_pcfg, ProbabilisticProduction
from nltk.tree import Tree
from nltk.parse import ViterbiParser
from viterbi import TokenLevelViterbiParser
from provider import TokenLevelProbabilityProvider
from local_llm import LocalLLM
import matplotlib.pyplot as plt
import os


def load_config(path: str) -> Dict[str, List[Any]]:
    """Load YAML or JSON configuration describing hyperparameters."""
    with open(path, 'r') as f:
        if path.endswith('.json'):
            data = json.load(f)
        else:
            data = yaml.safe_load(f)
    return data


def get_treebank_splits(dev_ratio: float = 0.1):
    """Return training and development splits of the Penn Treebank."""
    trees = list(nltk.corpus.treebank.parsed_sents())
    split = int(len(trees) * (1 - dev_ratio))
    train_trees = trees[:split]
    dev_trees = trees[split:]
    return train_trees, dev_trees


def build_pcfg(train_trees: List[Tree]):
    """Induce a PCFG using logic from ``grammar/learn_grammar.py``."""
    productions = []
    root_counts = defaultdict(int)

    for tree in train_trees:
        root_counts[tree.label()] += 1
        tree.chomsky_normal_form(horzMarkov=2)
        productions.extend(tree.productions())

    if len(root_counts) > 1:
        start = Nonterminal("ROOT")
        total = sum(root_counts.values())
        for lbl, cnt in root_counts.items():
            prob = cnt / total
            prod = ProbabilisticProduction(start, [Nonterminal(lbl)], prob=prob)
            productions.append(prod)
    else:
        start = Nonterminal(next(iter(root_counts)))

    return induce_pcfg(start, productions)


def evaluate(parser: ViterbiParser, dev_trees: List[Tree], *, theta: float, vocab: set):
    """Evaluate parser accuracy and inference time on development trees."""
    correct = 0
    total_time = 0.0
    for gold in dev_trees:
        sent = gold.leaves()

        # If we rely purely on the grammar and an unknown word is present,
        # treat as an inaccurate parse without attempting to parse.
        if theta == 1.0 and any(w not in vocab for w in sent):
            total_time += 0.0
            continue

        t0 = time.perf_counter()
        try:
            parsed = next(parser.parse(sent))
        except (StopIteration, ValueError):
            parsed = None
        total_time += time.perf_counter() - t0
        if parsed and parsed == gold:
            correct += 1
    acc = correct / len(dev_trees)
    avg_time = total_time / len(dev_trees)
    return acc, avg_time


def main(cfg_path: str):
    config = load_config(cfg_path)
    train_trees, dev_trees = get_treebank_splits()
    grammar = build_pcfg(train_trees)

    lexical_vocab = {
        prod.rhs()[0]
        for prod in grammar.productions()
        if len(prod.rhs()) == 1 and isinstance(prod.rhs()[0], str)
    }

    results = []
    for theta, model_name in itertools.product(config['data_score'], config['model']):
        if theta == 1.0:
            parser = ViterbiParser(grammar)
        else:
            llm = LocalLLM(model_name=model_name)
            nts = {str(p.lhs()) for p in grammar.productions()}
            provider = TokenLevelProbabilityProvider(llm, nts)
            parser = TokenLevelViterbiParser(grammar, provider, theta=theta)

        acc, avg_time = evaluate(parser, dev_trees, theta=theta, vocab=lexical_vocab)
        results.append({'data_score': theta, 'model': model_name,
                        'accuracy': acc, 'avg_inference_time': avg_time})

    os.makedirs('results', exist_ok=True)
    with open('results/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    for model in set(r['model'] for r in results):
        vals = sorted(
            [r for r in results if r['model'] == model], key=lambda x: x['data_score']
        )
        thetas = [r['data_score'] for r in vals]
        accs = [r['accuracy'] for r in vals]
        plt.plot(thetas, accs, marker='o', label=model)

    plt.xlabel('theta')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig('results/accuracy.png')
    plt.close()


if __name__ == '__main__':
    import argparse
    nltk.download('treebank')
    parser = argparse.ArgumentParser(description='Hyperparameter search for parsing')
    parser.add_argument('--config', help='Path to YAML/JSON config file', default='configs/sample.yaml')
    args = parser.parse_args()
    main(args.config)
