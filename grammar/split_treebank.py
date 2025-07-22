import os
import random
import nltk
from nltk.corpus import treebank
from nltk.tree import Tree


def ensure_treebank_loaded():
    """Ensure the Penn Treebank corpus is available."""
    try:
        treebank.ensure_loaded()
    except LookupError:
        nltk.download('treebank')
        treebank.ensure_loaded()


def tree_to_line(t: Tree) -> str:
    """Wrap a tree with a ROOT node and return single-line string."""
    return Tree('ROOT', [t]).pformat(margin=1000000)


def main():
    ensure_treebank_loaded()
    parsed = list(treebank.parsed_sents())
    rng = random.Random(42)
    rng.shuffle(parsed)

    n = len(parsed)
    train_end = int(0.8 * n)
    dev_end = train_end + int(0.1 * n)

    splits = {
        'train.clean': parsed[:train_end],
        'dev.clean': parsed[train_end:dev_end],
        'test.clean': parsed[dev_end:],
    }

    out_dir = os.path.join(os.path.dirname(__file__), 'splits')
    os.makedirs(out_dir, exist_ok=True)

    for name, trees in splits.items():
        path = os.path.join(out_dir, name)
        with open(path, 'w') as f:
            for t in trees:
                f.write(tree_to_line(t) + '\n')


if __name__ == '__main__':
    main()
