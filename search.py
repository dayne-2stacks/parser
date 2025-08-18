import nltk
import json
import yaml
from grammar.learn_grammar import learn_grammar
import itertools
from viterbi import TokenLevelViterbiParser, ViterbiParser
from provider import TokenLevelProbabilityProvider
from local_llm import LocalLLM
import torch

def get_treebank_splits(dev_ratio=0.1, test_ratio=0.1):
    """Get train, dev and test splits of the NLTK Treebank corpus."""
    # Get all the trees from the NLTK Treebank corpus
    parsed_trees = list(nltk.corpus.treebank.parsed_sents())
    sentences = list(nltk.corpus.treebank.sents())
    
    split_dev = int(len(parsed_trees) * (1 - dev_ratio - test_ratio))
    train_parsed = parsed_trees[:split_dev]
    dev_parsed = parsed_trees[split_dev : -int(len(parsed_trees) * test_ratio)] if test_ratio else parsed_trees[split_dev:]
    test_parsed = parsed_trees[-int(len(parsed_trees) * test_ratio) :] if test_ratio else []

    train_sents = sentences[:split_dev]
    dev_sents = sentences[split_dev : -int(len(sentences) * test_ratio)] if test_ratio else sentences[split_dev:]
    test_sents = sentences[-int(len(sentences) * test_ratio) :] if test_ratio else []

    return (train_parsed, dev_parsed, test_parsed), (train_sents, dev_sents, test_sents)


def main(path: str):
    # Load configuration
    with open(path, 'r') as f:
        if path.endswith(".json"):
            config = json.load(f)
        else:
            config = yaml.safe_load(f)
            
            
    

    # Get train, dev, and test splits
    (train_parsed, dev_parsed, test_parsed), (train_sents, dev_sents, test_sents) = get_treebank_splits(
        dev_ratio=config.get("dev_ratio", 0.1),
        test_ratio=config.get("test_ratio", 0.1)
    )
    
    # Build PCFG from training trees
    grammar_def = learn_grammar(train_parsed)
    grammar = nltk.PCFG.fromstring(grammar_def)
    
    nonterminals = {str(prod.lhs()) for prod in grammar.productions()}
    
    results = []
    current_model = None
    llm = None
    provider = None
    
    for theta, model_name in itertools.product(config["thetas"], config["model"]):
        
        if theta == 1.0:
            parser = ViterbiParser(grammar, trace=2)
        else:
            if current_model != model_name:
                if llm is not None:
                    del llm
                    torch.cuda.empty_cache()
                llm = LocalLLM(model_name=model_name)
                provider = TokenLevelProbabilityProvider(llm, nonterminals, cache_size=2048)
                current_model = model_name
            
            parser = TokenLevelViterbiParser(
                grammar=grammar,
                token_provider=provider,
                theta=theta,
            )
        
        
        test_sentences = [' '.join(sentence) for sentence in test_sents]
        
        for sentence in test_sentences:
            print(f"\nParsing sentence: '{sentence}'")
            tokens = nltk.TreebankWordTokenizer().tokenize(sentence)
            parse_results = list(parser.parse(tokens))
            
            if parse_results:
                for i, tree in enumerate(parse_results):
                    print(f"Parse {i+1} for '{sentence}':")
                    print(tree)
            else:
                print(f"No parse found for '{sentence}'")
    
    



if __name__ == "__main__":
    import argparse
    nltk.download("treebank")
    parser = argparse.ArgumentParser(description="Hyperparameter search for parsing")
    parser.add_argument("--config", help="Path to YAML config file", default="configs/sample.yaml")
    args = parser.parse_args()
    main(args.config)