import nltk
import json
import yaml
import sys
import os
from grammar.learn_grammar import learn_grammar
import itertools
from viterbi import TokenLevelViterbiParser, ViterbiParser
from provider import TokenLevelProbabilityProvider
from local_llm import LocalLLM
import torch

# Add benepar src directory to Python path so we can import evaluate
sys.path.append(os.path.join(os.path.dirname(__file__), "benepar/src"))
import evaluate


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


def evaluate_parser(parser, test_trees, test_sents, evalb_dir="EVALB/"):
    """
    Evaluate parser performance using EVALB.
    
    Args:
        parser: The parser to evaluate
        test_trees: Gold standard parse trees
        test_sents: Test sentences corresponding to the trees
        evalb_dir: Directory containing EVALB binaries
        
    Returns:
        FScore object with evaluation metrics
    """
    predicted_trees = []
    skipped_count = 0
    
    # Process each test sentence and get predictions
    for i, sentence in enumerate(test_sents[:1]):
        print(f"Parsing test sentence {i+1}/{len(test_sents)}")
        tokens = sentence if isinstance(sentence, list) else nltk.TreebankWordTokenizer().tokenize(sentence)
        
        try:
            # Get the best parse for this sentence
            parse_results = list(parser.parse(tokens))
            if parse_results:
                predicted_trees.append(parse_results[0])  # Take the highest probability parse
            else:
                # If no parse is found, create a flat tree as a fallback
                flat_tree = nltk.Tree('S', [(word, 'UNK') for word in tokens])
                predicted_trees.append(flat_tree)
                print(f"No parse found for sentence {i+1}, using flat tree")
                skipped_count += 1
        except Exception as e:
            print(f"Error parsing sentence {i+1}: {e}")
            # Create a flat tree as a fallback
            flat_tree = nltk.Tree('S', [(word, 'UNK') for word in tokens])
            predicted_trees.append(flat_tree)
            skipped_count += 1
    
    if skipped_count > 0:
        print(f"Warning: {skipped_count} sentences ({skipped_count/len(test_sents)*100:.1f}%) could not be parsed properly")
    
    # Make sure we have predictions for all test sentences
    assert len(predicted_trees) == len(test_trees[:1]), "Number of predictions must match number of test trees"
    
    # Use the EVALB evaluation
    return evaluate.evalb(evalb_dir, test_trees[:1], predicted_trees)


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
    
    # Try to find EVALB in different locations
    evalb_dir = config.get("evalb_dir", "EVALB/")
    possible_evalb_locations = [
        evalb_dir,
        os.path.join(os.path.dirname(__file__), evalb_dir),
        os.path.join(os.path.dirname(__file__), "benepar", "EVALB"),
        os.path.join(os.path.dirname(__file__), "benepar", "src", "EVALB")
    ]
    
    evalb_found = False
    for location in possible_evalb_locations:
        if os.path.exists(location):
            evalb_dir = location
            evalb_found = True
            print(f"Found EVALB directory at: {evalb_dir}")
            break
    
    if not evalb_found:
        print("EVALB not found. Here's how to get it:")
        print("1. Download EVALB from https://nlp.cs.nyu.edu/evalb/")
        print("2. Extract it to your project directory")
        print("3. Run 'make' in the EVALB directory to compile it")
        print("4. Specify the correct path in your config file with the 'evalb_dir' parameter")
        print("\nSearched in these locations:")
        for loc in possible_evalb_locations:
            print(f"- {loc}")
        raise FileNotFoundError(f"EVALB directory not found. Please install EVALB or provide the correct path.")
    
    for theta, model_name in itertools.product(config["thetas"], config["model"]):
        print(f"\n\n--- Evaluating with theta={theta}, model={model_name} ---")
        
        # Initialize the appropriate parser
        if theta == 1.0:
            parser = ViterbiParser(grammar, trace=0)  # Reduced trace level for evaluation
        else:
            if current_model != model_name:
                # Free memory from previous model if needed
                if llm is not None:
                    del llm
                    torch.cuda.empty_cache()
                
                print(f"Loading model: {model_name}")
                llm = LocalLLM(model_name=model_name)
                provider = TokenLevelProbabilityProvider(llm, nonterminals, cache_size=2048)
                current_model = model_name
            
            parser = TokenLevelViterbiParser(
                grammar=grammar,
                token_provider=provider,
                theta=theta,
            )
        
        # Evaluate parser on test set
        print(f"Evaluating parser on {len(test_parsed)} test sentences...")
        fscore = evaluate_parser(parser, test_parsed, test_sents, evalb_dir)
        
        # Record results
        result = {
            "theta": theta,
            "model": model_name,
            "fscore": fscore.fscore,
            "precision": fscore.precision,
            "recall": fscore.recall,
            "complete_match": fscore.complete_match
        }
        results.append(result)
        
        print(f"Evaluation results for theta={theta}, model={model_name}:")
        print(f"  F-Score: {fscore.fscore:.2f}")
        print(f"  Precision: {fscore.precision:.2f}")
        print(f"  Recall: {fscore.recall:.2f}")
        print(f"  Complete Match: {fscore.complete_match:.2f}")
    
    # Save results to file
    results_path = config.get("results_path", "parsing_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAll evaluations complete. Results saved to {results_path}")


if __name__ == "__main__":
    import argparse
    nltk.download("treebank")
    parser = argparse.ArgumentParser(description="Hyperparameter search for parsing")
    parser.add_argument("--config", help="Path to YAML config file", default="configs/sample.yaml")
    args = parser.parse_args()
    main(args.config)