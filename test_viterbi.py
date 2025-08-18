import logging
from nltk import PCFG, Nonterminal
from nltk.grammar import Production
from provider import TokenLevelProbabilityProvider
from local_llm import LocalLLM
from viterbi import TokenLevelViterbiParser
from nltk.tokenize import TreebankWordTokenizer
import torch
from gpu_logging_utils import log_cuda_memory_pytorch, flush_cuda_cache
# Add import for Treebank corpus
from nltk.corpus import treebank

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Define a small grammar for testing
    # grammar_str = """
    # S -> NP VP [1.0]
    # NP -> Det N [0.6] | Pronoun [0.4]
    # VP -> V NP [0.7] | V [0.3]
    # Det -> 'the' [0.8] | 'a' [0.2]
    # N -> 'dog' [0.4] | 'cat' [0.6]
    # V -> 'chased' [0.6] | 'saw' [0.4]
    # Pronoun -> 'he' [0.5] | 'she' [0.5]
    # """
    # grammar = PCFG.fromstring(grammar_str)
    
    log_cuda_memory_pytorch("before_grammar_load")
    with open("grammar/induced_grammar.cfg", "r") as f:
        grammar_str = f.read()
    grammar = PCFG.fromstring(grammar_str)
    
    # Extract all nonterminals from the grammar
    nonterminals = {str(prod.lhs()) for prod in grammar.productions()}
    log_cuda_memory_pytorch("after_grammar_load")
    
    logger.info("Initializing LLM")
    log_cuda_memory_pytorch("before_llm_init")
    llm = LocalLLM(model_name="llama3_1-70b")
    log_cuda_memory_pytorch("after_llm_init")
    
    logger.info(f"Creating token level probability provider with nonterminals: {nonterminals}")
    log_cuda_memory_pytorch("before_provider_init")
    provider = TokenLevelProbabilityProvider(llm, nonterminals, cache_size=2048)
    log_cuda_memory_pytorch("after_provider_init")

    logger.info("Creating token-level viterbi parser")
    log_cuda_memory_pytorch("before_parser_init")
    parser = TokenLevelViterbiParser(
        grammar=grammar,
        token_provider=provider,
        theta=0.9, 
        # trace=2,  # Detailed tracing
    )
    log_cuda_memory_pytorch("after_parser_init")

    # Original test sentences (commented out)
    original_test_sentences = [
        # "Pierre bans taxes in America, but left home because he was scared."
        # "Pierre Vinken , 61 years old, will join the board as a nonexecutive director Nov. 29."
        "Pierre Vinken is 61 years old."
    ]
    
    # Load sentences from the Treebank corpus
    logger.info("Loading sentences from Treebank corpus")
    treebank_sentences = treebank.sents()
    # Take a few sentences for testing (first 3)
    test_treebank_sentences = treebank_sentences[:3]
    
    # Convert token lists to strings for display and testing
    test_sentences = [' '.join(sentence) for sentence in test_treebank_sentences]
    
    for sentence in test_sentences:
        logger.info(f"\nParsing sentence: '{sentence}'")
        tokens = TreebankWordTokenizer().tokenize(sentence)
        logger.info(f"Tokens: {tokens}")

        log_cuda_memory_pytorch(f"before_parsing_{sentence[:20]}")
        parse_results = list(parser.parse(tokens))
        log_cuda_memory_pytorch(f"after_parsing_{sentence[:20]}")
        
        # if parse_results:
        #     for i, tree in enumerate(parse_results):
        #         logger.info(f"Parse {i+1}:")
        #         logger.info(f"Tree: {tree}")
        #         logger.info(f"Probability: {tree.prob()}")
        # else:
        #     logger.info("No parse found.")
            
        tree = parse_results[0]
        print(tree)
        
    # Clean up memory at the end
    flush_cuda_cache()
    log_cuda_memory_pytorch("after_cleanup")

if __name__ == "__main__":
    main()