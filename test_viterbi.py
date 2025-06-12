import logging
from nltk import PCFG, Nonterminal
from nltk.grammar import Production
from provider import TokenLevelProbabilityProvider
from local_llm import LocalLLM
from viterbi import TokenLevelViterbiParser
from nltk.tokenize import TreebankWordTokenizer
import torch

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
    
    with open("grammar/induced_grammar.cfg", "r") as f:
        grammar_str = f.read()
    grammar = PCFG.fromstring(grammar_str)
    
    # Extract all nonterminals from the grammar
    nonterminals = {str(prod.lhs()) for prod in grammar.productions()}
    
    logger.info("Initializing LLM")
    llm = LocalLLM(model_name="llama3_1-70b")
    
    logger.info(f"Creating token level probability provider with nonterminals: {nonterminals}")
    provider = TokenLevelProbabilityProvider(llm, nonterminals, cache_size=2048)

    logger.info("Creating token-level viterbi parser")
    parser = TokenLevelViterbiParser(
        grammar=grammar,
        token_provider=provider,
        theta=0.9, 
        # trace=2,  # Detailed tracing
    )

    test_sentences = [
        # "Pierre bans taxes in America, but left home because he was scared."
        # "Pierre Vinken , 61 years old, will join the board as a nonexecutive director Nov. 29."
        "Pierre Vinken is 61 years old."
        
    ]
    
    for sentence in test_sentences:
        logger.info(f"\nParsing sentence: '{sentence}'")
        tokens = TreebankWordTokenizer().tokenize(sentence)
        logger.info(f"Tokens: {tokens}")

        
        parse_results = list(parser.parse(tokens))
        
        # if parse_results:
        #     for i, tree in enumerate(parse_results):
        #         logger.info(f"Parse {i+1}:")
        #         logger.info(f"Tree: {tree}")
        #         logger.info(f"Probability: {tree.prob()}")
        # else:
        #     logger.info("No parse found.")
            
        tree = parse_results[0]

        print(tree)

if __name__ == "__main__":
    main()