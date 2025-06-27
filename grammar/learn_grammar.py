import nltk
from nltk.corpus import treebank
import json


try:
    treebank.ensure_loaded() 
except LookupError:
    nltk.download('treebank')
   
    treebank.ensure_loaded()

# Collect all productions from the parsed sentences
productions = []
for tree in treebank.parsed_sents():
    productions += tree.productions()

# Count frequencies of each production and each LHS nonterminal
from collections import defaultdict
prod_counts = defaultdict(int)
lhs_counts = defaultdict(int)
for prod in productions:
    prod_counts[prod] += 1
    lhs_counts[prod.lhs()] += 1

with open("changes.json", "r") as f:
    special_nt_map = json.load(f)

def format_nonterminal(sym):
    """Convert an NLTK Nonterminal to a safe string for grammar output."""
    sym_str = sym.symbol()  # get the symbol name (string) from Nonterminal
    
    # If symbol is of form -X- (leading and trailing hyphen, e.g., -NONE-), remove the hyphens
    if sym_str.startswith("-") and sym_str.endswith("-") and len(sym_str) > 1:
        sym_str = sym_str[1:-1]
        
    # Replace entirely if in special map (punctuation and others)
    if sym_str in special_nt_map:
        return special_nt_map[sym_str]
        
    # If the first character is not alphanumeric or underscore, prepend 'X' to make it valid
    if not (sym_str[0].isalnum() or sym_str[0] == "_"):
        sym_str = "X" + sym_str

    # Apply character replacements from the JSON file
    for char, replacement in special_nt_map.items():
        if len(char) == 1:  # Only apply for single-character replacements
            sym_str = sym_str.replace(char, replacement)
    
    return sym_str

def format_rhs_item(item):
    """Format a right-hand side item (Nonterminal or terminal) for grammar output."""
    if isinstance(item, nltk.grammar.Nonterminal):
        # Nonterminal: format it recursively
        return format_nonterminal(item)
    else:
        token = str(item)
        if "'" in token:
            token = '"' + token.replace('"', '\\"') + '"'
        else:
            token = "'" + token + "'"
        return token


grammar_lines = []
root_counts = defaultdict(int)
for tree in treebank.parsed_sents():
    root_label = tree.label() 
    root_counts[root_label] += 1

if len(root_counts) > 1:
    # Create a special ROOT symbol as the start
    total_roots = sum(root_counts.values())
    root_alternatives = []
    for root_sym, cnt in root_counts.items():
        # Format the root symbol as a Nonterminal (needs to be in grammar format too)
        root_nt = nltk.grammar.Nonterminal(root_sym)
        root_formatted = format_nonterminal(root_nt)
        prob = cnt / total_roots
        prob_str = f"{prob:.7f}"
        # Strip unnecessary trailing zeros and dot
        prob_str = prob_str.rstrip('0').rstrip('.') if '.' in prob_str else prob_str
        root_alternatives.append(f"{root_formatted} [{prob_str}]")
    # Construct the ROOT production line
    grammar_lines.append(f"ROOT -> " + " | ".join(root_alternatives))
    start_symbol = "ROOT"
    


rules_by_lhs = defaultdict(list)
for prod, count in prod_counts.items():
    lhs = prod.lhs()
    rhs = prod.rhs()
    lhs_str = format_nonterminal(lhs)
    rhs_parts = [format_rhs_item(item) for item in rhs]
    rhs_str = " ".join(rhs_parts)
    prob = count / lhs_counts[lhs]
    prob_str = f"{prob:.7f}"
    prob_str = prob_str.rstrip('0').rstrip('.') if '.' in prob_str else prob_str
    rules_by_lhs[lhs_str].append(f"{rhs_str} [{prob_str}]")


for lhs_str, expansions in rules_by_lhs.items():

    if lhs_str == start_symbol and lhs_str == "ROOT":
        continue

    filtered = []
    for exp in expansions:
        rhs_symbol = exp.split(" [")[0].strip()
        if not (len(rhs_symbol.split()) == 1 and rhs_symbol == lhs_str):
            filtered.append(exp)

    if not filtered:
        continue

    if len(filtered) == 1:
        grammar_lines.append(f"{lhs_str} -> {filtered[0]}")
    else:
        grammar_lines.append(f"{lhs_str} -> " + " | ".join(filtered))



grammar_text = "\n".join(grammar_lines)

with open("induced_grammar.cfg", "w") as outfile:
    outfile.write(grammar_text)
