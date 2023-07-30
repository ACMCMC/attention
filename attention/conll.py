"""
Parse a phrase into a CoNLL-U format.
"""
# %%
import spacy
from spacy.tokens import Doc
from pandas import DataFrame

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("conll_formatter", last=True)

def parse_to_conllu(phrase: str) -> DataFrame:
    """
    Parse a phrase into a CoNLL-U format.

    :param phrase: The phrase to parse.
    :param nlp: The spaCy language model to use.
    :return: The CoNLL-U format string.
    """
    doc: Doc = nlp(phrase)
    conll = doc._.conll_pd
    # Index by column ID
    conll = conll.set_index('ID')
    # Decrement the ID by 1
    conll.index = conll.index - 1
    # Decrement the HEAD by 1
    conll['HEAD'] = conll['HEAD'] - 1
    return conll
# %%
