"""
Parse a phrase into a CoNLL-U format.
"""
# %%
import spacy
from spacy.tokens import Doc
from pandas import DataFrame
import pandas as pd
from conllu import parse

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
    conll = conll.set_index("ID")
    # Decrement the ID by 1
    conll.index = conll.index - 1
    # Decrement the HEAD by 1
    conll["HEAD"] = conll["HEAD"] - 1
    # Sometimes, there's two phrases in the same sentence, so the ID is not unique. In that case, we need to reset the index in the second phrase (i.e. continue the numbering)
    # This happens if the last word of the dataframe does not have ID equal to the number of rows in the dataframe
    if conll.iloc[-1].name != conll.shape[0] - 1:
        # Now, for every row where the ID is not equal to the row position, we increment the HEAD by the difference
        for i in range(conll.shape[0]):
            # Skip the root word. The root word has HEAD = -1
            if conll.iloc[i, conll.columns.get_loc("HEAD")] == -1:
                continue
            if conll.iloc[i].name != i:
                difference = i - conll.iloc[i].name
                conll.iloc[i, conll.columns.get_loc("HEAD")] += difference
        # Reset the index so that the ID is unique and starts at 0
        conll = conll.reset_index(drop=True)
        # Call the index ID again
    return conll


# %%
def load_conllu_file(filename):
    # Use the library to load the .conllu file into a Dataframe
    with open(filename, "r") as f:
        data = f.read()
    sentences = parse(data)

    # The result is a list of sentences. Each sentence is a list of words. Each word is a dictionary with the keys FORM, HEAD, DEPREL, etc.
    # We need to convert it to a dataframe
    # First, get the list of words for each sentence
    sentences_words = [[word for word in sentence] for sentence in sentences]
    # Now, convert each word to a dictionary
    sentences_dicts = [
        {key: word[key] for key in word.keys()} for sentence in sentences_words for word in sentence
    ]
    # Now, convert the list of dictionaries to a dataframe
    return sentences_dicts

# %%
