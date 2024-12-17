# %%
"""
Parse a phrase into a CoNLL-U format.
"""
import logging
import os
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import spacy
from conllu import parse
from pandas import DataFrame
from spacy.tokens import Doc

logger = logging.getLogger(__name__)
# %%

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
        [{key: word[key] for key in word.keys()} for word in sentence]
        for sentence in sentences_words
    ]
    sentences_dicts = [
        [{k.upper(): v for k, v in e.items() if k not in ["misc"]} for e in p]
        for p in sentences_dicts
    ]

    # Some compound words are repeated in the dataset. For example, "respondents'" is expressed as 15-16 "respondents'"; 15 "respondents"; 16 "'". So, we want to ignore words indexed with a hyphen.
    # Also, if there is an implicit reference to word from another clause, such as here:
    # Mof-Ávvi is a peninsula bordered by [...], and ******[BORDERED]****** to the east by [...]
    # Then that word is repeated with index 26.1
    # Let's remove them before proceeding
    sentences_dicts = [[e for e in p if type(e["ID"]) == int] for p in sentences_dicts]

    # Shift the indices by substracting 1 so that they are 0-indexed
    # Some position information can be strange. For example, there are words that can have index '26.1' (i.e. the indexing is not only 1-indexed, but also not necessarily consecutive)
    sentences_dicts = [
        [
            {
                **e,
                "OLD_ID": e[
                    "ID"
                ],  # Now that we removed the words with a hyphen or a dot, we know that all the IDs are pure numbers
                "ID": e_index,
                # In exceptional cases, the DEPS column can be empty (the value is None). In that case, we need to assign an empty list to DEPS (instead of None)
                "DEPS": e["DEPS"] if isinstance(e["DEPS"], Iterable) else [],
            }
            for (e_index, e) in enumerate(p)
        ]
        for p in sentences_dicts
    ]

    # Now, parse the "HEAD" and "DEPS" columns to integers by resolving the references to the word ID using OLD_ID
    def get_id_from_old_id(old_id, sentence):
        try:
            old_id = next((word["ID"] for word in sentence if word["OLD_ID"] == old_id))
            return old_id
        except Exception as e:
            if old_id == 0:
                # The root word has HEAD = 0. This means that there is no head word.
                return -1
            logger.error(
                f"Could not find the word with OLD_ID={old_id} in the sentence {sentence}"
            )
            return None

    sentences_dicts = [
        [
            {
                **e,
                # Find the word with OLD_ID equal to HEAD and assign its ID to HEAD
                "HEAD": get_id_from_old_id(e["HEAD"], p),
                # DEPS usually contains a list of tuples (function, position)
                # However, in some cases, some of the tuples in DEPS could have a position of type (position_1, '.', position_2) because of the weird indexing described above
                # In that case, we need to join the elements of the tuple into a string and then use that string to find the word with OLD_ID equal to that string
                "DEPS": [
                    (
                        (fn, get_id_from_old_id(pos, p))
                        if type(pos) == int
                        else (
                            fn,
                            get_id_from_old_id("".join([str(elem) for elem in pos]), p),
                        )
                    )
                    for (fn, pos) in e["DEPS"]
                ],
            }
            for e in p
        ]
        for p in sentences_dicts
    ]
    return sentences_dicts


def get_all_possible_conll_phrases(path_to_conll_dataset: Path):
    # We get an input path which looks like this: '.../eu_bdt-ud-train.conllu'
    # We want to also try to load the corresponding test and dev datasets, and concatenate them to the training dataset

    all_possible_conll_phrases = load_conllu_file(
        path_to_conll_dataset
    )  # This is the original dataset
    # Now, does a test dataset exist?
    test_dataset_path = path_to_conll_dataset.parent / (
        path_to_conll_dataset.name.replace("train", "test")
    )
    if os.path.exists(test_dataset_path):
        all_possible_conll_phrases += load_conllu_file(test_dataset_path)
    else:
        logger.warning(f"Test dataset not found at {test_dataset_path}")

    # Now, does a dev dataset exist?
    dev_dataset_path = path_to_conll_dataset.parent / (
        path_to_conll_dataset.name.replace("train", "dev")
    )
    if os.path.exists(dev_dataset_path):
        all_possible_conll_phrases += load_conllu_file(dev_dataset_path)
    else:
        logger.warning(f"Dev dataset not found at {dev_dataset_path}")

    return all_possible_conll_phrases


# %%
