# %%
# Evaluate the model on the GLUE dataset
import logging
import os
from typing import Any, Dict, List

import datasets
import pandas as pd
import torch
import tqdm
from transformers import PreTrainedModel

from attention.conll import load_conllu_file, parse_to_conllu
from attention.max_attention_weights import (
    heads_matching_relation,
    max_attention_weights,
)
from attention.model_process import get_attention_matrix
from attention.variability import get_relative_variability

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


# Sometimes, there's very unfrequent relations, so if there's less than this number of relations present in the analyzed data, we don't output a diagram on that
MIN_WORDS_MATCHING_RELATION = 25
# In decoder models, relations like this:
# "I am human", where DEPENDANT=human and HEAD=am
# can be modeled.
# However, these models can't "attend to" the future.
# So it would be impossible to obserb the following attention pattern:
# "the green house", where DEPENDANT=green and HEAD=house
# This is because the head comes after the dependant.
# i.e. to obserb the relation, the head needs to come before
# If this parameter is enabled, we also consider a "hit" the fact that HEAD attends to DEPENDANT
# This is in line with current literature.
ACCEPT_BIDIRECTIONAL_RELATIONS = True


def eval_glue(model):
    conll_dataset = datasets.load_dataset("glue", "cola", split="test")
    get_matching_heads_sentence = generate_fn_get_matching_heads_sentence(model)
    heads_matching_sentence = conll_dataset.map(
        get_matching_heads_sentence, batched=False
    )

    # Plot the relations
    plot_relations(heads_matching_sentence, model=model, display=False)


def eval_ud(model, path_to_conll_dataset):
    conll_phrases = load_conllu_file(path_to_conll_dataset)
    # conll_dataset = datasets.Dataset.from_list(
    #    conll_phrases,
    # features=datasets.Features(
    #    {
    #        'id': datasets.Value('string'),
    #        "form": datasets.Value("string"),
    #        "lemma": datasets.Value("string"),
    #        "upos": datasets.Value("string"),
    #        "xpos": datasets.Value("string"),
    #        "feats": datasets.Value("string"),
    #        "head": datasets.Value("int32"),
    #        "deprel": datasets.Value("string"),
    #        "deps": datasets.Value("string"),
    #        #'misc': datasets.Value("string"),
    #    }
    # ),
    # )

    logger.info(f"About to process {len(conll_phrases)} examples...")

    conll_phrases = conll_phrases[:10]

    get_matching_heads_sentence = generate_fn_get_matching_heads_sentence(model)
    phrases_iterator = tqdm.tqdm(conll_phrases, unit="phrase")
    heads_matching_sentence = [get_matching_heads_sentence(e) for e in phrases_iterator]

    # Plot the relations
    plot_relations(heads_matching_sentence, model=model, display=False)

    uas = calculate_uas(heads_matching_sentence)
    logger.info(f"UAS: {uas}")


# %%
def generate_fn_get_matching_heads_sentence(model):
    """
    Returns a function that takes a sentence and returns the heads matching each relation.

    We need to generate a function because the Datasets.map() function only accepts functions with one argument and we need to pass the model to the function.
    """

    def get_matching_heads_sentence(example):
        # Does the example contain a sentence literal or a list of dicts representing a sentence?
        if isinstance(example, list):
            conll_pd = pd.DataFrame.from_records(example, index="ID")
        elif isinstance(example["sentence"], str):
            # If it is a sentence, parse it to a dataframe
            conll_pd = parse_to_conllu(example["sentence"])

        # Dependencies: a list of the form (dependent_word, head_word, relation)
        dependencies = [
            (
                index,
                conll_pd.loc[row["HEAD"] == conll_pd.index].index.item(),
                row["DEPREL"],
            )
            for index, row in conll_pd.iterrows()
            if row["HEAD"] > -1
        ]
        dependencies_head_and_dependant = [
            (dependent_word, head_word)
            for (dependent_word, head_word, _) in dependencies
        ]
        dependencies_reltype = [relation for (_, _, relation) in dependencies]

        # Take all the words in the sentence and get the heads matching the relation
        attention_matrix = get_attention_matrix(conll_pd=conll_pd, model=model)
        max_weights = max_attention_weights(attention_matrix)
        heads_matching_rel = heads_matching_relation(
            conll_pd,
            max_weights,
            accept_bidirectional_relations=ACCEPT_BIDIRECTIONAL_RELATIONS,
        )
        # The result is a list of tuples (layer, head, head_word, dependent_word)
        # Only keep the tuples that have a dependent_word matching the relation
        # For every tuple, get the dependent_word and check if it matches the relation. The dependent_word is in position 3, and it is a string, so we need to get the row with that word in the column FORM and check its DEPREL
        heads_matching_rel = [
            (*t, conll_pd[conll_pd["FORM"] == t[2]]["DEPREL"].values[0])
            for t in heads_matching_rel
        ]
        matching_heads_layer_and_head = [
            (layer, head) for (layer, head, _, _, dependency) in heads_matching_rel
        ]
        matching_heads_dependency = [
            dependency for (_, _, _, _, dependency) in heads_matching_rel
        ]

        # Get the variability of each head for each layer
        variability = get_relative_variability(attentions_matrix=attention_matrix[0])
        matching_heads_variability = [
            variability[layer, head]
            for (layer, head, _, _, dependency) in heads_matching_rel
        ]

        return {
            "matching_heads_layer_and_head": matching_heads_layer_and_head,
            "matching_heads_dependency": matching_heads_dependency,
            "matching_heads_variability": matching_heads_variability,
            "dependencies_head_and_dependant": dependencies_head_and_dependant,
            "dependencies_reltype": dependencies_reltype,
            "forms": conll_pd["FORM"],
            "max_attention_weights": max_weights,
        }

    return get_matching_heads_sentence


# %%
def plot_relations(heads_matching_sentence, model: PreTrainedModel, display=True):
    # Get all unique values for the relation
    # The relation is in the last position of the tuple 'dependencies'
    # The result is a list of strings
    relations = [
        dependency
        for example in heads_matching_sentence
        for dependency in example["dependencies_reltype"]
    ]
    unique_relations = set(relations)

    # Plot a 2D matrix with the heads matching each relation. The color of each cell is the number of heads matching that relation
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create a matrix with the number of heads matching each relation
    # The matrix is of size [num_layers, num_heads]
    # Use the counter to fill the matrix

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    for relation in unique_relations:
        matrix = torch.tensor(
            [[0 for _ in range(num_heads)] for _ in range(num_layers)]
        )
        # opacity matrix is a tensor of size [num_layers, num_heads] with all zeros
        opacity_matrix = torch.zeros((num_layers, num_heads))
        total_words_matching_relation = 0
        total_words = 0
        for example in heads_matching_sentence:
            # Join matching_heads_layer_and_head and matching_heads_dependency to get the tuple
            for (layer, head_position), dependency in zip(
                example["matching_heads_layer_and_head"],
                example["matching_heads_dependency"],
            ):
                if dependency == relation:
                    matrix[layer][head_position] += 1
            for (dependant_position, head_position), dependency in zip(
                example["dependencies_head_and_dependant"],
                example["dependencies_reltype"],
            ):
                if model.config.model_type == "bloom":
                    # For decoder-only models, the dependant has to be posterior to the head, otherwise the attention is not possible
                    if dependant_position < head_position and (
                        not ACCEPT_BIDIRECTIONAL_RELATIONS
                    ):
                        # We don't accept bidirectional relations, so it's impossible that we'll have an attention pattern DEPENDANT -> HEAD
                        logger.debug(
                            f'Skipping the dependency between {dependant_position} -> {head_position} ({example["forms"][dependant_position]} -> {example["forms"][head_position]})'
                        )
                        continue
                if dependency == relation:
                    total_words_matching_relation += 1
            for (layer, head_position), variability in zip(
                example["matching_heads_layer_and_head"],
                example["matching_heads_variability"],
            ):
                opacity_matrix[layer, head_position] += variability
            total_words += len(example["dependencies_reltype"])
        if total_words_matching_relation < MIN_WORDS_MATCHING_RELATION:
            continue
        # opacity_matrix /= total_words  # Get the average variability for each layer and head
        # Divide the opacity matrix by the maximum value to normalize it
        opacity_matrix /= opacity_matrix.max()

        # Plot the matrix. The upper limit of the colorbar is the number of words in the dataset
        # Title: relation
        # X axis: head
        # Y axis: layer
        plt.title(f"Heads matching relation {relation}")
        plt.xlabel("Head")
        plt.ylabel("Layer")
        sns.heatmap(
            matrix,
            cmap="YlGnBu",
            vmin=0,
            vmax=total_words_matching_relation,
            annot=True,
            alpha=1.0 - opacity_matrix,
            # Text size: 6
            annot_kws={"size": 6},
            # Do not use scientific notation
            fmt="g",
        )

        # Add the opacity matrix as a mask
        # sns.heatmap(
        #    opacity_matrix,
        #    # Color map alpha is the opacity
        #    cmap="Greys_r",
        #    vmin=0,
        #    vmax=1,
        #    annot=False,
        #    # mask=inverted_opacity_matrix < 0.5,
        #    alpha=1.0 - opacity_matrix,
        # )

        # If the directory does not exist, create it
        if not os.path.exists("figures"):
            os.makedirs("figures")

        # Store it as a PDF
        plt.savefig(f"figures/heads_matching_{relation}_{model.config.model_type}.pdf")
        if display:
            plt.show()
        # Reset the plot
        plt.clf()

        # Also store the matrix as a CSV
        matrix_df = pd.DataFrame(matrix)
        # Index column: layer
        matrix_df.index = [f"layer_{i}" for i in range(num_layers)]
        # Set the column names to the 'head_'+head number
        matrix_df.columns = [f"head_{i}" for i in range(num_heads)]
        matrix_df.to_csv(
            f"figures/heads_matching_{relation}_{model.config.model_type}.csv",
            index=True,
        )


def calculate_uas(heads_matching_sentence: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Calculate the Unlabeled Attachment Score (UAS) of the model on the dataset.

    The UAS is the number of correctly predicted heads divided by the total number of words in the dataset.

    The UAS is calculated as follows:
    - For each word in the dataset, check if the head predicted by the model matches the actual head.
    - If it matches, increment the number of correct words.
    - At the end, divide the number of correct words by the total number of words in the dataset.

    The input is a list of dictionaries, each containing the following keys:
    - 'matching_heads_layer_and_head': a list of tuples (layer, head) with the heads matching each relation
    - 'dependencies_head_and_dependant': a list of tuples (dependent_word, head_word) with the actual heads
    - 'dependencies_reltype': a list of strings with the relations
    - 'max_attention_weights': a tensor with the maximum attention weights for each row in the attention matrix

    We return the UAS for each dependency type, head and layer. The format is a dictionary with the dependency type as the key and a tensor of shape [num_layers, num_heads] as the UAS for that dependency type.
    """

    # First, get the total number of layers and heads. We can inspect the first example
    num_layers = heads_matching_sentence[0]["max_attention_weights"].shape[-3]
    num_heads = heads_matching_sentence[0]["max_attention_weights"].shape[-2]

    number_of_heads_matching_sentence_per_dependency = {} # This stores a matrix per dependency type, head and layer with the number of heads matching the relation

    total_tokens = 0

    # Now, for each dependency type, head and layer, calculate the UAS
    for example in heads_matching_sentence:
        # Here, we get the dependency type, head and layer for each word matching the relation
        # And we add 1 to the entry in the matrix for that dependency type, head and layer
        for (layer, head), dependency in zip(
            example["matching_heads_layer_and_head"], example["matching_heads_dependency"]
        ):
            if dependency not in number_of_heads_matching_sentence_per_dependency:
                number_of_heads_matching_sentence_per_dependency[dependency] = torch.zeros(
                    (num_layers, num_heads)
                )
            number_of_heads_matching_sentence_per_dependency[dependency][layer, head] += 1

        total_tokens += example["max_attention_weights"].shape[-1]

    # Now, we have the number of heads matching each relation for each layer and head
    # We can calculate the UAS for each relation
    uas_per_dependency = {}
    for dependency, matrix in number_of_heads_matching_sentence_per_dependency.items():
        uas_per_dependency[dependency] = matrix / total_tokens

    return uas_per_dependency


# %%
