# %%
# Evaluate the model on the GLUE dataset
import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List

import datasets
import mlflow
import pandas as pd
import torch
import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from attention.conll import load_conllu_file, parse_to_conllu
from attention.max_attention_weights import (
    heads_matching_relation,
    max_attention_weights,
)
from attention.model_process import UnprocessableSentenceException, get_attention_matrix
from attention.variability import get_relative_variability

logging.basicConfig(level=logging.INFO)


def eval_glue(model, accept_bidirectional_relations):
    conll_dataset = datasets.load_dataset("glue", "cola", split="test")
    get_matching_heads_sentence = generate_fn_get_matching_heads_sentence(
        model, accept_bidirectional_relations=accept_bidirectional_relations
    )
    heads_matching_sentence = conll_dataset.map(
        get_matching_heads_sentence, batched=False
    )

    # Plot the relations
    plot_relations(
        heads_matching_sentence,
        model=model,
        display=False,
        accept_bidirectional_relations=accept_bidirectional_relations,
    )


def perform_group_relations_by_family(
    conll_phrases: List[List[Dict[str, Any]]]
) -> List[List[Dict[str, Any]]]:
    """
    Group the relations by family. For example, 'nsubj', 'nsubjpass' and 'csubj' would be grouped together.
    """

    """
    Classes de funções/dependências sintácticas
        nominal verb arguments: obj, nsub, obl, iobj, (punct?)
        noun modifiers: nmod, amod, appos, acl
        noun specifiers: det, nummod, case
        verb auxiliars: aux, cop, mark, advmod
        clausal verb arguments: ccomp, xcomp, csubj
        compounds: flat, fixed, compound
        conjunctions: conj, cc
    É possível fazer classes mais genéricas juntando as seguintes:
        Verbal complements: 1+5
        Auxiliars+Specifiers: 3+4
    """

    # Define the relation families
    relation_families = {
        "nominal_verb_arguments": ["obj", "nsubj", "obl", "iobj", "punct"],
        "noun_modifiers": ["nmod", "amod", "appos", "acl"],
        "noun_specifiers": ["det", "nummod", "case"],
        "verb_auxiliars": ["aux", "cop", "mark", "advmod"],
        "clausal_verb_arguments": ["ccomp", "xcomp", "csubj"],
        "compounds": ["flat", "fixed", "compound"],
        "conjunctions": ["conj", "cc", "advcl"],
    }

    # First, map everything in the form XX:YY to XX
    for conll in conll_phrases:
        for word in conll:
            word["DEPREL"] = word["DEPREL"].split(":")[0]

    # Now, group the relations by family
    for conll in conll_phrases:
        for word in conll:
            for family, relations in relation_families.items():
                if word["DEPREL"] in relations:
                    word["DEPREL"] = family

    return conll_phrases


def eval_ud(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    conll_phrases: List[List[Dict[str, Any]]],
    output_dir=Path(__file__).parent.parent / "results",
    trim_dataset_size=None,
    random_seed=42,
    **kwargs,
):
    # Check that the following arguments are defined: 'accept_bidirectional_relations', 'min_words_matching_relation'
    if "accept_bidirectional_relations" not in kwargs:
        raise ValueError(
            "The argument 'accept_bidirectional_relations' must be defined"
        )
    if "min_words_matching_relation" not in kwargs:
        raise ValueError("The argument 'min_words_matching_relation' must be defined")
    if "group_relations_by_family" not in kwargs:
        raise ValueError("The argument 'group_relations_by_family' must be defined")
    if "remove_self_attention" not in kwargs:
        raise ValueError("The argument 'remove_self_attention' must be defined")

    # Set the random seed
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    mlflow.log_param("random_seed", random_seed)

    # Recreate the output dir - if it exists, delete it
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if kwargs["group_relations_by_family"] == True:
        logging.info(
            f"Grouping relations by family... (group_relations_by_family={kwargs['group_relations_by_family']})"
        )
        conll_phrases = perform_group_relations_by_family(conll_phrases)
    elif kwargs["group_relations_by_family"] == False:
        logging.info(
            f"Not grouping relations by family... (group_relations_by_family={kwargs['group_relations_by_family']})"
        )
    else:
        raise ValueError("The argument 'group_relations_by_family' must be defined")

    if kwargs["accept_bidirectional_relations"] == True:
        logging.info(
            f"Accepting bidirectional relations... (accept_bidirectional_relations={kwargs['accept_bidirectional_relations']})"
        )
    elif kwargs["accept_bidirectional_relations"] == False:
        logging.info(
            f"Rejecting bidirectional relations... (accept_bidirectional_relations={kwargs['accept_bidirectional_relations']})"
        )
    else:
        raise ValueError(
            "The argument 'accept_bidirectional_relations' must be defined"
        )

    mlflow.log_param("original_dataset_size", len(conll_phrases))
    if trim_dataset_size:
        # Randomly select a subset of the dataset. Choose it randomly to avoid bias
        logging.info(
            f"Trimming dataset from {len(conll_phrases)} to {trim_dataset_size} examples"
        )
        conll_phrases = random.sample(conll_phrases, trim_dataset_size)
    mlflow.log_param("trimmed_dataset_size", len(conll_phrases))

    logging.info(f"About to process {len(conll_phrases)} examples...")

    get_matching_heads_sentence = generate_fn_get_matching_heads_sentence(
        model, tokenizer, **kwargs
    )
    phrases_iterator = tqdm.tqdm(conll_phrases, unit="phrase")
    heads_matching_sentence = []
    ids_of_errored_phrases = []
    for i, phrase in enumerate(phrases_iterator):
        try:
            matching_heads_sentence = get_matching_heads_sentence(phrase)
            heads_matching_sentence.append(matching_heads_sentence)
        except UnprocessableSentenceException as e:
            logging.exception(
                f"Error processing sentence {' '.join([word['FORM'] for word in phrase])}"
            )
            ids_of_errored_phrases.append(i)

    logging.info(
        f"Processed {len(heads_matching_sentence)} examples. {len(ids_of_errored_phrases)} examples could not be processed (they raised an UnprocessableSentenceException)"
    )

    # Store a file with the IDs of the errored phrases
    os.makedirs(output_dir / "metadata", exist_ok=True)
    with open(output_dir / "metadata" / "metadata.json", "w") as f:
        json.dump(
            {
                "errored_phrases": ids_of_errored_phrases,
                "number_processed_correctly": len(heads_matching_sentence),
                "number_errored": len(ids_of_errored_phrases),
                "total_number": len(conll_phrases),
                "random_seed": random_seed,
            },
            f,
            indent=4,
        )

    variability_matrix = get_variability_matrix(
        model=model,
        heads_matching_sentence=heads_matching_sentence,
    )

    # Normalize the variability matrix by the maximum value
    variability_matrix_normalized = variability_matrix / variability_matrix.max()

    # Plot the relations
    plot_relations(
        heads_matching_sentence,
        model=model,
        display=False,
        output_dir=output_dir,
        variability_matrix_normalized=variability_matrix_normalized,
        **kwargs,
    )

    uas = calculate_uas(heads_matching_sentence, conll_phrases)
    # Output structure:
    # {
    #    'relation1': torch.Tensor([num_layers, num_heads]),
    #    'relation2': torch.Tensor([num_layers, num_heads]),
    #    ...
    # }
    # Save the UAS to a CSV file
    for relation, uas_matrix in uas.items():
        uas_df = pd.DataFrame(uas_matrix.numpy())
        os.makedirs(output_dir / "uas_scores", exist_ok=True)
        uas_df.to_csv(output_dir / "uas_scores" / f"uas_{relation}.csv")

    # Save the UAS to MLFlow
    # mlflow.log_table("uas", uas_df)
    mlflow.log_artifact(output_dir / "uas_scores" / f"uas_{relation}.csv")

    # Also store the variability as a CSV
    variability_df = pd.DataFrame(variability_matrix_normalized)
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    # Index column: layer
    variability_df.index = [f"layer_{i}" for i in range(num_layers)]
    # Set the column names to the 'head_'+head number
    variability_df.columns = [f"head_{i}" for i in range(num_heads)]
    variability_df.to_csv(
        output_dir / "variability" / f"variability_{relation}.csv",
        index=True,
    )

    mlflow.log_artifact(output_dir / "variability" / f"variability_{relation}.csv")


def get_variability_matrix(
    model: PreTrainedModel,
    heads_matching_sentence: List[Dict[str, Any]],
):
    """
    Get the variability matrix for the model.
    The variability matrix is a matrix of size [num_layers, num_heads] with the variability of each head for each layer.
    """
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    variability_matrix = torch.zeros((num_layers, num_heads))
    for example in heads_matching_sentence:
        for (layer, head_position), variability in zip(
            example["matching_heads_layer_and_head"],
            example["matching_heads_variability"],
        ):
            variability_matrix[layer, head_position] += variability

    return variability_matrix


def generate_fn_get_matching_heads_sentence(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    **kwargs,
):
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
            if row["HEAD"] >= 0
        ]
        dependencies_head_and_dependant = [
            (dependent_word, head_word)
            for (dependent_word, head_word, _) in dependencies
        ]
        dependencies_reltype = [relation for (_, _, relation) in dependencies]

        # Take all the words in the sentence and get the heads matching the relation
        attention_matrix, adjusted_conll_pd = get_attention_matrix(
            conll_pd=conll_pd, model=model, tokenizer=tokenizer
        )
        if kwargs["remove_self_attention"]:
            # Remove the self-attention from the attention matrix
            attention_matrix = attention_matrix.masked_fill(
                torch.eye(attention_matrix.shape[-1], dtype=bool), 0
            )

        heads_matching_rel = heads_matching_relation(
            conll_pd=adjusted_conll_pd,
            attention_matrix=attention_matrix,
            accept_bidirectional_relations=kwargs["accept_bidirectional_relations"],
        )
        # The result is a list of tuples (layer, head, head_word_id, dependent_word_id, dependency)

        matching_heads_layer_and_head = [
            (layer, head) for (layer, head, _, _, _) in heads_matching_rel
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
            "forms": adjusted_conll_pd["FORM"],
            "attention_matrix": attention_matrix,
            "max_attention_weights": max_attention_weights(attention_matrix),
        }

    return get_matching_heads_sentence


# %%
def plot_relations(
    heads_matching_sentence,
    model: PreTrainedModel,
    display=True,
    output_dir=(Path(__file__).parent.parent / "results"),
    variability_matrix_normalized: torch.Tensor = None,
    **kwargs,
):
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
                # TODO: Change this to apply to all decoder models
                if model.config.is_decoder:
                    # For decoder-only models, the dependant has to be posterior to the head, otherwise the attention is not possible
                    if dependant_position < head_position and (
                        not kwargs["accept_bidirectional_relations"]
                    ):
                        # We don't accept bidirectional relations, so it's impossible that we'll have an attention pattern DEPENDANT -> HEAD
                        logging.debug(
                            f'Skipping the dependency between {dependant_position} -> {head_position} ({example["forms"][dependant_position]} -> {example["forms"][head_position]})'
                        )
                        continue
                else:
                    # The model should not be an encoder-decoder. If it is, raise an error
                    if model.config.is_encoder_decoder:
                        raise ValueError(
                            "The model is an encoder-decoder model. This function is not prepared to handle encoder-decoder models"
                        )
                if dependency == relation:
                    total_words_matching_relation += 1
            total_words += len(example["dependencies_reltype"])
        if total_words_matching_relation < kwargs["min_words_matching_relation"]:
            continue

        if variability_matrix_normalized is None:
            # Generate a matrix of zeros
            variability_matrix_normalized = torch.zeros((num_layers, num_heads))

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
            alpha=1.0 - variability_matrix_normalized,
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
        os.makedirs(output_dir / "figures", exist_ok=True)
        os.makedirs(output_dir / "number_of_heads_matching", exist_ok=True)
        os.makedirs(output_dir / "variability", exist_ok=True)

        # Store it as a PDF
        plt.savefig(
            output_dir
            / "figures"
            / f"heads_matching_{relation}_{model.config.model_type}.pdf"
        )
        if display:
            plt.show()
        # Reset the plot
        plt.clf()

        mlflow.log_artifact(
            output_dir
            / "figures"
            / f"heads_matching_{relation}_{model.config.model_type}.pdf"
        )

        # Also store the matrix as a CSV
        matrix_df = pd.DataFrame(matrix)
        # Index column: layer
        matrix_df.index = [f"layer_{i}" for i in range(num_layers)]
        # Set the column names to the 'head_'+head number
        matrix_df.columns = [f"head_{i}" for i in range(num_heads)]
        matrix_df.to_csv(
            output_dir
            / "number_of_heads_matching"
            / f"heads_matching_{relation}_{model.config.model_type}.csv",
            index=True,
        )

        mlflow.log_artifact(
            output_dir
            / "number_of_heads_matching"
            / f"heads_matching_{relation}_{model.config.model_type}.csv"
        )


def calculate_uas(
    heads_matching_sentence: List[Dict[str, Any]],
    conll_phrases: List[List[Dict[str, Any]]],
) -> Dict[str, torch.Tensor]:
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

    number_of_heads_matching_sentence_per_dependency = (
        {}
    )  # This stores a matrix per dependency type, head and layer with the number of heads matching the relation

    total_tokens = 0
    # TODO: Cambiar para que sea una division por el numero de veces que esa relacion aparece en COLNN (numero de hits / numero de veces que aparece en el dataset)

    # Now, for each dependency type, head and layer, calculate the UAS
    for example in heads_matching_sentence:
        # Here, we get the dependency type, head and layer for each word matching the relation
        # And we add 1 to the entry in the matrix for that dependency type, head and layer
        for (layer, head), dependency in zip(
            example["matching_heads_layer_and_head"],
            example["matching_heads_dependency"],
        ):
            if dependency not in number_of_heads_matching_sentence_per_dependency:
                number_of_heads_matching_sentence_per_dependency[dependency] = (
                    torch.zeros((num_layers, num_heads))
                )
            number_of_heads_matching_sentence_per_dependency[dependency][
                layer, head
            ] += 1

        total_tokens += example["max_attention_weights"].shape[-1]

    # Now, we have the number of heads matching each relation for each layer and head
    # We can calculate the UAS for each relation
    uas_per_dependency = {}

    # Also, get the possible maximum for each dependency type. For this, use the gold standard (the CONLL-U file)
    total_relations_per_dependency_in_gold_standard = {
        dependency: 0
        for dependency in number_of_heads_matching_sentence_per_dependency.keys()
    }
    for conll in conll_phrases:
        # conll is a list of dictionaries, each representing a word
        for dependency in total_relations_per_dependency_in_gold_standard.keys():
            total_relations_per_dependency_in_gold_standard[dependency] += len(
                [word for word in conll if word["DEPREL"] == dependency]
            )

    for dependency, matrix in number_of_heads_matching_sentence_per_dependency.items():
        uas_per_dependency[dependency] = (
            # Divide the number of correct heads by the total number of relations in the gold standard, which is the maximum possible
            matrix
            / total_relations_per_dependency_in_gold_standard[dependency]
        )

    return uas_per_dependency


# %%
