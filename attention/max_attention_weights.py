"""
Maximum Attention Weights

We assign a relation between word wi and wj if j = argmax W[i] for each row (that corresponds to a word in attention matrix) i in attention matrix W.
"""

import pandas as pd
from typing import List, Tuple
import torch
import logging


def max_attention_weights(attention_matrix):
    """
    Returns the maximum attention weights for each row in the attention matrix

    Args:
        attention_matrix (torch.Tensor): Attention matrix of shape [batch_size, num_layers, num_heads, sequence_length, sequence_length]

    Returns:
        torch.Tensor: Maximum attention weights for each row in the attention matrix. Shape: [batch_size, num_layers, num_heads, sequence_length]
    """
    max = torch.max(attention_matrix, dim=-1)
    return max.indices


# %%


def heads_matching_relation(
    conll_pd: pd.DataFrame,
    attention_matrix: torch.Tensor,
    accept_bidirectional_relations=False,
):
    """
    Returns the heads matching relation between words in the phrase. There is a relation if conll row['HEAD'] == row['ID'].

    Args:
        conll (pandas.DataFrame): CoNLL-U format DataFrame
        indices_of_max_attention (torch.Tensor): Maximum attention weights for each row in the attention matrix of shape [batch_size, num_layers, num_heads, sequence_length]
        accept_didirectional_relations (bool): whether to consider DEPENDANT -> HEAD only, or DEPENDANT <-> HEAD (bidirectional)

    Returns:
        list: List of tuples with the relation between words in the phrase. Each tuple is (layer, head, head_word, dependent_word)
    """
    heads_matching_relations = []

    indices_of_max_attention = max_attention_weights(attention_matrix)

    def process_one_row(head_word_index: int, dependant_word_index: int, deprel: str):
        # Go through all the words in the phrase to find the heads matching relation

        # From the attention matrix, get all the elements where the last dimension in position i is equal to the head. Get their indices only
        heads_matching_this_relation = torch.where(
            indices_of_max_attention[:, :, :, dependant_word_index] == (head_word_index)
        )  # -1 because the indices start at 0
        # Iterate over the indices of the heads matching this relation
        # print(f'Heads matching this relation: {heads_matching_this_relation}')
        for batch, layer, head_index in zip(*heads_matching_this_relation):
            heads_matching_relations.append(
                # Return word indices instead of the word itself because we might have multiple instances of the same word in the phrase (e.g. "the" in "the man likes the proposal")
                (layer.item(), head_index.item(), head_word_index, dependant_word_index, deprel)
            )

    for current_row_index, row in conll_pd.iterrows():
        # The index is not necessarily monotonic (e.g. 1, 3, 4, 5...), but this is OK as the iterrows() function will return the value in the ID index, rather than a pure range index (0, 1, 2...)
        dependant_word_index = current_row_index
        head_word_index = row["HEAD"]
        deprel = row["DEPREL"]
        # If the head is -1, then it is the root word, so we skip it
        if head_word_index == -1:
            continue  # Skip the root word. No need to check in the reverse direction because this word only "depends on" itself
        process_one_row(
            head_word_index=head_word_index,
            dependant_word_index=dependant_word_index,
            deprel=deprel,
        )

        if accept_bidirectional_relations:
            # Repeat the process, but in the reverse order (HEAD -> DEPENDANT)

            dependant_word_index = row["HEAD"]
            head_word_index = current_row_index
            deprel = row["DEPREL"]
            process_one_row(
                head_word_index=head_word_index,
                dependant_word_index=dependant_word_index,
                deprel=deprel,
            )
    return heads_matching_relations


# %%
def join_subwords(
    attention_matrix: torch.Tensor,
    special_tokens_mask: torch.Tensor,
    words_to_tokenized_words: List[Tuple[str, List[str]]],
):
    """
    Joins the attention matrix of subwords into a matrix of words

    Args:
        attention_matrix (torch.Tensor): Attention matrix of shape [batch_size, num_layers, num_heads, sequence_length, sequence_length]
        words_to_tokenized_words (list): List of tuples with the original word and the list of subwords

    Returns:
        torch.Tensor: Attention matrix of shape [batch_size, num_layers, num_heads, sequence_length, sequence_length]
    """
    new_attention_matrix = attention_matrix.clone()
    # Sum the attention weights of the subwords on dimension -1 for the tokens that belong to the same word
    # The result is a tensor of shape [batch_size, num_heads, sequence_length, words]
    current_position = 0
    for word, tokenized_words in words_to_tokenized_words:
        if word is None:
            # This can happen with e.g. the [CLS] token in BERT
            # In this case, leave it as-is (treat it as any other token)
            pass

        if len(tokenized_words) == 0:
            # This should never happen
            # There may be some very rare cases where 1+ CONLL-U words are tokenized into 1 (e.g. CONLL ["[", "16"] -> ["[16"])
            logging.debug(
                f"The tokenized words list is empty for word {word} (in {words_to_tokenized_words})"
            )
        elif len(tokenized_words) == 1:
            # If there is only one tokenized word, there is nothing to do
            current_position += 1
        else:
            # Sum the attention weights of the subwords on dimension -1 for the tokens that belong to the same word
            for j in range(len(tokenized_words)):
                if j == 0:
                    continue
                new_attention_matrix[
                    :, :, :, :, current_position
                ] += new_attention_matrix[:, :, :, :, current_position + j]
                # Set the attention weights of the subwords to 0
                new_attention_matrix[:, :, :, :, current_position + j] = 0
            # Remove the positions from i + 1 to i + j
            new_attention_matrix = torch.cat(
                [
                    new_attention_matrix[:, :, :, :, : current_position + 1],
                    new_attention_matrix[:, :, :, :, current_position + j + 1 :],
                ],
                dim=-1,
            )
            # Now, for the dimension -2, we need to average the attention weights of the subwords
            for j in range(len(tokenized_words)):
                if j == 0:
                    continue
                new_attention_matrix[
                    :, :, :, current_position, :
                ] += new_attention_matrix[:, :, :, current_position + j, :]
                # Set the attention weights of the subwords to 0
                new_attention_matrix[:, :, :, current_position + j, :] = 0
            # Average the attention weights of the subwords
            new_attention_matrix[:, :, :, current_position, :] /= len(tokenized_words)
            # Remove the positions from i + 1 to i + j
            new_attention_matrix = torch.cat(
                [
                    new_attention_matrix[:, :, :, : current_position + 1, :],
                    new_attention_matrix[:, :, :, current_position + j + 1 :, :],
                ],
                dim=-2,
            )

            current_position += 1  # Move to the next word

    # Assert that the new attention matrix has the same shape in the last two dimensions as the number of words
    expected_final_number_of_tokens = len(words_to_tokenized_words)
    # Subtract the elements that have empty tokenized word lists
    expected_final_number_of_tokens -= len(
        [
            word
            for word, tokenized_words in words_to_tokenized_words
            if len(tokenized_words) == 0
        ]
    )

    expected_final_shape = torch.Size(
        attention_matrix.size()[:-2]
        + (expected_final_number_of_tokens, expected_final_number_of_tokens)
    )
    assert new_attention_matrix.size() == expected_final_shape

    return new_attention_matrix


# %%
