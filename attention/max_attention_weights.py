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
    conll: pd.DataFrame,
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

    def process_one_row(head_index, dependant_index):
        # Go through all the words in the phrase to find the heads matching relation
        head_word = conll.loc[head_index]["FORM"]
        current_word = conll.loc[dependant_index]["FORM"]
        # print(f'head: {head}, i: {i}, head_word: {head_word}, current_word: {current_word}')
        # From the attention matrix, get all the elements where the last dimension in position i is equal to the head. Get their indices only
        heads_matching_this_relation = torch.where(
            indices_of_max_attention[:, :, :, dependant_index] == (head_index)
        )  # -1 because the indices start at 0
        # Iterate over the indices of the heads matching this relation
        # print(f'Heads matching this relation: {heads_matching_this_relation}')
        for batch, layer, head_index in zip(*heads_matching_this_relation):
            heads_matching_relations.append(
                (layer.item(), head_index.item(), current_word, head_word)
            )

    for current_row_index, row in conll.iterrows():
        dependant_index = current_row_index
        head_index = row["HEAD"]
        # If the head is -1, then it is the root word, so we skip it
        if head_index == -1:
            continue  # Skip the root word. No need to check in the reverse direction because this word only "depends on" itself
        process_one_row(head_index=head_index, dependant_index=dependant_index)

        if accept_bidirectional_relations:
            # Repeat the process, but in the reverse order (HEAD -> DEPENDANT)

            # This is the inverse of above
            dependant_index = row["HEAD"]
            head_index = current_row_index
            process_one_row(head_index=head_index, dependant_index=dependant_index)
    return heads_matching_relations


# %%
def join_subwords(
    attention_matrix: torch.Tensor,
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
    for (word, tokenized_words) in words_to_tokenized_words:
        if len(tokenized_words) == 0:
            # There may be some very rare cases where 1+ CONLL-U words are tokenized into 1 (e.g. CONLL ["[", "16"] -> ["[16"])
            # Add a fake row and column of zeros
            new_attention_matrix = torch.cat(
                [
                    new_attention_matrix[:, :, :, :, : current_position],
                    torch.zeros(
                        new_attention_matrix.shape[0],
                        new_attention_matrix.shape[1],
                        new_attention_matrix.shape[2],
                        new_attention_matrix.shape[3],
                        1,
                    ),
                    new_attention_matrix[:, :, :, :, current_position :],
                ],
                dim=-1,
            )
            new_attention_matrix = torch.cat(
                [
                    new_attention_matrix[:, :, :, : current_position, :],
                    torch.zeros(
                        new_attention_matrix.shape[0],
                        new_attention_matrix.shape[1],
                        new_attention_matrix.shape[2],
                        1,
                        new_attention_matrix.shape[4],
                    ),
                    new_attention_matrix[:, :, :, current_position :, :],
                ],
                dim=-2,
            )
        elif len(tokenized_words) == 1:
            pass
        else:
            # Sum the attention weights of the subwords on dimension -1 for the tokens that belong to the same word
            # print(f'old: {new_attention_matrix[0,0,0,:,:]}')
            for j in range(len(tokenized_words)):
                if j == 0:
                    continue
                new_attention_matrix[:, :, :, :, current_position] += new_attention_matrix[
                    :, :, :, :, current_position + j
                ]
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
                new_attention_matrix[:, :, :, current_position, :] += new_attention_matrix[
                    :, :, :, current_position + j, :
                ]
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
            # print(f'new: {new_attention_matrix[0,0,0,:,:]}')
        current_position += 1 # Move to the next word
    
    # Assert that the new attention matrix has the same shape in the last two dimensions as the number of words
    assert new_attention_matrix.shape[-1] == len(words_to_tokenized_words)
    assert new_attention_matrix.shape[-2] == len(words_to_tokenized_words)

    return new_attention_matrix


# %%
