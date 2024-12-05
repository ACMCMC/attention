"""
Maximum Attention Weights

We assign a relation between word wi and wj if j = argmax W[i] for each row (that corresponds to a word in attention matrix) i in attention matrix W.
"""

import pandas as pd
# %%
import torch


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
    conll: pd.DataFrame, attention_matrix: torch.Tensor, accept_bidirectional_relations=False
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
def join_subwords(attention_matrix: torch.Tensor, words_to_tokenized_words: list):
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
    for i, (word, tokenized_words) in enumerate(words_to_tokenized_words):
        # If there is only one subword, there is no need to sum
        if len(tokenized_words) == 1:
            i += 1
            continue
        # Sum the attention weights of the subwords on dimension -1 for the tokens that belong to the same word
        # print(f'old: {new_attention_matrix[0,0,0,:,:]}')
        for j in range(len(tokenized_words)):
            if j == 0:
                continue
            new_attention_matrix[:, :, :, :, i] += new_attention_matrix[
                :, :, :, :, i + j
            ]
            # Set the attention weights of the subwords to 0
            new_attention_matrix[:, :, :, :, i + j] = 0
        # Remove the positions from i + 1 to i + j
        new_attention_matrix = torch.cat(
            [
                new_attention_matrix[:, :, :, :, : i + 1],
                new_attention_matrix[:, :, :, :, i + j + 1 :],
            ],
            dim=-1,
        )
        # Now, for the dimension -2, we need to average the attention weights of the subwords
        for j in range(len(tokenized_words)):
            if j == 0:
                continue
            new_attention_matrix[:, :, :, i, :] += new_attention_matrix[
                :, :, :, i + j, :
            ]
            # Set the attention weights of the subwords to 0
            new_attention_matrix[:, :, :, i + j, :] = 0
        # Average the attention weights of the subwords
        new_attention_matrix[:, :, :, i, :] /= len(tokenized_words)
        # Remove the positions from i + 1 to i + j
        new_attention_matrix = torch.cat(
            [
                new_attention_matrix[:, :, :, : i + 1, :],
                new_attention_matrix[:, :, :, i + j + 1 :, :],
            ],
            dim=-2,
        )
        # print(f'new: {new_attention_matrix[0,0,0,:,:]}')
        i += len(tokenized_words)
    return new_attention_matrix


# %%
