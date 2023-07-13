"""
Maximum Attention Weights

We assign a relation between word wi and wj if j = argmax W[i] for each row (that corresponds to a word in attention matrix) i in attention matrix W.
"""
# %%
import torch

def max_attention_weights(attention_matrix):
    """
    Returns the maximum attention weights for each row in the attention matrix

    Args:
        attention_matrix (torch.Tensor): Attention matrix of shape [batch_size, num_heads, sequence_length, sequence_length]

    Returns:
        torch.Tensor: Maximum attention weights for each row in the attention matrix
    """
    max = torch.max(attention_matrix, dim=-1)
    return max.indices
# %%

def heads_matching_relation(conll, indices_of_max_attention):
    """
    Returns the heads matching relation between words in the phrase. There is a relation if conll row['HEAD'] == row['ID'].

    Args:
        conll (pandas.DataFrame): CoNLL-U format DataFrame
        indices_of_max_attention (torch.Tensor): Maximum attention weights for each row in the attention matrix of shape [batch_size, num_layers, num_heads, sequence_length]

    Returns:
        list: List of tuples with the relation between words in the phrase. Each tuple is (layer, head, head_word, dependent_word)
    """
    heads_matching_relations = []
    for i, row in conll.iterrows():
        # Go through all the words in the phrase to find the heads matching relation
        head = row['HEAD']
        head_word = conll.iloc[head - 1]["FORM"]
        current_word = row["FORM"]
        print(f'Word: {row["FORM"]}, Head: {conll.iloc[head - 1]["FORM"]}')
        if head == 0:
            continue # Skip the root word
        # From the attention matrix, get all the elements where the last dimension in position i is equal to the head. Get their indices only
        heads_matching_this_relation = torch.where(indices_of_max_attention[:, :, :, i] == head)
        # Iterate over the indices of the heads matching this relation
        print(f'Heads matching this relation: {heads_matching_this_relation}')
        for batch, layer, head in zip(*heads_matching_this_relation):
            heads_matching_relations.append((layer.item(), head.item(), head_word, current_word))
    return heads_matching_relations

# %%
