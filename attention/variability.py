# %%
import torch


def get_relative_variability(attentions_matrix: torch.Tensor):
    """
    Returns a variability score based on whether the heads attend to a position X+n or not.

    Args:
        attention_matrix (torch.Tensor): Attention matrix of shape [batch_size, num_layers, num_heads, sequence_length, sequence_length]
    Returns:
        float: A score in the range [0,1]
    """
    # Step 1.
    # Get the average of the relative positions that each token is attending to.
    # For example, if token A attends to A with 0.2, B with 0.3, and C with 0.5, then the average is 0*0.2 + 1*0.3 + 2*0.5
    # To calculate it, first generate a matrix of the relative positions of each token (m[i,j]=j-i)
    # Generate a matrix of this form:
    # [
    #   [ 0, 1, 2],
    #   [-1, 0, 1],
    #   [-2,-1, 0],
    # ]
    sequence_length = attentions_matrix.size()[-1]
    positions_matrix = torch.tensor(
        [[j - i for j in range(sequence_length)] for i in range(sequence_length)]
    ).type(torch.FloatTensor)
    # Now, get the elementwise product of that matrix by the attentions matrix
    weighted_positions_matrix = attentions_matrix * positions_matrix
    # Now, sum the elements of the weighted positions matrix in the last dimension
    average_relative_position = weighted_positions_matrix.sum(dim=-1)
    # Step 2.
    # Get the standard deviation of the relative positions that each token is attending to.
    std_dev = average_relative_position.std(dim=-1)
    return std_dev

# %%
