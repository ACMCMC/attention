# %%
import torch
from attention.max_attention_weights import max_attention_weights

def get_relative_variability_std_dev(attentions_matrix: torch.Tensor):
    """
    [NOT USED]
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
def get_relative_variability(attentions_matrix: torch.Tensor):
    max_attentions = max_attention_weights(attentions_matrix)

    # subtract 0,1,2,3,4,... from each row
    # First, generate a tensor of the same shape as the last dimension of the attention matrix
    sequence_length = attentions_matrix.size()[-1]
    positions_matrix = torch.tensor(
        [i for i in range(sequence_length)]
    ).type(torch.FloatTensor)

    # Now, subtract the positions matrix from the max_attentions matrix
    relative_positions = max_attentions - positions_matrix

    # Now, get the standard deviation of the relative positions
    std_dev = relative_positions.std(dim=-1)

    # Normalize the standard deviation so that it is in the range [0,1]
    std_dev = std_dev / std_dev.max()

    return std_dev
# %%
