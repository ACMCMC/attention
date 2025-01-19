# %%
# Test file for max attention weights
# Use pytest to run this file

import torch

from attention.variability import get_relative_variability


def test_std_dev_0():
    # Matrix of shape [batch_size, num_layers, num_heads, sequence_length, sequence_length]
    attention_matrix = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ],
    )

    std_dev = get_relative_variability(attention_matrix)
    assert torch.all(std_dev == 0)


def test_std_dev_non0():
    attention_matrix = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
        ],
    )

    std_dev = get_relative_variability(attention_matrix)
    assert torch.all(std_dev != 0)


def test_batch():
    random_attn = torch.rand([3, 3, 3, 3]).softmax(dim=-1)
    identity_attn = torch.tensor(
        [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    )
    identity_attn = torch.stack([identity_attn] * 3)
    identity_attn = torch.stack([identity_attn] * 3)
    stacked_matrix = torch.stack([random_attn, identity_attn], dim=0)

    result = get_relative_variability(stacked_matrix)

    assert torch.all(result[0] != 0)
    assert torch.all(result[1] == 0)


# %%
