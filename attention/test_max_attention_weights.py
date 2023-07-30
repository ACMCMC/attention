# %%
# Test file for max attention weights
# Use pytest to run this file

from attention.conll import parse_to_conllu


def test_join_subwords():
    import torch
    from attention.max_attention_weights import join_subwords

    words_to_tokenized_words = [
        ("The", [101]),
        ("Sofa", [102, 103]),
        ("Sofa", [102, 103]),
    ]

    # Matrix of shape [batch_size, num_layers, num_heads, sequence_length, sequence_length]
    attention_matrix = torch.tensor(
        [
            [
                [
                    [
                        [0.1, 0.2, 0.3, 0.2, 0.3],
                        [0.4, 0.5, 0.6, 0.5, 0.6],
                        [0.7, 0.8, 0.9, 0.8, 0.9],
                        [0.4, 0.5, 0.6, 0.5, 0.6],
                        [0.7, 0.8, 0.9, 0.8, 0.9],
                    ],
                ],
                [
                    [
                        [0.1, 0.2, 0.3, 0.2, 0.3],
                        [0.4, 0.5, 0.6, 0.5, 0.6],
                        [0.7, 0.8, 0.9, 0.8, 0.9],
                        [0.4, 0.5, 0.6, 0.5, 0.6],
                        [0.7, 0.8, 0.9, 0.8, 0.9],
                    ],
                ],
            ]
        ]
    )

    # The expected result is a tensor of shape [batch_size, num_layers, num_heads, sequence_length]
    # It should leave the first column as it is, and sum the second and third columns
    # It should leave the first row as it is, and average the second and third rows
    expected = torch.tensor(
        [
            [
                [
                    [
                        [0.1, 0.5, 0.5],
                        [0.55, 1.4, 1.4],
                        [0.55, 1.4, 1.4],
                    ],
                    [
                        [0.1, 0.5, 0.5],
                        [0.55, 1.4, 1.4],
                        [0.55, 1.4, 1.4],
                    ],
                ]
            ]
        ]
    )

    max = join_subwords(attention_matrix, words_to_tokenized_words)
    assert torch.allclose(max, expected)

# %%
