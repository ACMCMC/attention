import pytest
import torch

from attention.dataset_eval import calculate_uas


def test_calculate_uas():
    # A model with two layers and two heads.
    # The sentence is "The kid eats fish" (4 words)
    # We will assume that none of the heads get it right, except for heads (0,0), (0,1) and (1,1) for dependency type "x" in one of the words
    heads_matching_sentence = [
        {
            "max_attention_weights": torch.tensor(
                [  # batch_size (1)
                    [  # num_layers (2)
                        [  # num_heads (2)
                            [0, 0, 2, 3],  # 0
                            [0, 0, 2, 3],  # 1
                        ],
                        [  # num_heads (2)
                            [0, 1, 2, 3],  # 0
                            [0, 0, 2, 3],  # 1
                        ],
                    ]
                ]
            ),
            "matching_heads_layer_and_head": [
                # (layer, head_position)
                (0, 0),  # e.g. kid -> the
                (0, 1),  # e.g. kid -> the
                (1, 1),  # e.g. kid -> eats
            ],
            "matching_heads_dependency": [
                "x",
                "x",
                "x",
            ],
        }
    ]
    assert heads_matching_sentence[0]["max_attention_weights"].shape == (1, 2, 2, 4)
    uas = calculate_uas(heads_matching_sentence)
    expected_result = torch.tensor(
        [  # num_layers (2)
            [0.25, 0.25],
            [0.0, 0.25],
        ]
    )
    assert torch.equal(uas["x"], expected_result)
