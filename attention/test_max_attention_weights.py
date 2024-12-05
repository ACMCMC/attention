# Test file for max attention weights
# Use pytest to run this file
import torch
import pandas as pd


def test_join_subwords():
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


def test_heads_matching_relation():
    from attention.max_attention_weights import heads_matching_relation

    conll = pd.DataFrame.from_records(
        data=[
            (
                0,
                "The",
                "the",
                "UPOS",
                "_",
                None,
                2,
                "det",
                list([("det", 2)]),
                None,
            ),(
                1,
                "large",
                "large",
                "UPOS",
                "_",
                None,
                2,
                "amod",
                list([("amod", 2)]),
                None,
            ),(
                2,
                "house",
                "house",
                "UPOS",
                "_",
                None,
                -1,
                "root",
                None,
                None,
            )
        ]
    )

    conll.columns = ["ID", "FORM", "LEMMA", "UPOS", "XPOS", "FEATS", "HEAD", "DEPREL", "DEPS", "MISC"]

    # A Tensor of a fake model with 2 layers and 3 heads. Batch size is 1. The seq length is 3, so the shape is [batch_size, num_layers, num_heads, sequence_length, sequence_length]
    attention_matrix = torch.zeros([1, 2, 3, 3, 3])
    # Since this is a right-masked model, position 0 can only attend to itself, in all layers and heads
    attention_matrix[0, :, :, 0, 0] = torch.ones([2, 3])
    # Head 0 attends to the first word in the sentence, always
    attention_matrix[0, :, 0, 1:, 0] = torch.ones([2, 2])
    # Head 1 attends to the second word in the sentence, for positions 1 and 2 
    attention_matrix[0, :, 1, 1:, 1] = torch.ones([2, 2])
    # Head 2 attends to the second word in the sentence, for position 1
    attention_matrix[0, :, 2, 1:2, 1] = torch.ones([2, 1])
    # Head 2 attends to the third word in the sentence, for position 2
    attention_matrix[0, :, 2, 2:3, 2] = torch.ones([2, 1])

    result = heads_matching_relation(
        conll=conll,
        attention_matrix=attention_matrix,
        accept_bidirectional_relations=False,
    )

    # We expect to see that nothing matches, since the head is the last word in the sentence and it can't be attended by any other word
    assert result == []

    # Now, repeat the process, but with bidirectional relations
    result = heads_matching_relation(
        conll=conll,
        attention_matrix=attention_matrix,
        accept_bidirectional_relations=True,
    )

    # We expect to see that there are identified relations now: Each tuple is (layer, head, head_word, dependent_word)
    expected = [
        (0, 0, "house", "The"),
        (1, 0, "house", "The"),
        (0, 1, "house", "large"),
        (1, 1, "house", "large"),
    ]
    assert result == expected
