# %%
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
                "If",
                "if",
                "SCONJ",
                "IN",
                None,
                2,
                "verb_auxiliars",
                list([("mark", 2)]),
                1,
            )(
                1,
                "you",
                "you",
                "PRON",
                "PRP",
                {"Case": "Nom", "Number": "Sing", "Person": "2", "PronType": "Prs"},
                2,
                "nominal_verb_arguments",
                list([("nsubj", 2)]),
                2,
            )(
                2,
                "live",
                "live",
                "VERB",
                "VBP",
                {
                    "Mood": "Ind",
                    "Number": "Sing",
                    "Person": "2",
                    "Tense": "Pres",
                    "VerbForm": "Fin",
                },
                12,
                "advcl",
                list([("advcl:if", 12)]),
                3,
            )(
                3,
                "in",
                "in",
                "ADP",
                "IN",
                None,
                8,
                "noun_specifiers",
                list([("case", 8)]),
                4,
            )(
                4,
                "or",
                "or",
                "CCONJ",
                "CC",
                None,
                5,
                "conjunctions",
                list([("cc", 5)]),
                5,
            )(
                5,
                "near",
                "near",
                "ADP",
                "IN",
                None,
                3,
                "conjunctions",
                list([("conj:or", 3), ("case", 8)]),
                6,
            )(
                6,
                "a",
                "a",
                "DET",
                "DT",
                {"Definite": "Ind", "PronType": "Art"},
                8,
                "noun_specifiers",
                list([("det", 8)]),
                7,
            )(
                7,
                "big",
                "big",
                "ADJ",
                "JJ",
                {"Degree": "Pos"},
                8,
                "noun_modifiers",
                list([("amod", 8)]),
                8,
            )(
                8,
                "city",
                "city",
                "NOUN",
                "NN",
                {"Number": "Sing"},
                2,
                "nominal_verb_arguments",
                list([("obl:in", 2)]),
                9,
            )(
                9,
                ".",
                ".",
                "PUNCT",
                ".",
                None,
                2,
                "nominal_verb_arguments",
                list([("punct", 2)]),
                10,
            )
        ]
    )
    indices_of_max_attention = torch.Tensor()
    result = heads_matching_relation(
        conll=conll,
        indices_of_max_attention=indices_of_max_attention,
        accept_bidirectional_relations=False,
    )

    assert result == []


# %%
