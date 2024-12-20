# %%
# Load model directly
import torch
import transformers
from transformers import PreTrainedModel, PreTrainedTokenizer
import re
import pandas as pd
import unicodedata

from attention.conll import parse_to_conllu
from attention.max_attention_weights import join_subwords

# %%
from attention.max_attention_weights import (
    heads_matching_relation,
)


class UnprocessableSentenceException(Exception):
    pass


def remove_diacritics(text):
    """
    Remove diacritics from the text

    This is useful because some models have trouble with diacritics. Specifically, the tokenization - detokenization process is not perfectly symmetrical so the Unicode characters are not exactly the same. The way to fix this is to remove the diacritics from the text before doing text-based comparisons.
    """
    return "".join([c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c)])


def get_words_to_tokenized_words(input_ids_as_string_tokens, encodings, words_with_diacritics: list):
    """
    Given the input_ids as string tokens and the encodings, get the words and their corresponding tokenized words
    """
    # Remove diacritics in the words - in some models, diacritics impede identification (because the tokenization - detokenization process is not perfectly symmetrical so the Unicode characters are not exactly the same)
    words_without_diacritics = [remove_diacritics(word) for word in words_with_diacritics]

    # This is a list of words and their corresponding tokenized words
    words_to_tokenized_words = []
    current_word_index = 0
    position_in_current_word = 0
    current_word_tokens = []
    for token, (start, end), is_special_token in zip(
        input_ids_as_string_tokens,
        encodings["offset_mapping"].squeeze(),
        encodings["special_tokens_mask"].squeeze(),
    ):
        # If this is a special token, we have to deal with it.
        # For instance, this can happen with the [CLS] token that is added at the beginning of the sentence in BERT
        if is_special_token:
            # We'll add the special token as a word, but that word is None
            words_to_tokenized_words.append((None, [token]))
            continue

        # We'll add the lenghts of the tokens until we have a length that is equal to the length of the word
        # Let's imagine we have "you"
        # And it's tokenized into "Ġyo", "u"
        # We'll find the first shared character, "y", and add 1 to the position_in_current_word
        # Then we'll find the second shared character, "o", and add 1 to the position_in_current_word
        # ... and so on
        current_word_tokens.append(
            token
        )  # By default, assume that this token is part of the current word

        # Remove diacritics in the token - in some models, this impedes identification (because the tokenization - detokenization process is not perfectly symmetrical so the Unicode characters are not exactly the same)
        token_without_diacritics = remove_diacritics(token)

        for i, character in enumerate(token_without_diacritics):
            if character == words_without_diacritics[current_word_index][position_in_current_word]:
                position_in_current_word += 1
                if position_in_current_word > len(words_without_diacritics[current_word_index]):
                    raise ValueError(
                        "The tokenized words are longer than the original words"
                    )
                if position_in_current_word == len(words_without_diacritics[current_word_index]):
                    # We have reached the end of the word
                    words_to_tokenized_words.append(
                        (words_with_diacritics[current_word_index], current_word_tokens)
                    )
                    current_word_index += 1
                    position_in_current_word = 0
                    current_word_tokens = []
                    # If this is not the end of the token, we could add it to the next word (this could fix the case where two words are only one token)
                    # However, this would mean we'd have to rearrange the attention matrix, which is not ideal
                    # So, we'll just ignore the token

        assert current_word_index <= len(words_with_diacritics), f"Current word index: {current_word_index}, words: {words_with_diacritics}"

    # Assert that we have the same number of words and entries in the list, also adding special tokens
    expected_number_of_words = len(words_with_diacritics) + sum(
        encodings["special_tokens_mask"].squeeze()
    )

    if expected_number_of_words != len(words_to_tokenized_words):
        raise UnprocessableSentenceException(
            f"Words: {words_with_diacritics}, "
            f"List: {words_to_tokenized_words}"
        )

    return words_to_tokenized_words


def adjust_conll_pd(conll_pd, words_to_tokenized_words):
    """
    Given that the model tokenizes the words, there might be a few extra tokens or missing tokens in the tokeniation that it performs.
    We need to account for this in the CoNLL-U DataFrame by shifting the ID, HEAD and DEPS columns accordingly.
    """
    # The words_to_tokenized_words list has three possible cases:
    # 1. The word is None (special token): In this case, there is 1+ token that corresponds to no word. We need to shift the conll_pd forward (i.e. add)
    # 2. The word is not None, but its list of tokenized words is empty: This means that the model tokenized 2+ words into in token. We need to account for this by shifting the conll_pd backwards (i.e. subtract)
    # 3. The word is not None, and its list of tokenized words is not empty: This is the normal case, where we have 1+ token that corresponds to 1 word. We don't need to do anything

    # Use a working copy of the index to know what everything should be re-assigned to
    index_working_copy = conll_pd.index.to_numpy().copy()

    for i, (word, tokenized_words) in enumerate(words_to_tokenized_words):
        if word is None:
            # This is a special token - a word that does not show up in the dataframe but we still need to shift the indices forward
            # Only shift from this point onwards, leave all previous indices as they are
            index_working_copy[i:] += 1
        elif len(tokenized_words) == 0:
            # This is a word that was tokenized into 0 tokens - we need to shift the indices backwards
            # Only shift from this point onwards, leave all previous indices as they are
            index_working_copy[i:] -= 1

    # Now, re-assign the indices (ID)
    # The old index had a name ("ID") that we need to keep
    # Turn the index_working_copy into a named index
    index_working_copy = pd.Index(index_working_copy, name="ID")
    conll_pd = conll_pd.set_index(index_working_copy, inplace=False)
    # Also, re-assign the HEAD column by using the index_working_copy as a lookup table
    conll_pd["HEAD"] = conll_pd["HEAD"].map(
        # With the current value, look up the new value in that position of the index_working_copy
        lambda head: index_working_copy[head] if head >= 0 else head # If head is -1, keep it as -1 (root)
    )
    # Also, re-assign the DEPS column by using the index_working_copy as a lookup table
    conll_pd["DEPS"] = conll_pd["DEPS"].map(
        # Deps is a list of ('relation', head) tuples
        lambda deps: [
            (relation, index_working_copy[head]) if head >= 0 else (relation, head) # If head is -1, keep it as -1 (root)
            for relation, head in deps
        ]
    )
    return conll_pd


def get_attention_matrix(
    conll_pd, model: PreTrainedModel, tokenizer: PreTrainedTokenizer
):
    """
    Get the attention matrix for a given phrase

    Args:
        conll_pd (pandas.DataFrame): CoNLL-U format DataFrame
        model (PreTrainedModel): Model to get the attention from

    Returns:
        torch.Tensor: Attention matrix for the phrase. Shape: [batch_size, num_layers, num_heads, sequence_length, sequence_length]
    """
    words = conll_pd["FORM"].tolist()
    # Join the words into a single string, but keep the words separated by spaces (beware: commas, periods, etc. would get an extra space)
    sentence = " ".join(words)
    # We need to take care of that extra space that would be added between punctuation and the previous word
    # e.g.: 'Over the past decades , the correction for sea surface roughness effects were studied based on the in - situ and airborne measurements ; for example , the experiments made from a tower [ 12 ] , wind and salinity experiments ( WISE ) [ 13 , 14 ] , airborne Passive - Active L - band Sensor ( PALS ) campaign [ 15 ] and Combined Airborne Radio instruments for Ocean and Land Studies ( CAROLS ) campaigns [ 16 , 17 ] .'
    sentence = re.sub(r"([\w\{\[\(])\s([^\w\s\[\(\{])", r"\1\2", sentence)
    # -> 'Over the past decades, the correction for sea surface roughness effects were studied based on the in- situ and airborne measurements; for example, the experiments made from a tower [ 12] , wind and salinity experiments ( WISE) [ 13, 14] , airborne Passive- Active L- band Sensor ( PALS) campaign [ 15] and Combined Airborne Radio instruments for Ocean and Land Studies ( CAROLS) campaigns [ 16, 17] .'
    sentence = re.sub(r"([\-\–\[\(\{])\s", r"\1", sentence)
    # -> 'Over the past decades, the correction for sea surface roughness effects were studied based on the in-situ and airborne measurements; for example, the experiments made from a tower [12] , wind and salinity experiments (WISE) [13, 14] , airborne Passive-Active L-band Sensor (PALS) campaign [15] and Combined Airborne Radio instruments for Ocean and Land Studies (CAROLS) campaigns [16, 17] .'
    sentence = re.sub(r"([\}\]\)])\s([^\w\s])", r"\1\2", sentence)
    # -> 'Over the past decades, the correction for sea surface roughness effects were studied based on the in-situ and airborne measurements; for example, the experiments made from a tower [12], wind and salinity experiments (WISE)[13, 14], airborne Passive-Active L-band Sensor (PALS) campaign [15] and Combined Airborne Radio instruments for Ocean and Land Studies (CAROLS) campaigns [16, 17].'

    # Get the tokenizer from the model
    encodings = tokenizer(
        sentence,
        return_offsets_mapping=True,
        return_tensors="pt",
        return_attention_mask=False,
        return_special_tokens_mask=True,
    )

    # Get the attention values from the model
    model.eval()
    with torch.no_grad():
        # Move to the same device as the model
        input_ids = encodings["input_ids"].to(model.device)
        outputs = model(input_ids=input_ids, output_attentions=True)

    # Returns a tuple with the attention tensors, one for each layer
    # The size of each tensor is [batch_size, num_heads, sequence_length, sequence_length]
    # Join them in a single tensor, of size [batch_size, num_layers, num_heads, sequence_length, sequence_length]
    unstacked_attentions = outputs["attentions"]
    stacked_attentions = torch.stack(unstacked_attentions, dim=1)

    try:
        input_ids_as_string_tokens = tokenizer.convert_ids_to_tokens(
            encodings["input_ids"].squeeze()
        )
    except TypeError as e:
        # If the sentence is empty, the input_ids will be empty, and we can't convert that to tokens
        # It would throw TypeError("iteration over a 0-d tensor")
        raise UnprocessableSentenceException(
            f"Could not convert input_ids to tokens for sentence {sentence}"
        ) from e

    words_to_tokenized_words = get_words_to_tokenized_words(
        input_ids_as_string_tokens=input_ids_as_string_tokens,
        encodings=encodings,
        words_with_diacritics=words,
    )

    original_words_stacked_attention_matrix = join_subwords(
        attention_matrix=stacked_attentions,
        special_tokens_mask=encodings["special_tokens_mask"],
        words_to_tokenized_words=words_to_tokenized_words,
    )

    try:
        adjusted_conll_pd = adjust_conll_pd(
            conll_pd=conll_pd, words_to_tokenized_words=words_to_tokenized_words
        )
    except TypeError as e:
        raise UnprocessableSentenceException(
            f"Could not adjust the CoNLL-U DataFrame for sentence {sentence}. Conll: {conll_pd}, words_to_tokenized_words: {words_to_tokenized_words}"
        ) from e

    return original_words_stacked_attention_matrix.cpu(), adjusted_conll_pd


# %%
if __name__ == "__main__":
    # This will only run if this file is called directly

    phrase = "The kid likes to eat fish"
    conll_pd = parse_to_conllu(phrase)
    words = conll_pd["FORM"].tolist()

    MODEL = "bert-base-uncased"
    model = transformers.AutoModelForMaskedLM.from_pretrained(MODEL)

    original_words_stacked_attention_matrix, adjusted_conll_pd = get_attention_matrix(
        conll_pd, model=model
    )
    original_words_unstacked_attentions = torch.unbind(
        original_words_stacked_attention_matrix, dim=1
    )

    # Use BERTviz to visualize the attention
    from bertviz import model_view

    # Convert attention into (batch_size(must be 1), num_heads, sequence_length, sequence_length)
    model_view(
        attention=original_words_unstacked_attentions,
        tokens=words,
    )

    heads_matching_rel = heads_matching_relation(
        conll_pd, attention_matrix=original_words_stacked_attention_matrix
    )
    print(heads_matching_rel)
# %%
