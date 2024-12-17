# %%
# Load model directly
import torch
import transformers
from transformers import PreTrainedModel, PreTrainedTokenizer
import re

from attention.conll import parse_to_conllu

# %%
from attention.max_attention_weights import (
    heads_matching_relation,
)


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
    sentence = re.sub(r"(\w)\s([^\w\s])", r"\1\2", sentence)

    # Get the tokenizer from the model
    encodings = tokenizer(
        sentence,
        return_offsets_mapping=True,
        return_tensors="pt",
        return_attention_mask=False,
    )

    # Squeeze the batch dimension
    encodings = {key: value.squeeze(0) for key, value in encodings.items()}

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

    input_ids_as_string_tokens = tokenizer.convert_ids_to_tokens(
        encodings["input_ids"]
    )

    # This is a list of words and their corresponding tokenized words
    words_to_tokenized_words = []
    current_word_index = 0
    position_in_current_word = 0
    current_word_tokens = []
    for token, (start, end) in zip(input_ids_as_string_tokens, encodings["offset_mapping"]):
        # We'll add the lenghts of the tokens until we have a length that is equal to the length of the word
        # Let's imagine we have "you"
        # And it's tokenized into "Ġyo", "u"
        # We'll find the first shared character, "y", and add 1 to the position_in_current_word
        # Then we'll find the second shared character, "o", and add 1 to the position_in_current_word
        # ... and so on
        current_word_tokens.append(token)
        for i, character in enumerate(token):
            if character == words[current_word_index][position_in_current_word]:
                position_in_current_word += 1
                if position_in_current_word > len(words[current_word_index]):
                    raise ValueError(
                        "The tokenized words are longer than the original words"
                    )
                if position_in_current_word == len(words[current_word_index]):
                    # We have reached the end of the word
                    words_to_tokenized_words.append(
                        (words[current_word_index], current_word_tokens)
                    )
                    current_word_index += 1
                    position_in_current_word = 0
                    current_word_tokens = []

        assert current_word_index <= len(words)

    from attention.max_attention_weights import join_subwords

    original_words_stacked_attention_matrix = join_subwords(
        stacked_attentions, words_to_tokenized_words
    )

    return original_words_stacked_attention_matrix.cpu()


# %%
if __name__ == "__main__":
    # This will only run if this file is called directly

    phrase = "The kid likes to eat fish"
    conll_pd = parse_to_conllu(phrase)
    words = conll_pd["FORM"].tolist()

    MODEL = "bert-base-uncased"
    model = transformers.AutoModelForMaskedLM.from_pretrained(MODEL)

    original_words_stacked_attention_matrix = get_attention_matrix(
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
