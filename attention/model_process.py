# %%
# Load model directly
import torch
import transformers
from transformers import PreTrainedModel, PreTrainedTokenizer

from attention.conll import parse_to_conllu

# %%
from attention.max_attention_weights import (
    heads_matching_relation,
)


def get_attention_matrix(conll_pd, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    """
    Get the attention matrix for a given phrase

    Args:
        conll_pd (pandas.DataFrame): CoNLL-U format DataFrame
        model (PreTrainedModel): Model to get the attention from

    Returns:
        torch.Tensor: Attention matrix for the phrase. Shape: [batch_size, num_layers, num_heads, sequence_length, sequence_length]
    """
    words = conll_pd["FORM"].tolist()
    # Get the tokenizer from the model
    tokenized_words = tokenizer(words)["input_ids"]
    words_to_tokenized_words = list(zip(words, tokenized_words))
    # Concatenate the list of lists into a single tensor
    input_ids = torch.concat(
        [torch.tensor(x) for x in tokenized_words], dim=0
    ).unsqueeze(0)
    input_ids_str = tokenizer.convert_ids_to_tokens(input_ids[0])
    # print(input_ids)

    # Get the attention values from the model
    model.eval()
    with torch.no_grad():
        # If the model is on the GPU, move the input to the GPU
        if model.device.type == "cuda":
            input_ids = input_ids.cuda()
        outputs = model(input_ids=input_ids, output_attentions=True)

    # Returns a tuple with the attention tensors, one for each layer
    # The size of each tensor is [batch_size, num_heads, sequence_length, sequence_length]
    # Join them in a single tensor, of size [batch_size, num_layers, num_heads, sequence_length, sequence_length]
    unstacked_attentions = outputs["attentions"]
    stacked_attentions = torch.stack(unstacked_attentions, dim=1)

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

    heads_matching_rel = heads_matching_relation(conll_pd, attention_matrix=original_words_stacked_attention_matrix)
    print(heads_matching_rel)
# %%
