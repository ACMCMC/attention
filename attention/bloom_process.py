# %%
# Load model directly
from transformers import AutoTokenizer, AutoModel
import torch
from attention.conll import parse_to_conllu

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
model = AutoModel.from_pretrained("bigscience/bloom-560m")
# %%

phrase = "The girlies like to eat fish"
conll_pd = parse_to_conllu(phrase)
words = conll_pd['FORM'].tolist()
tokenized_words = tokenizer(words)['input_ids']
words_to_tokenized_words = list(zip(words, tokenized_words))
# Concatenate the list of lists into a single tensor
input_ids = torch.concat([torch.tensor(x) for x in tokenized_words], dim=0).unsqueeze(0)
input_ids_str = tokenizer.convert_ids_to_tokens(input_ids[0])
print(input_ids)
# %%

# Get the attention values from the model
model.eval()
with torch.no_grad():
    outputs = model(input_ids=input_ids, output_attentions=True)

# Returns a tuple with the attention tensors, one for each layer
# The size of each tensor is [batch_size, num_heads, sequence_length, sequence_length]
# Join them in a single tensor, of size [batch_size, num_layers, num_heads, sequence_length, sequence_length]
unstacked_attentions = outputs['attentions']
stacked_attentions = torch.stack(unstacked_attentions, dim=1)
print(stacked_attentions.shape)

# %%
from attention.max_attention_weights import join_subwords
original_words_stacked_attention_matrix = join_subwords(stacked_attentions, words_to_tokenized_words)
original_words_unstacked_attentions = torch.unbind(original_words_stacked_attention_matrix, dim=1)
# %%
# Use BERTviz to visualize the attention
from bertviz import model_view

# convert attention into (batch_size(must be 1), num_heads, sequence_length, sequence_length)
model_view(
    attention=original_words_unstacked_attentions,
    tokens=words,
)
# %%
from attention.max_attention_weights import max_attention_weights, heads_matching_relation

max_weights = max_attention_weights(original_words_stacked_attention_matrix)
heads_matching_rel = heads_matching_relation(conll_pd, max_weights)
# %%
