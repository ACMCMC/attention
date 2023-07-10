# %%
# Load model directly
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
model = AutoModel.from_pretrained("bigscience/bloom-560m")
# %%

phrase = "I eat pizza"
input_ids = tokenizer(phrase, return_tensors="pt", add_special_tokens=True)
print(input_ids)

# Get the attention values from the model
model.eval()
with torch.no_grad():
    outputs = model(input_ids['input_ids'], output_attentions=True)

print(outputs['attentions'])
# Returns a tuple with the attention tensors, one for each layer
# The size of each tensor is [batch_size, num_heads, sequence_length, sequence_length]
# Join them in a single tensor, of size [batch_size, num_layers, num_heads, sequence_length, sequence_length]
stacked_attentions = torch.stack(outputs['attentions'], dim=1)
print(stacked_attentions.shape)
# %%
