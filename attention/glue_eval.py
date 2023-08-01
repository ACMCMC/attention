# %%
# Evaluate the model on the GLUE dataset
import datasets
from attention.conll import parse_to_conllu
from attention.model_process import get_attention_matrix
from attention.max_attention_weights import max_attention_weights, heads_matching_relation

glue_dataset = datasets.load_dataset("glue", "cola", split="test")
# %%
def get_matching_heads_sentence(example):
    #print(f"Sentence: {example['sentence']}")
    conll_pd = parse_to_conllu(example['sentence'])
    # Dependencies: a list of the form (dependent_word, head_word, relation)
    dependencies = [(row.name, conll_pd.loc[row['HEAD']].name, row['DEPREL']) for _, row in conll_pd.iterrows() if row['HEAD'] > -1]
    dependencies_head_and_dependant = [(dependent_word, head_word) for (dependent_word, head_word, _) in dependencies]
    dependencies_reltype = [relation for (_, _, relation) in dependencies]
    # Take all the words in the sentence and get the heads matching the relation
    attention_matrix = get_attention_matrix(conll_pd)
    max_weights = max_attention_weights(attention_matrix)
    heads_matching_rel = heads_matching_relation(conll_pd, max_weights)
    # The result is a list of tuples (layer, head, head_word, dependent_word)
    # Only keep the tuples that have a dependent_word matching the relation
    # For every tuple, get the dependent_word and check if it matches the relation. The dependent_word is in position 3, and it is a string, so we need to get the row with that word in the column FORM and check its DEPREL
    heads_matching_rel = [(*t, conll_pd[conll_pd['FORM'] == t[2]]['DEPREL'].values[0]) for t in heads_matching_rel]
    matching_heads_layer_and_head = [(layer, head) for (layer, head, _, _, dependency) in heads_matching_rel]
    matching_heads_dependency = [dependency for (_, _, _, _, dependency) in heads_matching_rel]

    return {
        'matching_heads_layer_and_head': matching_heads_layer_and_head,
        'matching_heads_dependency': matching_heads_dependency,
        'dependencies_head_and_dependant': dependencies_head_and_dependant,
        'dependencies_reltype': dependencies_reltype,
    }

heads_matching_sentence = glue_dataset.map(get_matching_heads_sentence, batched=False)
# %%
# Get all unique values for the relation
# The relation is in the last position of the tuple 'dependencies'
# The result is a list of strings
relations = [dependency for example in heads_matching_sentence for dependency in example['dependencies_reltype']]
unique_relations = set(relations)

# Plot a 2D matrix with the heads matching each relation. The color of each cell is the number of heads matching that relation
import seaborn as sns
import matplotlib.pyplot as plt

# Create a matrix with the number of heads matching each relation
# The matrix is of size [num_layers, num_heads]
# Use the counter to fill the matrix
from attention.model_process import model
num_layers = model.config.num_hidden_layers
num_heads = model.config.num_attention_heads
for relation in unique_relations:
    matrix = [[0 for _ in range(num_heads)] for _ in range(num_layers)]
    total_words = 0
    for example in heads_matching_sentence:
        # Join matching_heads_layer_and_head and matching_heads_dependency to get the tuple
        for (layer, head), dependency in zip(example['matching_heads_layer_and_head'], example['matching_heads_dependency']):
            if dependency == relation:
                matrix[layer][head] += 1
        for (dependant, head), dependency in zip(example['dependencies_head_and_dependant'], example['dependencies_reltype']):
            if model.config.model_type == 'bloom':
                # For decoder-only models, the dependant has to be later than the head, otherwise the attention is not possible
                if dependant < head:
                    continue
            if dependency == relation:
                total_words += 1
    if total_words < 25:
        continue
    # Plot the matrix. The upper limit of the colorbar is the number of words in the dataset
    # Title: relation
    # X axis: head
    # Y axis: layer
    plt.title(f'Heads matching relation {relation}')
    plt.xlabel('Head')
    plt.ylabel('Layer')
    sns.heatmap(matrix, cmap="YlGnBu", vmin=0, vmax=total_words)
    # Store it as a PDF
    plt.savefig(f'heads_matching_{relation}_{model.config.model_type}.pdf')
    plt.show()
# %%
