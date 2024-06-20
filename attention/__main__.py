# %%
import os

import transformers

from .dataset_eval import eval_ud

MODEL = "t5-base"
MODEL = "bert-base-uncased"
MODEL = "bigscience/bloom-560m"

model = transformers.AutoModelForCausalLM.from_pretrained(MODEL)

path_to_conll_dataset = os.path.join(
    os.path.dirname(__file__), "../UD_English-GUM/en_gum-ud-test.conllu"
)
path_to_conll_dataset = os.path.join(
    os.path.dirname(__file__), "../UD_Spanish-AnCora/es_ancora-ud-test.conllu"
)

eval_ud(model=model, path_to_conll_dataset=path_to_conll_dataset)
