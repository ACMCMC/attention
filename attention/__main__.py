# %%
import os

import transformers

from .dataset_eval import eval_ud

MODEL = "t5-base"
MODEL = "bigscience/bloom-560m"
MODEL = "bert-base-uncased"

model = transformers.AutoModelForMaskedLM.from_pretrained(MODEL)

path_to_conll_dataset = os.path.join(
    os.path.dirname(__file__), "../UD_English-GUM/en_gum-ud-test.conllu"
)

eval_ud(model=model, path_to_conll_dataset=path_to_conll_dataset)
