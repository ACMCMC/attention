# %%
import os
from pathlib import Path

import torch
import transformers

from .dataset_eval import eval_ud

MODEL = "t5-base"
MODEL = "bert-base-uncased"
MODEL = "bigscience/bloom-560m"
MODEL = "HiTZ/latxa-7b-v1"

model = transformers.AutoModelForCausalLM.from_pretrained(MODEL)
if torch.cuda.is_available():
    model.to("cuda")
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)

data_en = {
    "path_to_conll_dataset": Path(__file__).parent.parent
    / "UD_English-GUM"
    / "en_gum-ud-test.conllu",
    "output_dir": Path(__file__).parent.parent / "results_en",
}
data_es = {
    "path_to_conll_dataset": Path(__file__).parent.parent
    / "UD_Spanish-AnCora"
    / "es_ancora-ud-test.conllu",
    "output_dir": Path(__file__).parent.parent / "results_es",
}
data_gl = {
    "path_to_conll_dataset": Path(__file__).parent.parent
    / "UD_Galician-TreeGal"
    / "gl_treegal-ud-test.conllu",
    "output_dir": Path(__file__).parent.parent / "results_gl",
}
data_fr = {
    "path_to_conll_dataset": Path(__file__).parent.parent
    / "UD_French-Sequoia"
    / "fr_sequoia-ud-test.conllu",
    "output_dir": Path(__file__).parent.parent / "results_fr",
}
data_tr = {
    "path_to_conll_dataset": Path(__file__).parent.parent
    / "UD_Turkish-Penn-master"
    / "tr_penn-ud-test.conllu",
    "output_dir": Path(__file__).parent.parent / "results_tr",
}
data_eu = {
    "path_to_conll_dataset": Path(__file__).parent.parent
    / "UD_Basque-BDT-master"
    / "eu_bdt-ud-test.conllu",
    "output_dir": Path(__file__).parent.parent / "results_eu",
}

for data in [data_en, data_es, data_gl, data_fr, data_tr, data_eu]:
    eval_ud(
        model=model,
        tokenizer=tokenizer,
        path_to_conll_dataset=data["path_to_conll_dataset"],
        output_dir=data["output_dir"],
    )
