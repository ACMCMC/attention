# %%
import os
from pathlib import Path

import torch
import transformers

import mlflow

from .dataset_eval import eval_ud

MODELS = [
    "bert-base-uncased",
    "microsoft/mdeberta-v3-base",
    "google-bert/bert-base-multilingual-cased",
    "bigscience/bloom-560m",
    "microsoft/Phi-3-mini-4k-instruct",
    "mistralai/Mistral-7B-v0.1",
    "meta-llama/Meta-Llama-3.1-8B",
    "google/gemma-2-9b",
    "HiTZ/latxa-7b-v1.2",
    "PlanTL-GOB-ES/roberta-base-bne",
    "fpuentes/bert-galician",
    "projecte-aina/FLOR-1.3B",
    "proxectonos/Carballo-bloom-1.3B",
]

data_en = {
    "language": "en",
    "path_to_conll_dataset": Path(__file__).parent.parent
    / "UD_English-GUM"
    / "en_gum-ud-test.conllu",
}
data_es = {
    "language": "es",
    "path_to_conll_dataset": Path(__file__).parent.parent
    / "UD_Spanish-AnCora"
    / "es_ancora-ud-test.conllu",
}
data_gl = {
    "language": "gl",
    "path_to_conll_dataset": Path(__file__).parent.parent
    / "UD_Galician-TreeGal"
    / "gl_treegal-ud-test.conllu",
}
data_fr = {
    "language": "fr",
    "path_to_conll_dataset": Path(__file__).parent.parent
    / "UD_French-Sequoia"
    / "fr_sequoia-ud-test.conllu",
}
data_tr = {
    "language": "tr",
    "path_to_conll_dataset": Path(__file__).parent.parent
    / "UD_Turkish-Penn-master"
    / "tr_penn-ud-test.conllu",
}
data_eu = {
    "language": "eu",
    "path_to_conll_dataset": Path(__file__).parent.parent
    / "UD_Basque-BDT-master"
    / "eu_bdt-ud-test.conllu",
}
data_ca = {
    "language": "ca",
    "path_to_conll_dataset": Path(__file__).parent.parent
    / "UD_Catalan-AnCora-master"
    / "ca_ancora-ud-test.conllu",
}


for model_name in MODELS:
    loaded_model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    if torch.cuda.is_available():
        loaded_model.to("cuda")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    for data in [data_en, data_es, data_gl, data_fr, data_tr, data_eu, data_ca]:
        output_dir = Path(__file__).parent.parent / f"results_{data['language']}"

        eval_ud(
            model=loaded_model,
            tokenizer=tokenizer,
            path_to_conll_dataset=data["path_to_conll_dataset"],
            output_dir=output_dir,
        )

    del loaded_model
    del tokenizer
