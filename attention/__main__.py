# %%
# Load the experiment config from "experiment_config.yaml"
import argparse
import logging
import os
from pathlib import Path

import mlflow
import torch
import transformers
import yaml
from huggingface_hub import HfApi

from .dataset_eval import eval_ud

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--experiment_config_path",
    default=Path(__file__).parent.parent / "experiment_config.yaml",
    type=Path,
)
args = parser.parse_args()

with open(args.experiment_config_path, "r") as f:
    experiment_config = yaml.safe_load(f)

mlflow.set_experiment(experiment_config["experiment_name"])

trim_dataset_size = experiment_config.get("trim_dataset_size", None)

# See the definitions of family groups in the function `group_relations_by_family` in `dataset_eval.py`
group_relations_by_family = experiment_config.get("group_relations_by_family", False)

# In decoder models, relations like this:
# "I am human", where DEPENDANT=human and HEAD=am
# can be modeled.
# However, these models can't "attend to" the future.
# So it would be impossible to obserb the following attention pattern:
# "the green house", where DEPENDANT=green and HEAD=house
# This is because the head comes after the dependant.
# i.e. to obserb the relation, the head needs to come before
# If this parameter is enabled, we also consider a "hit" the fact that HEAD attends to DEPENDANT
# This is in line with current literature.
accept_bidirectional_relations = experiment_config.get(
    "accept_bidirectional_relations", False
)

api = HfApi()

# Sometimes, there's very unfrequent relations, so if there's less than this number of relations present in the analyzed data, we don't output a diagram on that
MIN_WORDS_MATCHING_RELATION = 25


for language, metadata in experiment_config["languages"].items():
    with mlflow.start_run(run_name=language) as mlrun:
        logger.info(f"Running evaluation for {language}...")
        models_to_evaluate = metadata["models"] + experiment_config.get(
            "multilingual_models", []
        )

        for model_uri in models_to_evaluate:
            with mlflow.start_run(
                run_name=f"{language}_{model_uri}",
                nested=True,
                tags={"language": language},
            ) as mlrun2:
                try:
                    model_name = model_uri.split("/")[-1]
                    # If this model is not registered in MLFlow, register it
                    if (
                        len(
                            mlflow.search_registered_models(
                                filter_string=f'name="{model_name}"'
                            )
                        )
                        == 0
                    ):
                        logger.info(f"Registering model {model_uri}...")
                        mlflow.register_model(
                            model_uri=f"transformers://{model_uri}", name=model_name
                        )

                    mlflow.set_tag("model_uri", model_uri)
                    mlflow.log_param("metadata", metadata)
                    mlflow.log_param(
                        "accept_bidirectional_relations", accept_bidirectional_relations
                    )
                    mlflow.log_param(
                        "min_words_matching_relation", MIN_WORDS_MATCHING_RELATION
                    )
                    mlflow.log_param("trim_dataset_size", trim_dataset_size)
                    mlflow.log_param(
                        "group_relations_by_family", group_relations_by_family
                    )

                    logger.info(f"Loading model {model_uri}...")
                    # Which one to use: AutoModelForMaskedLM or AutoModelForCausalLM?
                    # We can use the HFApi to get the model's metadata and if it's a decoder model, we use AutoModelForMaskedLM, otherwise, AutoModelForCausalLM
                    # model_metadata = api.model_info(model_uri)

                    # For now, we'll just use AutoModel - this will work for both encoder and decoder models
                    loaded_model = transformers.AutoModel.from_pretrained(
                        model_uri, trust_remote_code=True
                    )
                    if torch.cuda.is_available():
                        loaded_model.to("cuda")
                    tokenizer = transformers.AutoTokenizer.from_pretrained(
                        model_uri, trust_remote_code=True
                    )

                    output_dir = (
                        Path(__file__).parent.parent
                        / f"results_{language}"
                        / model_name
                    )
                    path_to_conll_dataset: Path = (
                        Path(__file__).parent.parent / metadata["path_to_conll_dataset"]
                    )
                    mlflow.set_tag("dataset", path_to_conll_dataset.parent.name)

                    eval_ud(
                        model=loaded_model,
                        tokenizer=tokenizer,
                        path_to_conll_dataset=path_to_conll_dataset,
                        output_dir=output_dir,
                        accept_bidirectional_relations=accept_bidirectional_relations,
                        min_words_matching_relation=MIN_WORDS_MATCHING_RELATION,
                        trim_dataset_size=trim_dataset_size,
                        group_relations_by_family=group_relations_by_family,
                        # trim_dataset_size=10,
                    )

                    del loaded_model
                    del tokenizer
                except Exception as e:
                    logger.error(f"Error while evaluating {model_uri}: {e}")
                    mlflow.log_param("error", str(e))
                    mlflow.end_run(status="FAILED")
