# %%
# Load the experiment config from "experiment_config.yaml"
import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import mlflow
import torch
import transformers
import yaml
from dotenv import load_dotenv
from huggingface_hub import HfApi

from attention.conll import (
    get_all_possible_conll_phrases,
    filter_out_null_head_examples,
)

from .dataset_eval import eval_ud

load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--experiment_config_path",
    default=Path(__file__).parent.parent / "experiment_config.yaml",
    type=Path,
)
parser.add_argument(
    "--accept_bidirectional_relations",
    action="store_true",
    help="If True, we also consider a 'hit' the fact that HEAD attends to DEPENDANT",
)
parser.add_argument(
    "--group_relations_by_family",
    action="store_true",
    help="If True, we group relations by family",
)
parser.add_argument(
    "--remove_self_attention",
    action="store_true",
    help="If True, we remove self-attention from the attention matrix by setting the diagonal to 0",
)
parser.add_argument(
    "--use_soft_scores",
    action="store_true",
    help="If True, we calculate the scores by taking the attention weights instead of a boolean variable indicating whether the max attention weight is attributed to a specific layer.",
)
parser.add_argument(
    # An argument that lets the user specify a list of languages to evaluate. If the user specifies this argument, only the languages in this list will be evaluated. Otherwise, all languages will be evaluated.
    "--languages",
    nargs="+",
    help="List of languages to evaluate",
)
args = parser.parse_args()

with open(args.experiment_config_path, "r") as f:
    experiment_config = yaml.safe_load(f)

mlflow.set_experiment(experiment_config["experiment_name"])

trim_dataset_size = experiment_config.get("trim_dataset_size", None)

# See the definitions of family groups in the function `group_relations_by_family` in `dataset_eval.py`
# If this is passed as a CLI argument, use that instead of the one in the config file
if args.group_relations_by_family is not None:
    logger.info(
        f"Group relations by family parameter passed as CLI argument: {args.group_relations_by_family}"
    )
    group_relations_by_family = args.group_relations_by_family
else:
    logger.info(
        f"Group relations by family parameter from config file: {experiment_config.get('group_relations_by_family', False)}"
    )
    group_relations_by_family = experiment_config.get(
        "group_relations_by_family", False
    )

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
# If this is passed as a CLI argument, use that instead of the one in the config file
if args.accept_bidirectional_relations is not None:
    logger.info(
        f"Accepting bidirectional relations parameter passed as CLI argument: {args.accept_bidirectional_relations}"
    )
    accept_bidirectional_relations = args.accept_bidirectional_relations
else:
    logger.info(
        f"Accepting bidirectional relations parameter from config file: {experiment_config.get('accept_bidirectional_relations', False)}"
    )
    accept_bidirectional_relations = experiment_config.get(
        "accept_bidirectional_relations", False
    )

api = HfApi()

# Sometimes, there's very unfrequent relations, so if there's less than this number of relations present in the analyzed data, we don't output a diagram on that
min_words_matching_relation = experiment_config.get("min_words_matching_relation", 1)


for language, metadata in experiment_config["languages"].items():
    # If the list of languages to evaluate is passed as a CLI argument, only evaluate those languages
    if args.languages is not None and language not in args.languages:
        logger.warning(
            f"Skipping evaluation for {language} because it's not in the list of languages passed as a CLI argument"
        )
        continue
    elif args.languages is not None and language in args.languages:
        logger.info(
            f"Evaluating {language} because it's in the list of languages passed as a CLI argument"
        )
    elif args.languages is None:
        logger.info(
            f"Evaluating {language} because no list of languages was passed as a CLI argument"
        )

    with mlflow.start_run(run_name=language) as mlrun:
        logger.info(f"Running evaluation for {language}...")
        models_to_evaluate = metadata["models"] + experiment_config.get(
            "multilingual_models", []
        )

        path_to_conll_dataset: Path = (
            Path(__file__).parent.parent / metadata["path_to_conll_dataset"]
        )
        mlflow.set_tag("dataset", path_to_conll_dataset.parent.name)

        conll_phrases: List[List[Dict[str, Any]]] = get_all_possible_conll_phrases(
            path_to_conll_dataset
        )

        # Filter out examples where the HEAD is null
        conll_phrases = filter_out_null_head_examples(conll_phrases)

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
                        "min_words_matching_relation", min_words_matching_relation
                    )
                    mlflow.log_param("trim_dataset_size", trim_dataset_size)
                    mlflow.log_param(
                        "group_relations_by_family", group_relations_by_family
                    )
                    mlflow.log_param(
                        "remove_self_attention", args.remove_self_attention
                    )
                    mlflow.log_param(
                        "use_soft_scores", args.use_soft_scores
                    )

                    output_dir = (
                        Path(__file__).parent.parent
                        / f"results_{language}"
                        / f"bidirectional_relations_{accept_bidirectional_relations}+group_relations_by_family_{group_relations_by_family}+remove_self_attention_{args.remove_self_attention}+soft_scores_{args.use_soft_scores}"
                        / model_name
                    )
                    # If the output directory exists and is not empty, skip this model
                    if output_dir.exists() and len(list(output_dir.iterdir())) > 0:
                        logger.info(
                            f"Output directory {output_dir} already exists and is not empty. Skipping..."
                        )
                        continue

                    logger.info(f"Loading model {model_uri}...")
                    # Which one to use: AutoModelForMaskedLM or AutoModelForCausalLM?
                    # We can use the HFApi to get the model's metadata and if it's a decoder model, we use AutoModelForMaskedLM, otherwise, AutoModelForCausalLM
                    # model_metadata = api.model_info(model_uri)

                    # For now, we'll just use AutoModel - this will work for both encoder and decoder models
                    loaded_model = transformers.AutoModel.from_pretrained(
                        model_uri, trust_remote_code=True, attn_implementation="eager"
                    )
                    if torch.cuda.is_available():
                        loaded_model.to("cuda")
                    tokenizer = transformers.AutoTokenizer.from_pretrained(
                        model_uri, trust_remote_code=True
                    )

                    eval_ud(
                        model=loaded_model,
                        tokenizer=tokenizer,
                        output_dir=output_dir,
                        accept_bidirectional_relations=accept_bidirectional_relations,
                        min_words_matching_relation=min_words_matching_relation,
                        trim_dataset_size=trim_dataset_size,
                        group_relations_by_family=group_relations_by_family,
                        remove_self_attention=args.remove_self_attention,
                        use_soft_scores=args.use_soft_scores,
                        conll_phrases=conll_phrases,
                        # trim_dataset_size=10,
                    )

                    logger.info(
                        f"CUDA memory usage before deleting model: reserved={torch.cuda.memory_reserved(0)}, allocated={torch.cuda.memory_allocated(0)}"
                    )
                    loaded_model.cpu()
                    del loaded_model
                    del tokenizer

                    import gc

                    gc.collect()
                    torch.cuda.empty_cache()
                    logger.info(
                        f"CUDA memory usage after deleting model: reserved={torch.cuda.memory_reserved(0)}, allocated={torch.cuda.memory_allocated(0)}"
                    )
                except Exception as e:
                    logger.exception(f"Error while evaluating {model_uri}")
                    mlflow.log_param("error", str(e))
                    mlflow.end_run(status="FAILED")
