import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import glob
import requests
from io import StringIO
import zipfile
import tempfile
import shutil
import time
from datetime import datetime, timezone
import yaml

# Set page config
st.set_page_config(
    page_title="Attention Analysis Results Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stSelectbox > div > div {
        background-color: white;
    }
</style>
""",
    unsafe_allow_html=True,
)


class AttentionResultsExplorer:
    def __init__(self, github_repo="ACMCMC/attention", use_cache=True):
        self.github_repo = github_repo
        self.use_cache = use_cache
        self.cache_dir = Path(tempfile.gettempdir()) / "attention_results_cache"
        self.base_path = self.cache_dir

        # Initialize cache directory
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Get available languages from GitHub without downloading
        self.available_languages = self._get_available_languages_from_github()
        self.relation_types = None

    def _download_experiment_config(self):
        """Download and parse the experiment_config.yaml file from GitHub"""
        config_path = self.cache_dir / "experiment_config.yaml"

        # Check if we have a cached version and use_cache is enabled
        if config_path.exists() and self.use_cache:
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)
            except Exception as e:
                st.warning(f"Could not load cached config, downloading fresh: {str(e)}")

        # Download from GitHub
        config_url = f"https://raw.githubusercontent.com/{self.github_repo}/master/experiment_config.yaml"
        response = self._make_github_request(
            config_url, "experiment configuration file"
        )

        if response is None:
            # Try to load from cache as fallback
            if config_path.exists():
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        return yaml.safe_load(f)
                except Exception:
                    pass
            return None

        try:
            config_content = response.text
            # Save to cache
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(config_content)

            # Parse and return
            return yaml.safe_load(StringIO(config_content))

        except Exception as e:
            st.error(f"Could not parse experiment configuration: {str(e)}")
            return None

    def _get_available_languages_from_github(self):
        """Get available languages from experiment config file"""
        config = self._download_experiment_config()

        if config is None:
            # Fallback to directory-based discovery
            return self._get_available_languages_from_directories()

        try:
            languages = list(config.get("languages", {}).keys())
            return sorted(languages)
        except Exception as e:
            st.warning(f"Could not parse languages from config: {str(e)}")
            # Fallback to directory-based discovery
            return self._get_available_languages_from_directories()

    def _get_available_languages_from_directories(self):
        """Fallback method: Get available languages from GitHub API directory listing"""
        api_url = f"https://api.github.com/repos/{self.github_repo}/contents"

        response = self._make_github_request(api_url, "available languages")
        if response is None:
            # Rate limit hit or other error, fallback to local cache
            return self._get_available_languages_local()

        try:
            contents = response.json()
            result_dirs = [
                item["name"]
                for item in contents
                if item["type"] == "dir" and item["name"].startswith("results_")
            ]

            languages = [d.replace("results_", "") for d in result_dirs]
            return sorted(languages)

        except Exception as e:
            st.warning(f"Could not parse language list from GitHub: {str(e)}")
            # Fallback to local cache if available
            return self._get_available_languages_local()

    def _get_models_for_language(self, language):
        """Get all models for a specific language from the experiment config"""
        config = self._download_experiment_config()

        if config is None:
            return []

        try:
            # Get language-specific models
            language_models = (
                config.get("languages", {}).get(language, {}).get("models", [])
            )

            # Get multilingual models
            multilingual_models = config.get("multilingual_models", [])

            # Combine both lists
            all_models = language_models + multilingual_models
            return sorted(list(set(all_models)))  # Remove duplicates and sort

        except Exception as e:
            st.warning(f"Could not get models for {language}: {str(e)}")
            return []

    def _get_first_language_model_pair(self):
        """Get the first language-model pair from the experiment config for configuration discovery"""
        config = self._download_experiment_config()

        if config is None:
            return None, None

        try:
            languages = config.get("languages", {})
            multilingual_models = config.get("multilingual_models", [])

            # Find first language with models
            for language, lang_config in languages.items():
                models = lang_config.get("models", [])
                if models:
                    return language, models[0]

            # If no language-specific models, use first language with first multilingual model
            if multilingual_models and languages:
                first_language = list(languages.keys())[0]
                return first_language, multilingual_models[0]

            return None, None

        except Exception as e:
            st.warning(f"Could not find language-model pair: {str(e)}")
            return None, None

    def _get_available_languages_local(self):
        """Get available languages from local cache"""
        if not self.base_path.exists():
            return []
        result_dirs = [
            d.name
            for d in self.base_path.iterdir()
            if d.is_dir() and d.name.startswith("results_")
        ]
        languages = [d.replace("results_", "") for d in result_dirs]
        return sorted(languages)

    def _ensure_specific_data_downloaded(self, language, config, model):
        """Download specific files for a language/config/model combination if not cached"""
        folder_model_name = self._model_name_to_folder_name(model)
        base_path = f"results_{language}/{config}/{model}"
        local_path = self.base_path / f"results_{language}" / config / folder_model_name

        # Check if we already have this specific combination cached
        if local_path.exists() and self.use_cache:
            # Quick check if essential files exist
            metadata_path = local_path / "metadata" / "metadata.json"
            if metadata_path.exists():
                return  # Already have the data

        with st.spinner(
            f"📥 Downloading data for {language.upper()}/{config}/{model}..."
        ):
            try:
                self._download_specific_model_data(language, config, model)
                st.success(f"✅ Downloaded {language.upper()}/{model} data!")
            except Exception as e:
                st.error(f"❌ Failed to download specific data: {str(e)}")
                raise

    def _download_specific_model_data(self, language, config, model):
        """Download only the specific model data needed"""
        folder_model_name = self._model_name_to_folder_name(model)
        base_remote_path = f"results_{language}/{config}/{folder_model_name}"

        # List of essential directories to download for a model
        essential_dirs = [
            "metadata",
            "uas_scores",
            "number_of_heads_matching",
            "variability",
            "figures",
        ]

        for dir_name in essential_dirs:
            remote_path = f"{base_remote_path}/{dir_name}"
            try:
                self._download_directory_targeted(
                    dir_name, remote_path, language, config, model
                )
            except Exception as e:
                st.warning(f"Could not download {dir_name} for {model}: {str(e)}")

    def _download_directory_targeted(
        self, dir_name, remote_path, language, config, model
    ):
        """Download a specific directory for a model"""
        api_url = (
            f"https://api.github.com/repos/{self.github_repo}/contents/{remote_path}"
        )

        response = self._make_github_request(
            api_url, f"directory {dir_name}", silent_404=True
        )
        if response is None:
            return  # Rate limit, 404, or other error

        try:
            contents = response.json()

            # Create local directory
            folder_model_name = self._model_name_to_folder_name(model)
            local_dir = (
                self.base_path
                / f"results_{language}"
                / config
                / folder_model_name
                / dir_name
            )
            local_dir.mkdir(parents=True, exist_ok=True)

            # Download all files in this directory
            for item in contents:
                if item["type"] == "file":
                    self._download_file(item, local_dir)

        except Exception as e:
            st.warning(f"Could not download directory {dir_name}: {str(e)}")

    def _get_available_configs_from_github(self, language):
        """Get available configurations for a language from GitHub"""
        api_url = f"https://api.github.com/repos/{self.github_repo}/contents/results_{language}"

        response = self._make_github_request(api_url, f"configurations for {language}")
        if response is None:
            return []

        try:
            contents = response.json()
            configs = [item["name"] for item in contents if item["type"] == "dir"]
            return sorted(configs)

        except Exception as e:
            st.warning(f"Could not parse configurations for {language}: {str(e)}")
            return []

    def _discover_config_parameters(self, language=None):
        """Dynamically discover configuration parameters from available configs

        Now uses the first language-model pair from experiment config to discover
        valid configuration parameters, since configurations are consistent across
        all language-model combinations.
        """
        try:
            # Get the first language-model pair from experiment config
            if language is None:
                language, model = self._get_first_language_model_pair()
                if language is None or model is None:
                    st.warning(
                        "Could not find any language-model pairs in experiment config"
                    )
                    return {}
                st.info(
                    f"🔍 Discovering configurations using {language.upper()}/{model} (configurations are consistent across all languages and models)"
                )
            else:
                # If language is specified, try to get first model for that language
                models = self._get_models_for_language(language)
                if not models:
                    st.warning(f"No models found for language {language}")
                    return {}
                model = models[0]

            available_configs = self._get_experimental_configs(language)
            if not available_configs:
                return {}

            # Parse all configurations to extract unique parameters
            all_params = set()
            param_values = {}

            for config in available_configs:
                params = self._parse_config_params(config)
                for param, value in params.items():
                    all_params.add(param)
                    if param not in param_values:
                        param_values[param] = set()
                    param_values[param].add(value)

            # Convert sets to sorted lists for consistent UI
            return {
                param: sorted(list(values)) for param, values in param_values.items()
            }

        except Exception as e:
            st.warning(f"Could not discover configuration parameters: {str(e)}")
            return {}

    def _build_config_from_params(self, param_dict):
        """Build configuration string from parameter dictionary"""
        config_parts = []
        for param, value in sorted(param_dict.items()):
            config_parts.append(f"{param}_{value}")
        return "+".join(config_parts)

    def _find_best_matching_config(self, language, target_params):
        """Find the configuration that best matches the target parameters"""
        available_configs = self._get_experimental_configs(language)

        best_match = None
        best_score = -1

        for config in available_configs:
            config_params = self._parse_config_params(config)

            # Calculate match score
            score = 0
            total_params = len(target_params)

            for param, target_value in target_params.items():
                if param in config_params and config_params[param] == target_value:
                    score += 1

            # Prefer configs with exact parameter count
            if len(config_params) == total_params:
                score += 0.5

            if score > best_score:
                best_score = score
                best_match = config

        return best_match, best_score == len(target_params)

    def _download_repository(self):
        """Download repository data from GitHub"""
        st.info("🔄 Downloading results data from GitHub... This may take a moment.")

        # GitHub API to get the repository contents
        api_url = f"https://api.github.com/repos/{self.github_repo}/contents"

        try:
            # Get list of result directories
            response = requests.get(api_url)
            response.raise_for_status()
            contents = response.json()

            result_dirs = [
                item["name"]
                for item in contents
                if item["type"] == "dir" and item["name"].startswith("results_")
            ]

            st.write(
                f"Found {len(result_dirs)} result directories: {', '.join(result_dirs)}"
            )

            # Download each result directory
            progress_bar = st.progress(0)
            for i, result_dir in enumerate(result_dirs):
                st.write(f"Downloading {result_dir}...")
                self._download_directory(result_dir)
                progress_bar.progress((i + 1) / len(result_dirs))

            st.success("✅ Download completed!")

        except Exception as e:
            st.error(f"❌ Error downloading repository: {str(e)}")
            st.error("Please check the repository URL and your internet connection.")
            raise

    def _parse_config_params(self, config_name):
        """Parse configuration parameters into a dictionary"""
        parts = config_name.split("+")
        params = {}
        for part in parts:
            if "_" in part:
                key_parts = part.split("_")
                if len(key_parts) >= 2:
                    key = "_".join(key_parts[:-1])
                    value = key_parts[-1]
                    params[key] = value == "True"
        return params

    def _download_directory(self, dir_name, path=""):
        """Recursively download a directory from GitHub"""
        url = (
            f"https://api.github.com/repos/{self.github_repo}/contents/{path}{dir_name}"
        )

        try:
            response = requests.get(url)
            response.raise_for_status()
            contents = response.json()

            local_dir = self.cache_dir / path / dir_name
            local_dir.mkdir(parents=True, exist_ok=True)

            for item in contents:
                if item["type"] == "file":
                    self._download_file(item, local_dir)
                elif item["type"] == "dir":
                    self._download_directory(item["name"], f"{path}{dir_name}/")

        except Exception as e:
            st.warning(f"Could not download {dir_name}: {str(e)}")

    def _download_file(self, file_info, local_dir):
        """Download a single file from GitHub"""
        try:
            # Use the rate limit handling for file downloads too
            file_response = self._make_github_request(
                file_info["download_url"], f"file {file_info['name']}"
            )
            if file_response is None:
                return  # Rate limit or other error

            # Save to local cache
            local_file = local_dir / file_info["name"]

            # Handle different file types
            if file_info["name"].endswith((".csv", ".json")):
                with open(local_file, "w", encoding="utf-8") as f:
                    f.write(file_response.text)
            else:  # Binary files like PDFs
                with open(local_file, "wb") as f:
                    f.write(file_response.content)

        except Exception as e:
            st.warning(f"Could not download file {file_info['name']}: {str(e)}")

    def _get_available_languages(self):
        """Get all available language directories"""
        return self.available_languages

    def _get_experimental_configs(self, language):
        """Get all experimental configurations for a language from GitHub API"""
        api_url = f"https://api.github.com/repos/{self.github_repo}/contents/results_{language}"
        response = self._make_github_request(
            api_url, f"experimental configs for {language}"
        )

        if response is not None:
            try:
                contents = response.json()
                configs = [item["name"] for item in contents if item["type"] == "dir"]
                return sorted(configs)
            except Exception as e:
                st.warning(
                    f"Could not parse experimental configs for {language}: {str(e)}"
                )

        # Fallback to local cache if available
        lang_dir = self.base_path / f"results_{language}"
        if lang_dir.exists():
            configs = [d.name for d in lang_dir.iterdir() if d.is_dir()]
            return sorted(configs)
        return []

    def _find_matching_config(self, language, target_params):
        """Find the first matching configuration from target parameters"""
        return self._find_best_matching_config(language, target_params)

    def _get_models(self, language, config):
        """Get all models for a language and configuration from experiment config"""
        # First try to get models from experiment config
        models = self._get_models_for_language(language)

        if models:
            return models

        # Fallback to GitHub API directory listing if config unavailable
        api_url = f"https://api.github.com/repos/{self.github_repo}/contents/results_{language}/{config}"
        response = self._make_github_request(api_url, f"models for {language}/{config}")

        if response is not None:
            try:
                contents = response.json()
                models = [item["name"] for item in contents if item["type"] == "dir"]
                return sorted(models)
            except Exception as e:
                st.warning(f"Could not parse models for {language}/{config}: {str(e)}")

        # Final fallback to local cache if available
        config_dir = self.base_path / f"results_{language}" / config
        if config_dir.exists():
            models = [d.name for d in config_dir.iterdir() if d.is_dir()]
            return sorted(models)
        return []

    def _parse_config_name(self, config_name):
        """Parse configuration name into readable format"""
        parts = config_name.split("+")
        config_dict = {}
        for part in parts:
            if "_" in part:
                key, value = part.split("_", 1)
                config_dict[key.replace("_", " ").title()] = value
        return config_dict

    def _load_metadata(self, language, config, model):
        """Load metadata for a specific combination"""
        # Ensure we have the specific data downloaded
        self._ensure_specific_data_downloaded(language, config, model)

        folder_model_name = self._model_name_to_folder_name(model)
        metadata_path = (
            self.base_path
            / f"results_{language}"
            / config
            / folder_model_name
            / "metadata"
            / "metadata.json"
        )
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                return json.load(f)
        return None

    def _load_uas_scores(self, language, config, model):
        """Load UAS scores data"""
        # Ensure we have the specific data downloaded
        self._ensure_specific_data_downloaded(language, config, model)

        folder_model_name = self._model_name_to_folder_name(model)
        uas_dir = (
            self.base_path
            / f"results_{language}"
            / config
            / folder_model_name
            / "uas_scores"
        )
        if not uas_dir.exists():
            return {}

        uas_data = {}
        csv_files = list(uas_dir.glob("uas_*.csv"))

        if csv_files:
            with st.spinner("Loading UAS scores data..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, csv_file in enumerate(csv_files):
                    relation = csv_file.stem.replace("uas_", "")
                    status_text.text(f"Loading UAS data: {relation}")

                    try:
                        df = pd.read_csv(csv_file, index_col=0)
                        uas_data[relation] = df
                    except Exception as e:
                        st.warning(f"Could not load {csv_file.name}: {e}")

                    progress_bar.progress((i + 1) / len(csv_files))
                    time.sleep(0.01)  # Small delay for smoother progress

                progress_bar.empty()
                status_text.empty()

        return uas_data

    def _load_head_matching(self, language, config, model):
        """Load head matching data"""
        # Ensure we have the specific data downloaded
        self._ensure_specific_data_downloaded(language, config, model)

        folder_model_name = self._model_name_to_folder_name(model)
        heads_dir = (
            self.base_path
            / f"results_{language}"
            / config
            / folder_model_name
            / "number_of_heads_matching"
        )
        if not heads_dir.exists():
            return {}

        heads_data = {}
        csv_files = list(heads_dir.glob("heads_matching_*.csv"))

        if csv_files:
            with st.spinner("Loading head matching data..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, csv_file in enumerate(csv_files):
                    relation = csv_file.stem.replace("heads_matching_", "").replace(
                        f"_{folder_model_name}", ""
                    )
                    status_text.text(f"Loading head matching data: {relation}")

                    try:
                        df = pd.read_csv(csv_file, index_col=0)
                        heads_data[relation] = df
                    except Exception as e:
                        st.warning(f"Could not load {csv_file.name}: {e}")

                    progress_bar.progress((i + 1) / len(csv_files))
                    time.sleep(0.01)  # Small delay for smoother progress

                progress_bar.empty()
                status_text.empty()

        return heads_data

    def _load_variability(self, language, config, model):
        """Load variability data"""
        # Ensure we have the specific data downloaded
        self._ensure_specific_data_downloaded(language, config, model)

        folder_model_name = self._model_name_to_folder_name(model)
        var_path = (
            self.base_path
            / f"results_{language}"
            / config
            / folder_model_name
            / "variability"
            / "variability_list.csv"
        )
        if var_path.exists():
            try:
                return pd.read_csv(var_path, index_col=0)
            except Exception as e:
                st.warning(f"Could not load variability data: {e}")
        return None

    def _get_available_figures(self, language, config, model):
        """Get all available figure files"""
        # Ensure we have the specific data downloaded
        self._ensure_specific_data_downloaded(language, config, model)

        folder_model_name = self._model_name_to_folder_name(model)
        figures_dir = (
            self.base_path
            / f"results_{language}"
            / config
            / folder_model_name
            / "figures"
        )
        if not figures_dir.exists():
            return []
        return list(figures_dir.glob("*.pdf"))

    def _handle_rate_limit_error(self, response):
        """Handle GitHub API rate limit errors with detailed user feedback"""
        if response.status_code in (403, 429):
            # Check if it's a rate limit error
            if (
                "rate limit" in response.text.lower()
                or "api rate limit" in response.text.lower()
            ):
                # Extract rate limit information from headers
                remaining = response.headers.get("x-ratelimit-remaining", "unknown")
                reset_timestamp = response.headers.get("x-ratelimit-reset")
                limit = response.headers.get("x-ratelimit-limit", "unknown")

                # Calculate reset time
                reset_time_str = "unknown"
                if reset_timestamp:
                    try:
                        reset_time = datetime.fromtimestamp(
                            int(reset_timestamp), tz=timezone.utc
                        )
                        reset_time_str = reset_time.strftime("%Y-%m-%d %H:%M:%S UTC")

                        # Calculate time until reset
                        now = datetime.now(timezone.utc)
                        time_until_reset = reset_time - now
                        minutes_until_reset = int(time_until_reset.total_seconds() / 60)

                        if minutes_until_reset > 0:
                            reset_time_str += f" (in {minutes_until_reset} minutes)"
                    except (ValueError, TypeError):
                        pass

                # Display comprehensive rate limit information
                st.error("🚫 **GitHub API Rate Limit Exceeded**")

                with st.expander("📊 Rate Limit Details", expanded=True):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Requests Remaining", remaining)
                        st.metric("Rate Limit", limit)

                    with col2:
                        st.metric("Reset Time", reset_time_str)
                        if reset_timestamp:
                            try:
                                reset_time = datetime.fromtimestamp(
                                    int(reset_timestamp), tz=timezone.utc
                                )
                                now = datetime.now(timezone.utc)
                                time_until_reset = reset_time - now
                                if time_until_reset.total_seconds() > 0:
                                    st.metric(
                                        "Time Until Reset",
                                        f"{int(time_until_reset.total_seconds() / 60)} minutes",
                                    )
                            except (ValueError, TypeError):
                                pass

                return True  # Indicates rate limit error was handled

        return False  # Not a rate limit error

    def _make_github_request(
        self, url, description="GitHub API request", silent_404=False
    ):
        """Make a GitHub API request with rate limit handling"""
        try:
            # Add GitHub token if available
            headers = {}
            github_token = os.environ.get("GITHUB_TOKEN")
            if github_token:
                headers["Authorization"] = f"token {github_token}"

            response = requests.get(url, headers=headers)

            # Check for rate limit before raising for status
            if self._handle_rate_limit_error(response):
                return None  # Rate limit handled, return None

            # Handle 404 errors silently if requested (for optional directories)
            if response.status_code == 404 and silent_404:
                return None

            response.raise_for_status()
            return response

        except requests.exceptions.RequestException as e:
            if hasattr(e, "response") and e.response is not None:
                # Handle 404 silently if requested
                if e.response.status_code == 404 and silent_404:
                    return None

                if not self._handle_rate_limit_error(e.response):
                    st.warning(f"Request failed for {description}: {str(e)}")
            else:
                st.warning(f"Network error for {description}: {str(e)}")
            return None

    def _model_name_to_folder_name(self, model_name):
        """Convert model name from config format to folder format

        Examples:
        - 'PlanTL-GOB-ES/roberta-base-ca' -> 'roberta-base-ca'
        - 'microsoft/deberta-v3-base' -> 'deberta-v3-base'
        - 'bert-base-uncased' -> 'bert-base-uncased' (no change)
        """
        if "/" in model_name:
            return model_name.split("/")[-1]
        return model_name

    def _get_available_languages_local(self):
        """Get available languages from local cache"""
        if not self.base_path.exists():
            return []
        result_dirs = [
            d.name
            for d in self.base_path.iterdir()
            if d.is_dir() and d.name.startswith("results_")
        ]
        languages = [d.replace("results_", "") for d in result_dirs]
        return sorted(languages)

    def _ensure_specific_data_downloaded(self, language, config, model):
        """Download specific files for a language/config/model combination if not cached"""
        folder_model_name = self._model_name_to_folder_name(model)
        base_path = f"results_{language}/{config}/{model}"
        local_path = self.base_path / f"results_{language}" / config / folder_model_name

        # Check if we already have this specific combination cached
        if local_path.exists() and self.use_cache:
            # Quick check if essential files exist
            metadata_path = local_path / "metadata" / "metadata.json"
            if metadata_path.exists():
                return  # Already have the data

        with st.spinner(
            f"📥 Downloading data for {language.upper()}/{config}/{model}..."
        ):
            try:
                self._download_specific_model_data(language, config, model)
                st.success(f"✅ Downloaded {language.upper()}/{model} data!")
            except Exception as e:
                st.error(f"❌ Failed to download specific data: {str(e)}")
                raise

    def _download_specific_model_data(self, language, config, model):
        """Download only the specific model data needed"""
        folder_model_name = self._model_name_to_folder_name(model)
        base_remote_path = f"results_{language}/{config}/{folder_model_name}"

        # List of essential directories to download for a model
        essential_dirs = [
            "metadata",
            "uas_scores",
            "number_of_heads_matching",
            "variability",
            "figures",
        ]

        for dir_name in essential_dirs:
            remote_path = f"{base_remote_path}/{dir_name}"
            try:
                self._download_directory_targeted(
                    dir_name, remote_path, language, config, model
                )
            except Exception as e:
                st.warning(f"Could not download {dir_name} for {model}: {str(e)}")

    def _download_directory_targeted(
        self, dir_name, remote_path, language, config, model
    ):
        """Download a specific directory for a model"""
        api_url = (
            f"https://api.github.com/repos/{self.github_repo}/contents/{remote_path}"
        )

        response = self._make_github_request(
            api_url, f"directory {dir_name}", silent_404=True
        )
        if response is None:
            return  # Rate limit, 404, or other error

        try:
            contents = response.json()

            # Create local directory
            folder_model_name = self._model_name_to_folder_name(model)
            local_dir = (
                self.base_path
                / f"results_{language}"
                / config
                / folder_model_name
                / dir_name
            )
            local_dir.mkdir(parents=True, exist_ok=True)

            # Download all files in this directory
            for item in contents:
                if item["type"] == "file":
                    self._download_file(item, local_dir)

        except Exception as e:
            st.warning(f"Could not download directory {dir_name}: {str(e)}")

    def _get_available_configs_from_github(self, language):
        """Get available configurations for a language from GitHub"""
        api_url = f"https://api.github.com/repos/{self.github_repo}/contents/results_{language}"

        response = self._make_github_request(api_url, f"configurations for {language}")
        if response is None:
            return []

        try:
            contents = response.json()
            configs = [item["name"] for item in contents if item["type"] == "dir"]
            return sorted(configs)

        except Exception as e:
            st.warning(f"Could not parse configurations for {language}: {str(e)}")
            return []

    def _discover_config_parameters(self, language=None):
        """Dynamically discover configuration parameters from available configs

        Now uses the first language-model pair from experiment config to discover
        valid configuration parameters, since configurations are consistent across
        all language-model combinations.
        """
        try:
            # Get the first language-model pair from experiment config
            if language is None:
                language, model = self._get_first_language_model_pair()
                if language is None or model is None:
                    st.warning(
                        "Could not find any language-model pairs in experiment config"
                    )
                    return {}
                st.info(
                    f"🔍 Discovering configurations using {language.upper()}/{model} (configurations are consistent across all languages and models)"
                )
            else:
                # If language is specified, try to get first model for that language
                models = self._get_models_for_language(language)
                if not models:
                    st.warning(f"No models found for language {language}")
                    return {}
                model = models[0]

            available_configs = self._get_experimental_configs(language)
            if not available_configs:
                return {}

            # Parse all configurations to extract unique parameters
            all_params = set()
            param_values = {}

            for config in available_configs:
                params = self._parse_config_params(config)
                for param, value in params.items():
                    all_params.add(param)
                    if param not in param_values:
                        param_values[param] = set()
                    param_values[param].add(value)

            # Convert sets to sorted lists for consistent UI
            return {
                param: sorted(list(values)) for param, values in param_values.items()
            }

        except Exception as e:
            st.warning(f"Could not discover configuration parameters: {str(e)}")
            return {}

    def _build_config_from_params(self, param_dict):
        """Build configuration string from parameter dictionary"""
        config_parts = []
        for param, value in sorted(param_dict.items()):
            config_parts.append(f"{param}_{value}")
        return "+".join(config_parts)

    def _find_best_matching_config(self, language, target_params):
        """Find the configuration that best matches the target parameters"""
        available_configs = self._get_experimental_configs(language)

        best_match = None
        best_score = -1

        for config in available_configs:
            config_params = self._parse_config_params(config)

            # Calculate match score
            score = 0
            total_params = len(target_params)

            for param, target_value in target_params.items():
                if param in config_params and config_params[param] == target_value:
                    score += 1

            # Prefer configs with exact parameter count
            if len(config_params) == total_params:
                score += 0.5

            if score > best_score:
                best_score = score
                best_match = config

        return best_match, best_score == len(target_params)

    def _download_repository(self):
        """Download repository data from GitHub"""
        st.info("🔄 Downloading results data from GitHub... This may take a moment.")

        # GitHub API to get the repository contents
        api_url = f"https://api.github.com/repos/{self.github_repo}/contents"

        try:
            # Get list of result directories
            response = requests.get(api_url)
            response.raise_for_status()
            contents = response.json()

            result_dirs = [
                item["name"]
                for item in contents
                if item["type"] == "dir" and item["name"].startswith("results_")
            ]

            st.write(
                f"Found {len(result_dirs)} result directories: {', '.join(result_dirs)}"
            )

            # Download each result directory
            progress_bar = st.progress(0)
            for i, result_dir in enumerate(result_dirs):
                st.write(f"Downloading {result_dir}...")
                self._download_directory(result_dir)
                progress_bar.progress((i + 1) / len(result_dirs))

            st.success("✅ Download completed!")

        except Exception as e:
            st.error(f"❌ Error downloading repository: {str(e)}")
            st.error("Please check the repository URL and your internet connection.")
            raise

    def _parse_config_params(self, config_name):
        """Parse configuration parameters into a dictionary"""
        parts = config_name.split("+")
        params = {}
        for part in parts:
            if "_" in part:
                key_parts = part.split("_")
                if len(key_parts) >= 2:
                    key = "_".join(key_parts[:-1])
                    value = key_parts[-1]
                    params[key] = value == "True"
        return params

    def _download_directory(self, dir_name, path=""):
        """Recursively download a directory from GitHub"""
        url = (
            f"https://api.github.com/repos/{self.github_repo}/contents/{path}{dir_name}"
        )

        try:
            response = requests.get(url)
            response.raise_for_status()
            contents = response.json()

            local_dir = self.cache_dir / path / dir_name
            local_dir.mkdir(parents=True, exist_ok=True)

            for item in contents:
                if item["type"] == "file":
                    self._download_file(item, local_dir)
                elif item["type"] == "dir":
                    self._download_directory(item["name"], f"{path}{dir_name}/")

        except Exception as e:
            st.warning(f"Could not download {dir_name}: {str(e)}")

    def _download_file(self, file_info, local_dir):
        """Download a single file from GitHub"""
        try:
            # Use the rate limit handling for file downloads too
            file_response = self._make_github_request(
                file_info["download_url"], f"file {file_info['name']}"
            )
            if file_response is None:
                return  # Rate limit or other error

            # Save to local cache
            local_file = local_dir / file_info["name"]

            # Handle different file types
            if file_info["name"].endswith((".csv", ".json")):
                with open(local_file, "w", encoding="utf-8") as f:
                    f.write(file_response.text)
            else:  # Binary files like PDFs
                with open(local_file, "wb") as f:
                    f.write(file_response.content)

        except Exception as e:
            st.warning(f"Could not download file {file_info['name']}: {str(e)}")

    def _get_available_languages(self):
        """Get all available language directories"""
        return self.available_languages

    def _get_experimental_configs(self, language):
        """Get all experimental configurations for a language from GitHub API"""
        api_url = f"https://api.github.com/repos/{self.github_repo}/contents/results_{language}"
        response = self._make_github_request(
            api_url, f"experimental configs for {language}"
        )

        if response is not None:
            try:
                contents = response.json()
                configs = [item["name"] for item in contents if item["type"] == "dir"]
                return sorted(configs)
            except Exception as e:
                st.warning(
                    f"Could not parse experimental configs for {language}: {str(e)}"
                )

        # Fallback to local cache if available
        lang_dir = self.base_path / f"results_{language}"
        if lang_dir.exists():
            configs = [d.name for d in lang_dir.iterdir() if d.is_dir()]
            return sorted(configs)
        return []

    def _find_matching_config(self, language, target_params):
        """Find the first matching configuration from target parameters"""
        return self._find_best_matching_config(language, target_params)

    def _get_models(self, language, config):
        """Get all models for a language and configuration from experiment config"""
        # First try to get models from experiment config
        models = self._get_models_for_language(language)

        if models:
            return models

        # Fallback to GitHub API directory listing if config unavailable
        api_url = f"https://api.github.com/repos/{self.github_repo}/contents/results_{language}/{config}"
        response = self._make_github_request(api_url, f"models for {language}/{config}")

        if response is not None:
            try:
                contents = response.json()
                models = [item["name"] for item in contents if item["type"] == "dir"]
                return sorted(models)
            except Exception as e:
                st.warning(f"Could not parse models for {language}/{config}: {str(e)}")

        # Final fallback to local cache if available
        config_dir = self.base_path / f"results_{language}" / config
        if config_dir.exists():
            models = [d.name for d in config_dir.iterdir() if d.is_dir()]
            return sorted(models)
        return []

    def _parse_config_name(self, config_name):
        """Parse configuration name into readable format"""
        parts = config_name.split("+")
        config_dict = {}
        for part in parts:
            if "_" in part:
                key, value = part.split("_", 1)
                config_dict[key.replace("_", " ").title()] = value
        return config_dict

    def _load_metadata(self, language, config, model):
        """Load metadata for a specific combination"""
        # Ensure we have the specific data downloaded
        self._ensure_specific_data_downloaded(language, config, model)

        folder_model_name = self._model_name_to_folder_name(model)
        metadata_path = (
            self.base_path
            / f"results_{language}"
            / config
            / folder_model_name
            / "metadata"
            / "metadata.json"
        )
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                return json.load(f)
        return None

    def _load_uas_scores(self, language, config, model):
        """Load UAS scores data"""
        # Ensure we have the specific data downloaded
        self._ensure_specific_data_downloaded(language, config, model)

        folder_model_name = self._model_name_to_folder_name(model)
        uas_dir = (
            self.base_path
            / f"results_{language}"
            / config
            / folder_model_name
            / "uas_scores"
        )
        if not uas_dir.exists():
            return {}

        uas_data = {}
        csv_files = list(uas_dir.glob("uas_*.csv"))

        if csv_files:
            with st.spinner("Loading UAS scores data..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, csv_file in enumerate(csv_files):
                    relation = csv_file.stem.replace("uas_", "")
                    status_text.text(f"Loading UAS data: {relation}")

                    try:
                        df = pd.read_csv(csv_file, index_col=0)
                        uas_data[relation] = df
                    except Exception as e:
                        st.warning(f"Could not load {csv_file.name}: {e}")

                    progress_bar.progress((i + 1) / len(csv_files))
                    time.sleep(0.01)  # Small delay for smoother progress

                progress_bar.empty()
                status_text.empty()

        return uas_data

    def _load_head_matching(self, language, config, model):
        """Load head matching data"""
        # Ensure we have the specific data downloaded
        self._ensure_specific_data_downloaded(language, config, model)

        folder_model_name = self._model_name_to_folder_name(model)
        heads_dir = (
            self.base_path
            / f"results_{language}"
            / config
            / folder_model_name
            / "number_of_heads_matching"
        )
        if not heads_dir.exists():
            return {}

        heads_data = {}
        csv_files = list(heads_dir.glob("heads_matching_*.csv"))

        if csv_files:
            with st.spinner("Loading head matching data..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, csv_file in enumerate(csv_files):
                    relation = csv_file.stem.replace("heads_matching_", "").replace(
                        f"_{folder_model_name}", ""
                    )
                    status_text.text(f"Loading head matching data: {relation}")

                    try:
                        df = pd.read_csv(csv_file, index_col=0)
                        heads_data[relation] = df
                    except Exception as e:
                        st.warning(f"Could not load {csv_file.name}: {e}")

                    progress_bar.progress((i + 1) / len(csv_files))
                    time.sleep(0.01)  # Small delay for smoother progress

                progress_bar.empty()
                status_text.empty()

        return heads_data

    def _load_variability(self, language, config, model):
        """Load variability data"""
        # Ensure we have the specific data downloaded
        self._ensure_specific_data_downloaded(language, config, model)

        folder_model_name = self._model_name_to_folder_name(model)
        var_path = (
            self.base_path
            / f"results_{language}"
            / config
            / folder_model_name
            / "variability"
            / "variability_list.csv"
        )
        if var_path.exists():
            try:
                return pd.read_csv(var_path, index_col=0)
            except Exception as e:
                st.warning(f"Could not load variability data: {e}")
        return None

    def _get_available_figures(self, language, config, model):
        """Get all available figure files"""
        # Ensure we have the specific data downloaded
        self._ensure_specific_data_downloaded(language, config, model)

        folder_model_name = self._model_name_to_folder_name(model)
        figures_dir = (
            self.base_path
            / f"results_{language}"
            / config
            / folder_model_name
            / "figures"
        )
        if not figures_dir.exists():
            return []
        return list(figures_dir.glob("*.pdf"))

    def _handle_rate_limit_error(self, response):
        """Handle GitHub API rate limit errors with detailed user feedback"""
        if response.status_code in (403, 429):
            # Check if it's a rate limit error
            if (
                "rate limit" in response.text.lower()
                or "api rate limit" in response.text.lower()
            ):
                # Extract rate limit information from headers
                remaining = response.headers.get("x-ratelimit-remaining", "unknown")
                reset_timestamp = response.headers.get("x-ratelimit-reset")
                limit = response.headers.get("x-ratelimit-limit", "unknown")

                # Calculate reset time
                reset_time_str = "unknown"
                if reset_timestamp:
                    try:
                        reset_time = datetime.fromtimestamp(
                            int(reset_timestamp), tz=timezone.utc
                        )
                        reset_time_str = reset_time.strftime("%Y-%m-%d %H:%M:%S UTC")

                        # Calculate time until reset
                        now = datetime.now(timezone.utc)
                        time_until_reset = reset_time - now
                        minutes_until_reset = int(time_until_reset.total_seconds() / 60)

                        if minutes_until_reset > 0:
                            reset_time_str += f" (in {minutes_until_reset} minutes)"
                    except (ValueError, TypeError):
                        pass

                # Display comprehensive rate limit information
                st.error("🚫 **GitHub API Rate Limit Exceeded**")

                with st.expander("📊 Rate Limit Details", expanded=True):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Requests Remaining", remaining)
                        st.metric("Rate Limit", limit)

                    with col2:
                        st.metric("Reset Time", reset_time_str)
                        if reset_timestamp:
                            try:
                                reset_time = datetime.fromtimestamp(
                                    int(reset_timestamp), tz=timezone.utc
                                )
                                now = datetime.now(timezone.utc)
                                time_until_reset = reset_time - now
                                if time_until_reset.total_seconds() > 0:
                                    st.metric(
                                        "Time Until Reset",
                                        f"{int(time_until_reset.total_seconds() / 60)} minutes",
                                    )
                            except (ValueError, TypeError):
                                pass

                return True  # Indicates rate limit error was handled

        return False  # Not a rate limit error

    def _make_github_request(
        self, url, description="GitHub API request", silent_404=False
    ):
        """Make a GitHub API request with rate limit handling"""
        try:
            # Add GitHub token if available
            headers = {}
            github_token = os.environ.get("GITHUB_TOKEN")
            if github_token:
                headers["Authorization"] = f"token {github_token}"

            response = requests.get(url, headers=headers)

            # Check for rate limit before raising for status
            if self._handle_rate_limit_error(response):
                return None  # Rate limit handled, return None

            # Handle 404 errors silently if requested (for optional directories)
            if response.status_code == 404 and silent_404:
                return None

            response.raise_for_status()
            return response

        except requests.exceptions.RequestException as e:
            if hasattr(e, "response") and e.response is not None:
                # Handle 404 silently if requested
                if e.response.status_code == 404 and silent_404:
                    return None

                if not self._handle_rate_limit_error(e.response):
                    st.warning(f"Request failed for {description}: {str(e)}")
            else:
                st.warning(f"Network error for {description}: {str(e)}")
            return None


def main():
    # Title
    st.markdown(
        '<div class="main-header">🔍 Attention Analysis Results Explorer</div>',
        unsafe_allow_html=True,
    )

    # Sidebar for navigation
    st.sidebar.title("🔧 Configuration")

    # Cache management section
    st.sidebar.markdown("### 📁 Data Management")

    # Initialize explorer
    use_cache = st.sidebar.checkbox(
        "Use cached data",
        value=True,
        help="Use previously downloaded data if available",
    )

    if st.sidebar.button("🔄 Clear Cache", help="Clear all cached data"):
        # Clear cache and re-download
        cache_dir = Path(tempfile.gettempdir()) / "attention_results_cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            st.sidebar.success("✅ Cache cleared!")
        st.rerun()

    # Show cache status
    cache_dir = Path(tempfile.gettempdir()) / "attention_results_cache"
    if cache_dir.exists():
        # Get more detailed cache information
        cached_items = []
        for lang_dir in cache_dir.iterdir():
            if lang_dir.is_dir() and lang_dir.name.startswith("results_"):
                lang = lang_dir.name.replace("results_", "")
                configs = [d.name for d in lang_dir.iterdir() if d.is_dir()]
                if configs:
                    models_count = 0
                    for config_dir in lang_dir.iterdir():
                        if config_dir.is_dir():
                            models = [
                                d.name for d in config_dir.iterdir() if d.is_dir()
                            ]
                            models_count += len(models)
                    cached_items.append(
                        f"{lang} ({len(configs)} configs, {models_count} models)"
                    )

        if cached_items:
            st.sidebar.success("✅ **Cached Data:**")
            for item in cached_items[:3]:  # Show first 3
                st.sidebar.text(f"• {item}")
            if len(cached_items) > 3:
                st.sidebar.text(f"... and {len(cached_items) - 3} more")
        else:
            st.sidebar.info("📥 Cache exists but empty")
    else:
        st.sidebar.info("📥 No cached data")

    st.sidebar.markdown("---")

    # Initialize explorer with error handling
    try:
        with st.spinner("🔄 Initializing attention analysis explorer..."):
            explorer = AttentionResultsExplorer(use_cache=use_cache)
    except Exception as e:
        st.error(f"❌ Failed to initialize data explorer: {str(e)}")
        st.error("Please check your internet connection and try again.")

        # Show some debugging information
        with st.expander("🔍 Debugging Information"):
            st.code(f"Error details: {str(e)}")
            st.markdown("**Possible solutions:**")
            st.markdown("- Check your internet connection")
            st.markdown("- Try clearing the cache")
            st.markdown("- Wait a moment and refresh the page")
        return

    # Check if any languages are available
    if not explorer.available_languages:
        st.error("❌ No result data found. Please check the GitHub repository.")
        st.markdown("**Expected repository structure:**")
        st.markdown("- Repository should contain `results_*` directories")
        st.markdown("- Each directory should contain experimental configurations")
        return

    # Show success message
    st.sidebar.success(
        f"✅ Found {len(explorer.available_languages)} languages: {', '.join(explorer.available_languages)}"
    )

    # Language selection
    selected_language = st.sidebar.selectbox(
        "Select Language",
        options=explorer.available_languages,
        help="Choose the language dataset to explore",
    )

    st.sidebar.markdown("---")

    # Configuration selection with dynamic discovery
    st.sidebar.markdown("### ⚙️ Experimental Configuration")

    # Discover available configuration parameters (optimized to use first language only)
    with st.spinner("🔍 Discovering configuration options..."):
        config_parameters = explorer._discover_config_parameters()

    if not config_parameters:
        st.sidebar.error("❌ Could not discover configuration parameters")
        st.stop()

    # Show discovered parameters
    st.sidebar.success(f"✅ Found {len(config_parameters)} configuration parameters")
    st.sidebar.info(
        "💡 Configuration options are consistent across all languages - using optimized discovery"
    )

    # Create UI elements for each discovered parameter
    selected_params = {}

    for param_name, possible_values in config_parameters.items():
        # Clean up parameter name for display
        display_name = param_name.replace("_", " ").title()

        if len(possible_values) == 2 and set(possible_values) == {True, False}:
            # Boolean parameter - use checkbox
            default_value = False  # Default to False for boolean params
            selected_params[param_name] = st.sidebar.checkbox(
                display_name, value=default_value, help=f"Parameter: {param_name}"
            )
        else:
            # Multi-value parameter - use selectbox
            selected_params[param_name] = st.sidebar.selectbox(
                display_name, options=possible_values, help=f"Parameter: {param_name}"
            )

    # Find the best matching configuration
    selected_config, config_exists = explorer._find_matching_config(
        selected_language, selected_params
    )

    st.sidebar.markdown("**Matched Configuration:**")
    st.sidebar.code(
        selected_config if selected_config else "No match found", language="text"
    )

    # Show configuration status
    if config_exists:
        st.sidebar.success("✅ Exact configuration match found!")
    else:
        st.sidebar.warning("⚠️ Using best available match")

    st.sidebar.markdown("---")

    # Get models for selected language and config
    if not selected_config:
        st.error("❌ No valid configuration found")
        st.info("Please try different parameter combinations.")
        st.stop()

    models = explorer._get_models(selected_language, selected_config)
    if not models:
        st.warning(f"❌ No models found for {selected_language}/{selected_config}")
        st.info(
            "This configuration may not exist for the selected language. Try adjusting the configuration parameters above."
        )
        st.stop()

    # Model selection
    selected_model = st.sidebar.selectbox(
        "Select Model", options=models, help="Choose the model to analyze"
    )

    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Overview",
            "🎯 UAS Scores",
            "🧠 Head Matching",
            "📈 Variability",
            "🖼️ Figures",
        ]
    )

    # Tab 1: Overview
    with tab1:
        st.markdown(
            '<div class="section-header">Experiment Overview</div>',
            unsafe_allow_html=True,
        )

        # Show current configuration in a friendly format
        st.markdown("### 🔧 Current Configuration")
        config_params = explorer._parse_config_params(selected_config)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Configuration Parameters:**")
            for param, value in config_params.items():
                emoji = "✅" if value else "❌" if isinstance(value, bool) else "🔹"
                readable_param = param.replace("_", " ").title()
                st.markdown(f"{emoji} **{readable_param}**: {value}")

        with col2:
            st.markdown("**Selected Parameters vs Actual:**")
            for param in selected_params:
                selected_val = selected_params[param]
                actual_val = config_params.get(param, "N/A")
                match_emoji = "✅" if selected_val == actual_val else "⚠️"
                st.markdown(f"{match_emoji} **{param}**: {selected_val} → {actual_val}")

            st.markdown("**Raw Configuration String:**")
            st.code(selected_config, language="text")

        st.markdown("---")

        # Load metadata
        metadata = explorer._load_metadata(
            selected_language, selected_config, selected_model
        )
        if metadata:
            st.markdown("### 📊 Experiment Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Samples", metadata.get("total_number", "N/A"))
            with col2:
                st.metric(
                    "Processed Correctly",
                    metadata.get("number_processed_correctly", "N/A"),
                )
            with col3:
                st.metric("Errors", metadata.get("number_errored", "N/A"))
            with col4:
                success_rate = (
                    (
                        metadata.get("number_processed_correctly", 0)
                        / metadata.get("total_number", 1)
                    )
                    * 100
                    if metadata.get("total_number")
                    else 0
                )
                st.metric("Success Rate", f"{success_rate:.1f}%")

            if metadata.get("random_seed"):
                st.markdown(f"**Random Seed:** {metadata.get('random_seed')}")

            if metadata.get("errored_phrases"):
                with st.expander("🔍 View Errored Phrase IDs"):
                    st.write(metadata["errored_phrases"])
        else:
            st.warning("No metadata available for this configuration.")

        # Quick stats about available data
        st.markdown("---")
        st.markdown(
            '<div class="section-header">Available Data Summary</div>',
            unsafe_allow_html=True,
        )

        # Show loading message since we're now loading on-demand
        with st.spinner("Loading data summary..."):
            uas_data = explorer._load_uas_scores(
                selected_language, selected_config, selected_model
            )
            heads_data = explorer._load_head_matching(
                selected_language, selected_config, selected_model
            )
            variability_data = explorer._load_variability(
                selected_language, selected_config, selected_model
            )
            figures = explorer._get_available_figures(
                selected_language, selected_config, selected_model
            )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("UAS Relations", len(uas_data))
        with col2:
            st.metric("Head Matching Relations", len(heads_data))
        with col3:
            st.metric("Variability Data", "✓" if variability_data is not None else "✗")
        with col4:
            st.metric("Figure Files", len(figures))

        # Show what was just downloaded
        if uas_data or heads_data or variability_data is not None or figures:
            st.success(
                f"✅ Successfully loaded data for {selected_language.upper()}/{selected_model}"
            )
        else:
            st.warning("⚠️ No data files found for this configuration")

    # Tab 2: UAS Scores
    with tab2:
        st.markdown(
            '<div class="section-header">UAS (Unlabeled Attachment Score) Analysis</div>',
            unsafe_allow_html=True,
        )

        uas_data = explorer._load_uas_scores(
            selected_language, selected_config, selected_model
        )

        if uas_data:
            # Relation selection
            selected_relation = st.selectbox(
                "Select Dependency Relation",
                options=list(uas_data.keys()),
                help="Choose a dependency relation to visualize UAS scores",
            )

            if selected_relation and selected_relation in uas_data:
                df = uas_data[selected_relation]

                # Display the data table
                st.markdown("**UAS Scores Matrix (Layer × Head)**")
                st.dataframe(df, use_container_width=True)

                # Create heatmap
                fig = px.imshow(
                    df.values,
                    x=[f"Head {i}" for i in df.columns],
                    y=[f"Layer {i}" for i in df.index],
                    color_continuous_scale="Viridis",
                    title=f"UAS Scores Heatmap - {selected_relation}",
                    labels=dict(color="UAS Score"),
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)

                # Statistics
                st.markdown("**Statistics**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Max Score", f"{df.values.max():.4f}")
                with col2:
                    st.metric("Min Score", f"{df.values.min():.4f}")
                with col3:
                    st.metric("Mean Score", f"{df.values.mean():.4f}")
                with col4:
                    st.metric("Std Dev", f"{df.values.std():.4f}")
        else:
            st.warning("No UAS score data available for this configuration.")

    # Tab 3: Head Matching
    with tab3:
        st.markdown(
            '<div class="section-header">Attention Head Matching Analysis</div>',
            unsafe_allow_html=True,
        )

        heads_data = explorer._load_head_matching(
            selected_language, selected_config, selected_model
        )

        if heads_data:
            # Relation selection
            selected_relation = st.selectbox(
                "Select Dependency Relation",
                options=list(heads_data.keys()),
                help="Choose a dependency relation to visualize head matching patterns",
                key="heads_relation",
            )

            if selected_relation and selected_relation in heads_data:
                df = heads_data[selected_relation]

                # Display the data table
                st.markdown("**Head Matching Counts Matrix (Layer × Head)**")
                st.dataframe(df, use_container_width=True)

                # Create heatmap
                fig = px.imshow(
                    df.values,
                    x=[f"Head {i}" for i in df.columns],
                    y=[f"Layer {i}" for i in df.index],
                    color_continuous_scale="Blues",
                    title=f"Head Matching Counts - {selected_relation}",
                    labels=dict(color="Match Count"),
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)

                # Create bar chart of total matches per layer
                layer_totals = df.sum(axis=1)
                fig_bar = px.bar(
                    x=layer_totals.index,
                    y=layer_totals.values,
                    title=f"Total Matches per Layer - {selected_relation}",
                    labels={"x": "Layer", "y": "Total Matches"},
                )
                fig_bar.update_layout(height=400)
                st.plotly_chart(fig_bar, use_container_width=True)

                # Statistics
                st.markdown("**Statistics**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Matches", int(df.values.sum()))
                with col2:
                    st.metric("Max per Cell", int(df.values.max()))
                with col3:
                    best_layer = layer_totals.idxmax()
                    st.metric("Best Layer", f"Layer {best_layer}")
                with col4:
                    best_head_idx = np.unravel_index(
                        df.values.argmax(), df.values.shape
                    )
                    st.metric("Best Head", f"L{best_head_idx[0]}-H{best_head_idx[1]}")
        else:
            st.warning("No head matching data available for this configuration.")

    # Tab 4: Variability
    with tab4:
        st.markdown(
            '<div class="section-header">Attention Variability Analysis</div>',
            unsafe_allow_html=True,
        )

        variability_data = explorer._load_variability(
            selected_language, selected_config, selected_model
        )

        if variability_data is not None:
            # Display the data table
            st.markdown("**Variability Matrix (Layer × Head)**")
            st.dataframe(variability_data, use_container_width=True)

            # Create heatmap
            fig = px.imshow(
                variability_data.values,
                x=[f"Head {i}" for i in variability_data.columns],
                y=[f"Layer {i}" for i in variability_data.index],
                color_continuous_scale="Reds",
                title="Attention Variability Heatmap",
                labels=dict(color="Variability Score"),
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

            # Create line plot for variability trends
            fig_line = go.Figure()
            for col in variability_data.columns:
                fig_line.add_trace(
                    go.Scatter(
                        x=variability_data.index,
                        y=variability_data[col],
                        mode="lines+markers",
                        name=f"Head {col}",
                        line=dict(width=2),
                    )
                )

            fig_line.update_layout(
                title="Variability Trends Across Layers",
                xaxis_title="Layer",
                yaxis_title="Variability Score",
                height=500,
            )
            st.plotly_chart(fig_line, use_container_width=True)

            # Statistics
            st.markdown("**Statistics**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Max Variability", f"{variability_data.values.max():.4f}")
            with col2:
                st.metric("Min Variability", f"{variability_data.values.min():.4f}")
            with col3:
                st.metric("Mean Variability", f"{variability_data.values.mean():.4f}")
            with col4:
                most_variable_idx = np.unravel_index(
                    variability_data.values.argmax(), variability_data.values.shape
                )
                st.metric(
                    "Most Variable", f"L{most_variable_idx[0]}-H{most_variable_idx[1]}"
                )
        else:
            st.warning("No variability data available for this configuration.")

    # Tab 5: Figures
    with tab5:
        st.markdown(
            '<div class="section-header">Generated Figures</div>',
            unsafe_allow_html=True,
        )

        figures = explorer._get_available_figures(
            selected_language, selected_config, selected_model
        )

        if figures:
            st.markdown(f"**Available Figures: {len(figures)}**")

            # Group figures by relation type
            figure_groups = {}
            for fig_path in figures:
                # Extract relation from filename
                filename = fig_path.stem
                relation = filename.replace("heads_matching_", "").replace(
                    f"_{selected_model}", ""
                )
                if relation not in figure_groups:
                    figure_groups[relation] = []
                figure_groups[relation].append(fig_path)

            # Select relation to view
            selected_fig_relation = st.selectbox(
                "Select Relation for Figure View",
                options=list(figure_groups.keys()),
                help="Choose a dependency relation to view its figure",
            )

            if selected_fig_relation and selected_fig_relation in figure_groups:
                fig_path = figure_groups[selected_fig_relation][0]

                st.markdown(f"**Figure: {fig_path.name}**")
                st.markdown(f"**Path:** `{fig_path}`")

                # Note about PDF viewing
                st.info(
                    "📄 PDF figures are available in the results directory. "
                    "Due to Streamlit limitations, PDF files cannot be displayed directly in the browser. "
                    "You can download or view them locally."
                )

                # Provide download link
                try:
                    with open(fig_path, "rb") as file:
                        st.download_button(
                            label=f"📥 Download {fig_path.name}",
                            data=file.read(),
                            file_name=fig_path.name,
                            mime="application/pdf",
                        )
                except Exception as e:
                    st.error(f"Could not load figure: {e}")

            # List all available figures
            st.markdown("**All Available Figures:**")
            for relation, paths in figure_groups.items():
                with st.expander(f"📊 {relation} ({len(paths)} files)"):
                    for path in paths:
                        st.markdown(f"- `{path.name}`")
        else:
            st.warning("No figures available for this configuration.")

    # Footer
    st.markdown("---")

    # Data source information
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(
            "🔬 **Attention Analysis Results Explorer** | "
            f"Currently viewing: {selected_language.upper()} - {selected_model} | "
            "Built with Streamlit"
        )
    with col2:
        st.markdown(
            f"📊 **Data Source**: [GitHub Repository](https://github.com/{explorer.github_repo})"
        )


if __name__ == "__main__":
    main()
