# Attention Analysis Results Explorer

A Streamlit web application for exploring attention analysis results from transformer models across multiple languages and configurations.

## Features

- 🌐 **Multi-language Support**: Explore results for 8 languages (Catalan, English, Spanish, Basque, French, Galician, Portuguese, Turkish)
- 🔧 **Configuration Comparison**: Compare different experimental configurations
- 🤖 **Model Analysis**: Analyze results across various transformer models (BERT, Llama, Mistral, etc.)
- 📊 **Interactive Visualizations**: Heatmaps, charts, and statistical summaries
- 📈 **Multiple Data Types**: UAS scores, head matching patterns, variability analysis
- 🖼️ **Figure Access**: Download and view generated PDF figures

## Data Source

The app automatically downloads results data from the GitHub repository: https://github.com/ACMCMC/attention

## Local Development

1. **Clone the repository** (if developing locally)
2. **Create virtual environment**:
   ```bash
   python -m venv streamlit_env
   source streamlit_env/bin/activate  # On Windows: streamlit_env\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r streamlit_requirements.txt
   ```
4. **Run the app**:
   ```bash
   streamlit run streamlit_app.py
   ```

## Deployment to HuggingFace Spaces

### Option 1: Direct Upload

1. Create a new Space on HuggingFace Spaces
2. Choose "Streamlit" as the SDK
3. Upload these files:
   - `streamlit_app.py`
   - `streamlit_requirements.txt` (rename to `requirements.txt`)
4. The app will automatically build and deploy

### Option 2: Git Repository

1. Create a new repository for the Streamlit app
2. Copy `streamlit_app.py` and rename `streamlit_requirements.txt` to `requirements.txt`
3. Connect your HuggingFace Space to the Git repository

### Required Files for HuggingFace

- `streamlit_app.py` - Main application file
- `requirements.txt` - Python dependencies (rename from `streamlit_requirements.txt`)
- `README.md` - This documentation

### Environment Variables (Optional)

You can set these in HuggingFace Spaces settings if needed:
- `GITHUB_REPO` - Override the default repository (default: "ACMCMC/attention")

## Usage

1. **Select Language**: Choose from available language datasets
2. **Choose Configuration**: Pick an experimental configuration to analyze
3. **Select Model**: Choose a transformer model to examine
4. **Explore Data**: Navigate through different tabs:
   - **Overview**: General statistics and metadata
   - **UAS Scores**: Unlabeled Attachment Score analysis
   - **Head Matching**: Attention head matching patterns
   - **Variability**: Attention variability across layers
   - **Figures**: Download generated visualizations

## Data Structure

The app expects results organized as:
```
results_{language}/
├── {configuration}/
│   ├── {model}/
│   │   ├── metadata/
│   │   │   └── metadata.json
│   │   ├── uas_scores/
│   │   │   └── uas_{relation}.csv
│   │   ├── number_of_heads_matching/
│   │   │   └── heads_matching_{relation}_{model}.csv
│   │   ├── variability/
│   │   │   └── variability_list.csv
│   │   └── figures/
│   │       └── heads_matching_{relation}_{model}.pdf
```

## Technical Details

- **Framework**: Streamlit
- **Visualization**: Plotly
- **Data Processing**: Pandas, NumPy
- **Data Source**: GitHub API
- **Caching**: Local filesystem cache for performance

## License

Same as the main attention analysis project.
