# Course Catalog Named Entity Recognition (NER)

# Group6 Named Entity Recognition (NER) Pipeline

This repository contains all the code and data files used by Group 6 for a domain-specific Named Entity Recognition (NER) project as part of our NLP course at UTS.

## Repository Structure

| File/Folder                               | Description                                                                 |
|-------------------------------------------|-----------------------------------------------------------------------------|
| `Group6_NLP_NER.ipynb`                    | Main notebook with tokenization, training, evaluation, and inference logic |
| `ner-ui.py`                               | Streamlit-based UI to test the trained NER model                           |
| `project-7-at-2025-05-16-18-37-eec27312.json` | Annotated dataset exported from Label Studio                           |
| `uiuc_chunk.zip`                          | Raw data chunks used for annotation                                         |
| `uiuc_labelstudio_preannotated.json`      | Pre-annotated Label Studio-compatible JSON file                             |
| `README.md`                               | This file                                                                   |

## Getting Started

### 1. Clone the repository
```bash
git clone git@github.com:your-username/your-repo.git
cd your-repo
```

### 2. Setup the environment
Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> Note: Add a `requirements.txt` with relevant libraries like `transformers`, `datasets`, `seqeval`, `streamlit`, `torch`, etc.

### 3. Run the Jupyter Notebook
```bash
jupyter notebook Group6_NLP_NER.ipynb
```

### 4. Launch the Streamlit UI
```bash
streamlit run ner-ui.py
```


## Model Details

- Model: Fine-tuned transformer (e.g., `bert-base-cased`)
- Metrics: F1 Score using `seqeval`


## Contributors

This project was developed by a student group as part of the NLP course (42850) at the University of Technology Sydney (UTS).

Team Members:
- Anusha Pariti
- Abhinandan
- Gaurav
- Jeevanandh
- Rajdeep

License
This repository is intended for academic use. Dataset sources retain their original licensing. Please cite if used in derivative works.
