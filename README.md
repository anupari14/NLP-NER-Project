# Course Catalog Named Entity Recognition (NER)

This project applies transformer-based Named Entity Recognition (NER) to extract structured metadata from unstructured university course descriptions. The system identifies entities such as `CourseCode`, `CourseTitle`, `Instructor`, `Credit`, `TimeSlot`, and `Location` to support automated indexing, academic advising, and education analytics.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [NER Labels](#ner-labels)
- [Annotation Workflow](#annotation-workflow)
- [Model Architecture](#model-architecture)
- [Training Setup](#training-setup)
- [Evaluation](#evaluation)
- [Streamlit Inference UI](#streamlit-inference-ui)
- [Repository Structure](#repository-structure)
- [Contributors](#contributors)
- [License](#license)

## Overview

This project demonstrates an end-to-end pipeline for fine-tuning a BERT-based token classification model on university course catalog data. The system includes data preprocessing, manual and model-assisted annotation using Label Studio, training with Hugging Face Transformers, and deployment via a lightweight Streamlit interface.

## Dataset

- **Source**: [UIUC Course Catalog Dataset](https://discovery.cs.illinois.edu/dataset/course-catalog/)
- **Format**: Preprocessed into JSON and text chunks
- **Annotations**: Exported from Label Studio in BIO format for token classification

## NER Labels

The model is trained to recognize the following custom entity classes:

- `CourseCode`
- `CourseTitle`
- `Credit`
- `Instructor`
- `Location`
- `TimeSlot`

## Annotation Workflow

1. Extract course description text from the UIUC catalog.
2. Annotate sample records manually in Label Studio.
3. Fine-tune an initial BERT model.
4. Use the model to pre-label new data chunks.
5. Manually review and correct predictions.
6. Retrain with the expanded dataset.

## Model Architecture

- **Base Model**: `bert-base-cased`
- **Task**: Token classification
- **Tokenizer**: WordPiece tokenizer (Hugging Face)
- **Framework**: Hugging Face Transformers + Datasets

## Training Setup

- **Batch Size**: 8
- **Epochs**: 5
- **Learning Rate**: 5e-5
- **Optimizer**: AdamW
- **Validation Split**: 20%
- **Metrics**: Precision, Recall, F1-score

## Evaluation

Model performance was evaluated on a manually annotated validation set. F1-scores ranged from high (1.0 for `TimeSlot`) to moderate (0.63 for `CourseTitle`) depending on label frequency and variability.

> See `notebooks/NER_training_evaluation.ipynb` for detailed metrics.

## Streamlit Inference UI

A simple web app is provided under the `app/` directory for real-time inference. You can input a course description and visualize extracted entities.

### Run locally:

```bash
cd app
streamlit run ner-ui.py

