# Medical NLP Assignment
## Overview

This project implements an end-to-end Medical Natural Language Processing (NLP) pipeline for analyzing doctor–patient conversations. The system extracts structured medical information, analyzes patient sentiment and intent, and generates a clinically consistent SOAP note.

The design follows a hybrid approach that combines machine learning with rule-based clinical logic to ensure accuracy, explainability, and safety.

## Features

The pipeline consists of three main components:

### edical NLP Summarization

Extracts structured medical information such as symptoms, diagnosis, treatment, current status, and prognosis.

### Patient Sentiment and Intent Analysis

Analyzes patient responses to understand emotional state and intent in a clinically safe manner.

### SOAP Note Generation

Automatically generates a structured SOAP note:

        1. Subjective

        2. Objective

        3. Assessment

        4. Plan

## Technology Stack

Python

spaCy and SciSpacy (medical named entity recognition)

Transformer-based models (DistilBERT / ClinicalBERT)

Custom transformer classifier for SOAP section classification

Rule-based clinical reasoning layer

JSON for structured outputs

## Project Structure
.
├── data/
│   └── conversation.txt
├── outputs/
│   ├── medical_summary.json
│   ├── sentiment_intent.json
│   └── soap_note.json
├── src/
│   ├── text_loader.py
│   ├── transcript_parser.py
│   ├── nlp_summarization.py
│   ├── sentiment_intent.py
│   ├── sentence_utils.py
│   ├── soap_classifier.py
│   ├── soap_generator.py
│   └── soap_builders.py
├── requirements.txt
├── main.py
└── README.md

## Setup and Execution
1. Create a virtual environment
python -m venv venv
source venv/bin/activate

2. Install dependencies
pip install -r requirements.txt

3. Run the pipeline
python main.py

## Outputs

After successful execution, the following files are generated in the outputs/ directory:

medical_summary.json
Structured medical information extracted from the conversation.

sentiment_intent.json
Patient sentiment and intent analysis.

soap_note.json
Final SOAP note generated using the hybrid ML and rule-based approach.
