# RAG System for Medical Operations Information Retrieval

## Overview

I wrote an RAG system using a deepseek reasoning model for information extraction on medical operation data. I then evaluated my
system on some toy evaluation datasets I annotated myself. This repo includes the preprocessing, hand-annotated files evaluation files, and the retrieval and evaluation class.
See my final paper for more information.

## Data
While my evaluation data can be found on this repo, the rest of this data was too large to keep in Github. Here is a [link](https://drive.google.com/drive/u/2/folders/1BW1p_cuer0tcVCRouEq0c86BN_gtKcIa) to the rest of the data
in Google Drive. Within this file you can also find my presentation and notes on the results of running my evaluation script on the evaluation data.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <https://github.com/BenLambright/Medical-Operations-RAG.git>
    cd <Medical-Operations-RAG>
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Preprocessing

1.  Navigate to the `preprocessing` directory:
    ```bash
    cd preprocessing
    ```

2.  Run the necessary preprocessing scripts, noting that this builds the "verbose" dataset:
    ```bash
    python <build_database.py>
    ```

### Retrieval

1.  Build a rag system
    ```bash
    python retrieval.py
    ```
    See the comments at the bottom of this file to see how you can create a retriever object and invoke the model.

### Evaluation

1.  Run the `eval.py` script:
    ```bash
    python eval.py
    ```
    Currently set to evaluate on the materials.json dataset
