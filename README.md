# Cancer Extraction and Classification using BERT and LangChain

**Author:** Rahul Nair

**Last Modified:** 30-05-2025

<br><br>

## Overview

The objective here is to classify research paper abstracts (text-classification)  in to two classes (binary classification) cancer vs non-cancer. Also,  extract  specific cases of cancer from these paper abstracts. 

**Steps:** 

1. Given a dataset of 500 datapoints (PubMed - PMID, Title, Abstract) each for Cancer and Non-Cancer classes. We call this the target dataset

2. Preprocess data:

   ​	a. Detect encoding and parse the data - separate PMID, Title, and Abstract.

   ​	b. Remove metadata like email, copyright, citations, doi etc., using pattern matching

   ​	c. Remove html content, formatting.

3. Get additional datasets for training and pre-process it: 

   ​	(i) `ncbi/ncbi_disease`,  (ii) `bigbio/bionlp_st_2013_cg` (iii) `bigbio/bc5cdr`

4. Split and combine the dataset such that 20% of target dataset is used for testing.

   The rest is split in to train and validation data.

5. Obtain a BERT model for fine-tuning. Model used: `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`.

6. Similarly, obtain a model for extracting cancer term from the abstract. Model used: `alvaroalon2/biobert_diseases_ner`.

7. Run prediction on the models sequentially using Langchain.

<br><br>

## Environment Setup:

```
./download_data.sh
```

<br>

**Set up docker container**

```
cd docker
./init-docker.sh
./build_docker.sh
./run_docker.sh
```

<br><br>

## Train and Evaluate: 

<br>

**Demo**
Base model used for finetuning: [BiomedBERT]("https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
Extraction model used: [NER Model]("https://huggingface.co/alvaroalon2/biobert_diseases_ner")
```
python demo.py --train --full_extraction --example

--train: finetunes the base model and provides before and after finetuning results 
--full_extraction: extracts cancers from the given abstract using extraction model
--example: perform class prediction and extraction on an example

```

<br>

**LangChain Demo**

Classification and extraction models are run in a LangChain

```
python langchain_demo.py

```

<br><br>

## Results & Observations:

<br><br>


