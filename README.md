# Stabl: sparse and reliable biomarker discovery in predictive modeling of high-dimensional omic data

This is a scikit-learn compatible Python implementation of Stabl, coupled with useful functions and
example notebooks to rerun the analyses on the different use cases located in the `sample data` folder

[![DOI]](doi:10.5061/dryad.stqjq2c7d)

## Overview
High-content omic technologies coupled with sparsity-promoting regularization methods (SRM) have transformed the 
biomarker discovery process. However, the translation of computational results into a clinical use-case scenario remains
challenging. A rate-limiting step is the rigorous selection of reliable biomarker candidates among a host of biological 
features included in multivariate models. We propose Stabl, a machine learning framework that unifies the biomarker 
discovery process with multivariate predictive modeling of clinical outcomes by selecting a sparse and reliable set of 
biomarkers. Evaluation of Stabl on synthetic datasets and four independent clinical studies demonstrates improved 
biomarker sparsity and reliability compared to commonly used SRMs at similar predictive performance. Stabl readily 
extends to double- and triple-omics integration tasks and identifies a sparser and more reliable set of biomarkers than 
those selected by state-of-the-art early- and late-fusion SRMs, thereby facilitating the biological interpretation and 
clinical translation of complex multi-omic predictive models. 

## Requirements

* scikit-learn >= 1.1.2
* knockpy >= 1.2
* pandas >= 1.4.2
* numpy >= 1.23.1
* joblib >= 1.1.0
* tqdm >= 4.64.0
* seaborn >= 0.12.0
* matplotlib >= 3.5.2


## Installation
Install Directly from github:

```
pip install git+https://github.com/gregbellan/Stabl.git
```
or 

---

Download Stabl:

```
git clone https://github.com/gregbellan/Stabl.git
```
Install requirements and Stabl:

```
pip install .
```

The general installation time is less than 10 seconds, and have been tested on mac OS and linux system.

## Input data
When using your own data, you have to provide

* The preprocessed input data matrix (preferably a pandas DataFrame having column names)
* The outcomes (preferably a pandas Series having a names)
* (Input Data and outcomes should have the same indices)

## Sample Data

The "Sample Data" folder contains data for the following use cases:
### Onset of Labor
#### Training
* **Outcome**: Days before Labor, `150` samples — `53` patients 
* **Proteomics**: `150` samples — `1317` biomarkers
* **CyTOF**: `150` samples — `1502` biomarkers
* **Metabolomics**: `150` samples — `3529` biomarkers 
#### Validation
* **Outcome**: Days before Labor, `27` samples — `10` patients 
* **Proteomics**: `21` samples — `1317` biomarkers
* **CyTOF**: `27` samples — `1502` biomarkers

### COVID-19
#### Training
* **Outcome**: Mild/Moderate (`43`) Vs. Severe (`25`)
* **Proteomics**: `68` samples — `1463` biomarkers
#### Validation
* **Outcome**: Mild/Moderate (`125`) Vs. Severe (`659`)
* **Proteomics**: `784` samples — `1420` biomarkers

### CFRNA Preeclampsia
#### Training
* **Outcome**: Control (`63`) Vs. Preeclampsia (`96`) — `48` patients
* **CFRNA**: `159` samples — `37184` biomarkers

### Surgical Site Infections (SSI)
#### Training
* **Outcome**: Control (`77`) Vs. SSI (`16`)
* **CyTOF**: `93` samples — `1125` biomarkers
* **Proteomics**: `91` samples — `721` biomarkers

### Benchmarks
* `Tutorial Notebooks.ipynb`: Tutorial on how to use the library
* `* Benchmarks.ipynb`: Jupyter Notebook rerunning all the benchmarks 

## Cite
Julien Hedou, Ivana Maric, Grégoire Bellan et al. Stabl: sparse and reliable biomarker discovery in predictive modeling 
of high-dimensional omic data, 27 February 2023, PREPRINT (Version 1) available at Research Square 
[https://doi.org/10.21203/rs.3.rs-2609859/v1]
  
