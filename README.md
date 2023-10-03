# Stabl: sparse and reliable biomarker discovery in predictive modeling of high-dimensional omic data

This is a scikit-learn compatible Python implementation of Stabl, coupled with useful functions and
example notebooks to rerun the analyses on the different use cases located in the `sample data` folder

[![DOI]](doi:10.5061/dryad.stqjq2c7d)

## Abstract
High-content omic technologies coupled with sparsity-promoting regularization methods (SRM) have transformed the
biomarker discovery process. However, the translation of computational results into a clinical use-case scenario remains
challenging. A rate-limiting step is the rigorous selection of reliable biomarker candidates among a host of biological
features included in multivariable models. We propose Stabl, a general machine learning framework that unifies the
biomarker discovery process with multivariable predictive modeling of clinical outcomes by selecting a sparse and 
reliable set of biomarkers. Evaluation of Stabl on synthetic datasets and five independent clinical studies demonstrates
improved biomarker sparsity and reliability compared to commonly used SRMs at similar predictive performance. Stabl 
readily extends to double- and triple-omics integration tasks and identifies a sparser and more reliable set of 
biomarkers than those selected by state-of-the-art early- and late-fusion SRMs, thereby facilitating the biological
interpretation and clinical translation of complex multi-omic predictive models.

## Requirements

Python version : from 3.7 up to 3.10

Python packages:

* joblib == 1.1.0
* tqdm == 4.64.0
* matplotlib == 3.5.2
* numpy == 1.23.1
* cmake == 3.27.1
* knockpy == 1.2
* scikit-learn == 1.1.2
* seaborn == 0.12.0
* groupyr == 0.3.2
* pandas == 1.4.2
* statsmodels == 0.14.0
* openpyxl == 3.0.7
* adjustText == 0.8
* scipy == 1.10.1
* julia == 0.6.1
* osqp == 0.6.2


Julia package for noise generation (version 1.9.2) :

* Bigsimr
* Distributions
* PyCall

## Installation

### Julia installation
To install Julia, please follow these instructions: 

1. Download Julia from [here](https://julialang.org/downloads/).
2. Follow the instructions for your operating system [here](https://julialang.org/downloads/platform/).
3. Install the required julia packages :
    ```

    julia -e 'using Pkg; Pkg.add("Bigsimr"); Pkg.add("Distributions"); Pkg.add("PyCall"); Pkg.add("IJulia")'

    ```
4. Finally, install Julia for python:
    ```
    pip install julia
    python -c "import julia; julia.install()"
    ``` 

### CMake installation

In order to install the python libraries required to generate the noise, we need to install :
* CMake

You can install this module by :
* using the default system package manager, like on this [website](https://cgold.readthedocs.io/en/latest/first-step/installation.html)
* following instructions on [CMake](https://cmake.org/install/).


### Python installation (>= 3.7 and < 3.11)

Install Directly from github:

```
pip install git+https://github.com/gregbellan/Stabl.git
pip install numpy==1.23.2

```
or 

---

Download Stabl:

```
git clone https://github.com/gregbellan/Stabl.git
```
Install requirements and Stabl:

```
cd Stabl
pip install .
pip install numpy==1.23.2
```

The general installation time is less than 10 seconds, and have been tested on mac OS and linux system.

> **_NOTE:_**  There is a behavior with Julia library:
> - you can run the script in a notebook, but you need to run the import block two times. The first will throw an error and the second one will finalize the import.
> - It is not possible to run the script in command line if you are installing the library with conda
> To resolve this issue, either you install the library without conda or you run the script into a notebook.
> 
> If there is still an issue with Julia in a notebook, run the following command in the first cell of the notebook:
> ```
> from julia.api import Julia
> jl = Julia(compiled_modules=False) 
> ```

## Use of the library

To use the library and the associated benchmark in the folder `Notebook examples`, you need to download the repository :

```
git clone https://github.com/gregbellan/Stabl.git
cd Stabl/
unzip Sample\ Data/data.zip -d Sample\ Data/
```

### Benchmarks
* `Tutorial Notebook.ipynb`: Tutorial on how to use the library
* `run_cv_*.py`: Python scripts to run the sample datas in Cross-Validation
* `run_val_*.py`: Python scripts to run the sample datas in Training-Validation
* `run_synthetic_*.py`: Python scripts to run the synthetic benchmarks

> **_NOTE:_** 
> The different scripts may take some time to begin because of the dependence with julia. However, once started, the 
> time to run should come back to normal.

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

### Dream
#### Training
* **Outcome**: Preterm (`609`) Vs. Non-preterm (`960`) - 580 patients
* **Taxonomy**: `1569` samples — `3725` biomarkers
* **Phylotype**: `1569` samples — `5468` biomarkers

## Cite
Julien Hedou, Ivana Maric, Grégoire Bellan et al. Stabl: sparse and reliable biomarker discovery in predictive modeling 
of high-dimensional omic data, 27 February 2023, PREPRINT (Version 1) available at Research Square 
[https://doi.org/10.21203/rs.3.rs-2609859/v1]
  
