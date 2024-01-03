import pandas as pd
import numpy as np
from .preprocessing import remove_low_info_samples
from os.path import join


"""All the load functions return the following in this order:
train_data_dict: dict
    Dictionary of training data. Keys are the omics names and values are the dataframes.
valid_data_dict: dict or None
    Dictionary of validation data. Keys are the omics names and values are the dataframes.
    If None, the dataset does not have a validation set.
y_train: pd.Series
    Series of training labels for the outcome.
y_valid: pd.Series or None
    Series of validation labels for the outcome.
    If None, the dataset does not have a validation set.
patients_id: pd.Series
    Series of patient IDs.
task_type: str
    Type of the task. Either "binary" or "regression".
"""


def load_onset_of_labor(data_path):
    # continuous outcome dataset with validation set multi-omics
    patients_id = pd.read_csv(
        join(data_path, "Training", "ID.csv"), index_col=0).Id

    # importing the training data
    y_train = pd.read_csv(
        join(data_path, "Training", "DOS.csv"), index_col=0).DOS
    cyto_train = pd.read_csv(
        join(data_path, "Training", "CyTOF.csv"), index_col=0)
    prot_train = pd.read_csv(
        join(data_path, "Training", "Proteomics.csv"), index_col=0)

    # importing the validation data
    y_valid = pd.read_csv(join(data_path, "Validation",
                               "DOS_validation.csv"), index_col=0).DOS
    cyto_valid = pd.read_csv(
        join(data_path, "Validation", "CyTOF_validation.csv"), index_col=0)
    cyto_valid.iloc[:, :-41] = cyto_valid.iloc[:, :-41].apply(lambda x: np.sinh(x)*5)

    prot_valid = pd.read_csv(
        join(data_path, "Validation", "Proteomics_validation.csv"), index_col=0)

    train_data_dict = {
        "CyTOF": cyto_train,
        "Proteomics": prot_train
    }

    valid_data_dict = {
        "CyTOF": cyto_valid,
        "Proteomics": prot_valid
    }
    task_type = "regression"

    return train_data_dict, valid_data_dict, y_train, y_valid, patients_id, task_type


def load_dream(data_path):
    # continuous outcome dataset with validation set multi-omics
    patients_id = pd.read_csv(join(data_path, "Patients_id.csv"), index_col=0).participant_id

    # importing the training data
    y_train = pd.read_csv(join(data_path, "Preterm.csv"), index_col=0).was_preterm
    y_train = y_train.astype(int)
    taxo_train = pd.read_csv(join(data_path, "Taxonomy.csv"), index_col=0)
    phylo_train = pd.read_csv(join(data_path, "Phylotype.csv"), index_col=0)

    train_data_dict = {
        "Phylotype": phylo_train,
        "Taxonomy": taxo_train
    }

    task_type = "binary"

    return train_data_dict, None, y_train, None, patients_id, task_type


def load_onset_of_labor_cv(data_path):
    # continuous outcome dataset without validation set multi-omics
    y_train = pd.read_csv(
        join(data_path, "Training", "DOS.csv"), index_col=0).DOS
    patients_id = pd.read_csv(
        join(data_path, "Training", "ID.csv"), index_col=0).Id

    meta_train = pd.read_csv(
        join(data_path, "Training", "Metabolomics.csv"), index_col=0)
    cyto_train = pd.read_csv(
        join(data_path, "Training", "CyTOF.csv"), index_col=0)
    prot_train = pd.read_csv(
        join(data_path, "Training", "Proteomics.csv"), index_col=0)

    train_data_dict = {
        "CyTOF": cyto_train,
        "Proteomics": prot_train,
        "Metabolomics": meta_train
    }
    task_type = "regression"

    return train_data_dict, None, y_train, None, patients_id, task_type


def load_cfrna(data_path, percentile=None):
    # categorical outcome dataset without validation set mono-omic
    X = pd.read_csv(join(data_path, "cfrna_dataFINAL.csv"), index_col=0)
    IDs = pd.read_csv(join(data_path, "ID.csv"), index_col=0)
    y = pd.read_csv(join(data_path, "all_outcomes.csv"), index_col=0)

    # Removing samples without any information
    X = remove_low_info_samples(X)
    # Applying the log2(x+1) transformation
    X = X.apply(lambda x: np.log2(x+1))

    if percentile is not None:
        thresh_var = np.percentile(X.var(), percentile)
        X = X.loc[:, X.var() >= thresh_var]

    IDs = IDs.loc[X.index].ID
    y = y.loc[X.index].Preeclampsia

    X = {
        "CFRNA": X
    }
    task_type = "binary"

    return X, None, y, None, IDs, task_type


def load_covid_19(data_path):
    # categorical outcome dataset with validation set mono-omic
    # Importing the training data
    X_train = pd.read_csv(
        join(data_path, "Training", "Proteomics.csv"), index_col="sampleID")
    y_train = pd.read_csv(
        join(data_path, "Training", "Mild&ModVsSevere.csv"), index_col=0).iloc[:, 0]

    # Importing the validation data
    X_val = pd.read_csv(join(data_path, "Validation",
                        "Validation_proteomics.csv"), index_col=0)
    y_val = ~pd.read_csv(join(data_path, "Validation",
                         "Validation_outcome(WHO.0 >= 5).csv"), index_col=0).iloc[:, 0]

    X_train = {
        "Proteomics": X_train
    }
    X_val = {
        "Proteomics": X_val
    }
    task_type = "binary"

    return X_train, X_val, y_train, y_val, None, task_type


def load_ssi(data_path):
    # categorical outcome dataset without validation set multi-omics
    # importing the training data
    y_train = pd.read_csv(
        join(data_path, "outcome.csv"), index_col=0).model1b
    cyto_train = pd.read_csv(
        join(data_path, "CyTOF.csv"), index_col=0)
    prot_train = pd.read_csv(
        join(data_path, "Proteomics.csv"), index_col=0)

    train_data_dict = {
        "CyTOF": cyto_train,
        "Proteomics": prot_train
    }

    task_type = "binary"

    return train_data_dict, None, y_train, None, None, task_type
