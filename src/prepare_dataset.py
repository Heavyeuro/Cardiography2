import os

import numpy as np
from pandas import DataFrame
from sklearn.impute import SimpleImputer
import core_action as ca
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path


def detect_vals(obj_to_describe, val, exclude_col=[]):
    # missing values by columns
    obj_to_describe = obj_to_describe.drop(exclude_col, axis=1)
    missing_val_count_by_column = (obj_to_describe.isin([val]).sum())
    return (missing_val_count_by_column)


def prepare_ecg(ECGS_filename):
    source_path = '../DataSource/'
    directory = source_path + 'metrics/'
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            if 'df' not in locals():
                df = ca.read_csv(f)
                df_inlined = inline_single_ecg(df)
            else:
                current_line = ca.read_csv(f)
                df = pd.concat([df, current_line])
                df_inlined = pd.concat([df_inlined, inline_single_ecg(current_line)])

    df.to_csv('../DataSource/' + ECGS_filename)
    df_inlined.to_csv('../DataSource/' + 'inlined_'+ECGS_filename)


def inline_single_ecg(df: DataFrame):
    patient_id = df.iloc[0, 0]
    coloumns_to_drop = ['patient_id', 'ecg_id', 'lead']
    for col in coloumns_to_drop:
        df.drop(col, axis=1, inplace=True)
    df.index = df.index + 1
    df_out = df.stack()
    df_out.index = df_out.index.map('{0[1]}_{0[0]}'.format)
    df_inlined = df_out.to_frame().T
    df_inlined['patient_id'] = patient_id
    return df_inlined


# Refactoring output data col to True and False
# Can be replaced in future by Label Encoding for categorical variable
def prepare_dataset_core(name_csv: str):
    df = ca.read_csv('../DataSource/' + name_csv)
    col = 'diagnostic_superclass'
    ca.replace_val_in_cols(df, [col], "['MI']", 'False')
    ca.replace_val_in_cols(df, [col], "['NORM']", 'True')

    ca.replace_val_in_cols_except(df, col, ['True', 'False'], col)
    ca.drop_line_if_it_contains(df, col, col)

    df = apply_encoder(df)

    sid = simple_imputing_data(df, df)

    return sid[0]


# Replacing missing values (imputing) according to certain strategy
def simple_imputing_data(X_train, X_valid, strategy:str = 'mean'):
    simple_imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
    imputed_X_train = pd.DataFrame(simple_imputer.fit_transform(X_train))
    imputed_X_valid = pd.DataFrame(simple_imputer.transform(X_valid))
    # Imputation removed column names; put them back
    imputed_X_train.columns = X_train.columns
    imputed_X_valid.columns = X_valid.columns

    return imputed_X_train, imputed_X_valid


# Encoding categorical variables
def detect_vals(obj_to_describe, val, exclude_col=[]):
    # missing values by columns
    obj_to_describe = obj_to_describe.drop(exclude_col, axis=1)
    missing_val_count_by_column = (obj_to_describe.isin([val]).sum())
    return (missing_val_count_by_column)


def apply_encoder(X: DataFrame):
    s = (X.dtypes == 'object')
    object_cols = list(s[s].index)

    for col in object_cols:
        X[col] = LabelEncoder().fit_transform(X[col])

    return X
