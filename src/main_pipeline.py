import numpy as np
from pandas import DataFrame

import data_analysis as da
import prepare_dataset as pds
import learning_model as lm
import core_action as ca


# replacing inf with nan
def replace_inf_with_nan_and_impute(X: DataFrame):
    df = pds.apply_encoder(X)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = pds.simple_imputing_data(df, df)
    return df[0]


# build heatmaps for 1st and 2nd sensors
def build_heatmaps(ECGS_data: DataFrame):
    ECGS_data = pds.apply_encoder(ECGS_data)
    ECGS_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    ECGS_data1 = ECGS_data.iloc[:, 1:14]

    # forming specific for 2nd sensor column names
    cols2 = []
    for col in ECGS_data1.columns:
        cols2.append(col[:-1]+'2')
    ECGS_data2 = ECGS_data[cols2]

    da.correlation_heatmap(ECGS_data1)
    da.correlation_heatmap(ECGS_data2)


if __name__ == '__main__':
    patients_info_filename = 'patientsInfo.csv'
    ECGS_filename = 'ecgs.csv'
    path = '../DataSource/'

    # # build graphs and analyze dataset/(5min)
    # # run only once if inlined_ecgs.csv was not generated
    # pds.prepare_ecg(ECGS_filename)

    # prepare generated DF(values replacing, imputing)
    dataframe = pds.prepare_dataset_core(patients_info_filename)

    # Joining ECGS_data and patients info
    ECGS_data = ca.read_csv('../DataSource/' + 'inlined_' + ECGS_filename)

    X = dataframe.set_index('patient_id').join(ECGS_data.set_index('patient_id'))
    X.to_csv(path + "joined_data.csv")
    X = replace_inf_with_nan_and_impute(X)

    # building ML
    lm.build_and_score_ml_model_core(X)