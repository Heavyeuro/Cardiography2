import data_analysis as da
import prepare_dataset as pds
import learning_model as lm
import core_action as ca


if __name__ == '__main__':
    patients_info_filename = 'patientsInfo.csv'
    ECGS_filename = 'ecgs.csv'

    # build graphs and analyze dataset
    pds.prepare_ecg(ECGS_filename)

    # prepare generated DF(values replacing, imputing)
    dataframe = pds.prepare_dataset_core(patients_info_filename)

    # build graphs and analyze dataset
    # da.describe_dataframe_core(dataframe)

    # Joining ECGS_data and patients info
    ECGS_data = ca.read_csv('../DataSource/' + 'inlined_' + ECGS_filename)
    X = dataframe.set_index('patient_id').join(ECGS_data.set_index('patient_id'))
    # X = dataframe.join(ECGS_data)
    path = '../DataSource/'
    X.to_csv(path + "joined_data.csv")

    # building ML
    lm.build_and_score_ml_model_core(X)
