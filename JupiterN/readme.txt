To execute model please run "main()" function


Data transformation:
1) Initial preprocessed data could be found on path \DataSource\metrics.
2) It is converted(function prepare_ecg) to the single file: inlined_ecgs.csv without any changes.
3) A file patientsInfo.csv contains medical patients data wich was not processed anyhow but taken from initial database.
4) Data from the file patientsInfo.csv is "binarized".
Casting values in diagnosis col to binary task replacing:
    MI as false(0)
    NORM as true(1)
    other values are dropped out from dataframe

5) A file joined_data.csv is a result on joining sets of initial data and general information about the patient.

6) Based on the joined_data.csv predition model was buid. (np.inf were replaced with np.nan and non numerical values categorized)
