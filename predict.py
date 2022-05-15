import pandas as pd
import numpy as np
import pickle
import sys
import os
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score


def feature_agg(patient_df):
    patient_df = patient_df.drop(columns=['SepsisLabel'])
    mean = patient_df.mean().to_numpy()
    mins = patient_df.min().to_numpy()
    maxs = patient_df.max().to_numpy()
    patient_vector = np.concatenate((mean, mins, maxs), axis=0)
    patient_series = pd.Series(patient_vector).fillna(-1)
    return patient_series


def main():
    # load the pretrained model from the training phase
    rfc = pickle.load(open('RF_Classifier.pkl', 'rb'))
    filepath = str(sys.argv[1])

    # patients_dict = {}
    # for filename in os.listdir(filepath):
    #     if filename.endswith(".psv"):
    #         with open(filepath + filename) as openfile:
    #             patient = filename.split("_")[1]
    #             patient = patient.split(".")[0]
    #             df = pd.read_csv(openfile, sep="|")
    #             patients_dict[patient] = df
    #
    # processed_patients_dict = {}
    # patients_list = []
    # for k, v in patients_dict.items():
    #     for index, row in v.iterrows():
    #         if row['SepsisLabel'] == 1:
    #             processed_patients_dict[k] = v[:index + 1]
    #             processed_patients_dict[k]['SepsisLabel'] = 1
    #             processed_patients_dict[k]['patient_id'] = k
    #             patients_list.append(processed_patients_dict[k])
    #             break
    #     if k not in processed_patients_dict.keys():
    #         processed_patients_dict[k] = v
    #         processed_patients_dict[k]['patient_id'] = k
    #         patients_list.append(processed_patients_dict[k])
    #
    # df = pd.concat(patients_list, axis=0, ignore_index=True)

    agg_df = pd.DataFrame()
    for patient_file in os.listdir(filepath):
        patient_df = pd.read_csv(os.path.join(filepath, patient_file), sep='|')
        patient_df = patient_df.drop(
            columns=['EtCO2', 'Temp', 'Bilirubin_direct', 'Fibrinogen', 'TroponinI', 'Hct', 'HCO3', 'DBP'])
        if 1 in patient_df['SepsisLabel'].unique():
            first_row_sepsis = patient_df[patient_df['SepsisLabel'] == 1].iloc[0].name
            patient_df = patient_df.iloc[:first_row_sepsis + 1]
        patient_series = feature_agg(patient_df)
        is_sick = 1 if 1 in patient_df['SepsisLabel'].unique() else 0
        patient_series['PatientLabel'] = is_sick
        patient_series['PatientId'] = int(patient_file.split('.')[0].split('_')[1])
        agg_df = agg_df.append(patient_series, ignore_index=True)

    patient_id = agg_df['PatientId'].astype("int")

    # The columns were selected using SelectKBest
    cols = [1,  2,  3, 13, 20, 22, 31, 33, 34, 35, 45, 50, 52, 54, 65, 66, 67, 84, 86, 95]

    X_test = agg_df.iloc[:, cols]
    Y_test = agg_df.iloc[:, -1]

    rfc_pred = rfc.predict(X_test)
    rfc_pred = [int(x) for x in rfc_pred]
    print(f'Test set scores of the model is:')
    f1 = f1_score(Y_test, rfc_pred, average='binary')
    print(f"F1 score {f1}")
    acc = accuracy_score(Y_test, rfc_pred)
    print(f"Accuracy score {acc}")
    roc_auc = roc_auc_score(Y_test, rfc.predict_proba(X_test)[:, 1])
    print(f"ROC-AUC score {roc_auc}")
    predict_df = pd.DataFrame(zip(patient_id, rfc_pred), columns=['Id', 'SepsisLabel'])
    predict_df.to_csv('prediction.csv', index=False, header=False)


if __name__ == '__main__':
    main()
