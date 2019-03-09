import pandas as pd
import datetime
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import f1_score, mean_squared_error


def get_f1_score(clf, X_train, X_test, y_train, y_test):
    y_predict = clf.predict(X_train)

    result = f1_score(y_train, y_predict)
    print(result)

    print('-------training set---------')
    print()

    y_predict = clf.predict(X_test)
    result = f1_score(y_test, y_predict)
    print(result)

    print('-------test set---------')
    print()


file_path = 'data/AI Dataset.csv'
# data_df = pd.read_csv(file_path, encoding='latin1',header=0, index_col=0)
data_df = pd.read_csv(file_path, encoding='latin1')

data_df = data_df[['Project Geographic District ', 'Project School Name', 'Project Type ', 'Project Phase Actual Start Date','Project Phase Planned End Date','Project Phase Actual End Date']]
# training_data_df = data_df.filter(items=[
#             'Project Geographic District' ,'Project Building Identifier','Project School Name'
#     ,'Project Type' ,'Project Phase Name','Project Status Name','Project Phase Actual Start Date'
#     ,'Project Phase Planned End Date','Project Phase Actual End Date','Project Budget Amount','Final Estimate of Actual Costs Through End of Phase Amount',
#     'Total Phase Actual Spending Amount','DSF Number(s)'
#         ])

# data_df = data_df.filter(items=[
#             'Project Geographic District ', 'Project Type ', 'Project Phase Actual Start Date'
#         ])

# print(data_df[['Project Geographic District ', 'Project School Name', 'Project Type ', 'Project Phase Actual Start Date','Project Phase Planned End Date','Project Phase Actual End Date']])
data_df['pass_timeline'] = np.nan
data_df['concurrent_projects_num'] = 0


school_dict = {}
district_dict = {}
type_dict = {}
concurrent_projects_num_dict = {}

for index, row in data_df.iterrows():
    try:
        start_date = datetime.strptime(row[3], '%m/%d/%Y')
        data_df.at[index, 'Project Phase Actual Start Date'] = start_date
        # date_dt2 = datetime.strptime(row[4], '%m/%d/%Y')
    except:
        data_df.at[index, 'Project Phase Actual Start Date'] = np.nan

    try:
        planned_end_date = datetime.strptime(row[4], '%m/%d/%Y')
        data_df.at[index, 'Project Phase Planned End Date'] = planned_end_date
        # date_dt2 = datetime.strptime(row[4], '%m/%d/%Y')
    except:
        data_df.at[index, 'Project Phase Planned End Date'] = np.nan
        planned_end_date = None

    try:
        actual_end_date = datetime.strptime(row[5], '%m/%d/%Y')
        data_df.at[index, 'Project Phase Actual End Date'] = actual_end_date
        # date_dt2 = datetime.strptime(row[4], '%m/%d/%Y')
    except:
        data_df.at[index, 'Project Phase Actual End Date'] = np.nan
        actual_end_date = None


    if row[1] not in school_dict:
        school_dict[row[1]] = len(school_dict)

    if row[0] not in district_dict:
        district_dict[row[0]] = len(district_dict)

    if row[2] not in type_dict:
        type_dict[row[2]] = len(type_dict)

    if planned_end_date is None and actual_end_date is not None:
        data_df.at[index, 'pass_timeline'] = 0
    elif planned_end_date is None and actual_end_date is None:
        data_df.at[index, 'pass_timeline'] = np.nan
    elif planned_end_date is not None and actual_end_date is None:
        data_df.at[index, 'pass_timeline'] = 1
    elif planned_end_date > actual_end_date:
        data_df.at[index, 'pass_timeline'] = 0
    else:
        data_df.at[index, 'pass_timeline'] = 1



data_df = data_df.dropna(subset=['pass_timeline', 'Project Phase Actual Start Date','Project Phase Planned End Date','Project Phase Actual End Date'])
# data_df['pass_timeline'].astype(int)
# print(data_df)


for index, row in data_df.iterrows():
    concurrent_projects = data_df.loc[
        (data_df['Project School Name'] == row[1]) & (data_df['Project Phase Actual Start Date'] >= row[3]) & (
            data_df['Project Phase Actual Start Date'] <= row[5])]
    concurrent_pj_num = concurrent_projects.shape[0] - 1
    data_df.at[index, 'concurrent_projects_num'] = concurrent_pj_num
    if concurrent_projects.shape[0] - 1 not in concurrent_projects_num_dict:
        concurrent_projects_num_dict[concurrent_pj_num] = concurrent_pj_num


features_set =[]
label_set = []

for index, row in data_df.iterrows():


    district_id_vector = [0] * (len(district_dict) + 1)
    if row[0] not in district_dict:
        district_id_vector[len(district_dict) + 1] = 1
    else:
        district_id_vector[district_dict[row[0]]] = 1

    school_dict_id_vector = [0] * (len(school_dict) + 1)
    if row[1] not in school_dict:
        school_dict_id_vector[len(school_dict) + 1] = 1
    else:
        school_dict_id_vector[school_dict[row[1]]] = 1

    type_dict_id_vector = [0] * (len(type_dict) + 1)
    if row[2] not in type_dict:
        type_dict_id_vector[len(type_dict) + 1] = 1
    else:
        type_dict_id_vector[type_dict[row[2]]] = 1

    concurrent_projects_num_vector = [0] * (len(concurrent_projects_num_dict) + 1)
    if row[7] not in concurrent_projects_num_dict:
        concurrent_projects_num_vector[len(concurrent_projects_num_dict) + 1] = 1
    else:
        concurrent_projects_num_vector[concurrent_projects_num_vector[row[7]]] = 1




    # print(int(row[6]))
    features_set.append(district_id_vector + school_dict_id_vector + type_dict_id_vector + concurrent_projects_num_vector)
    label_set.append(int(row[6]))

X_train, X_test, y_train, y_test = train_test_split(features_set, label_set, test_size=0.2, random_state=40)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

get_f1_score(clf, X_train, X_test, y_train, y_test)