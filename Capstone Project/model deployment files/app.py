import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load the model pickle file for age prediction
model_age = pickle.load(open('model_age_sc01.pkl', 'rb'))

# Load the model pickle file for gender prediction
model_gender = pickle.load(open('model_gender_sc01.pkl', 'rb'))

# Load the test.csv with 50 device_ids
df_test_50 = pd.read_csv('Test_csv_50.csv')
df_age_data = df_test_50.drop(['Unnamed: 0'],axis=1)

gender_result = []
age_result = []

for index, row in df_test_50.iterrows():
    gender_result.append(model_gender.predict_proba([[row['Device_id'],row['Group'], row['event_id'], row['Total_events'], row['Avg_events'], row['Active Hour'], row['Total_places_visited'], row['is_installed'], row['is_active'], row['category'],row['cluster_labels']]])[0][0])
df_test_50['gender_probas'] = gender_result

for index, row in df_age_data.iterrows():
    age_result.append(model_age.predict(np.asarray([[row['Device_id'],row['Group'], row['event_id'], row['Total_events'], row['Avg_events'], row['Active Hour'], row['Total_places_visited'], row['is_installed'], row['is_active'], row['category'],row['cluster_labels']]])))
df_age_data['age'] = age_result

# Create final output dataframe
df_ouput = pd.DataFrame()
df_ouput['Device_id'] = df_test_50['Device_id']
df_ouput['Gender_predicted_probas'] = df_test_50['gender_probas']
df_ouput['Age_predicted'] = df_age_data['age']

# Since we have done modelling on subset of data we got maximum probability as 0.617210 for the bottom decile.
# Hence we are just using the best threshold identified via KS statistic to predict gender
def gender_final (row):
    # As per KS Statistic we get 0.379930 as best probability
    # Since we are getting the 0th class probability which is for Female in our case below is the condition
    if (row['Gender_predicted_probas'] > 0.379930):
        row['Gender'] = 'F'
    else:
        row['Gender'] = 'M'
    return row

def gender_campaign (row):
    # As per KS Statistic we get 0.379930 as best probability
    if (row['Gender'] == 'F'):
        row['Selected campaign (Based on Gender)'] = 'Campaign 1 and Campaign 2'
    else:
        row['Selected campaign (Based on Gender)'] = 'Campaign 3'
    return row

def age_campaign (row):
    # As per KS Statistic we get 0.379930 as best probability
    age = int(str(row['Age_predicted'])[1:-1])
    if(age <= 24):
        row['Selected campaign (Based on Age)'] = 'Campaign 4'
    elif((age >= 24) &(age <= 32)):
        row['Selected campaign (Based on Age)'] = 'Campaign 5'
    elif(age >= 32):
        row['Selected campaign (Based on Age)'] = 'Campaign 6'
    return row

df_ouput = df_ouput.apply(lambda row: gender_final(row), axis=1)
df_ouput = df_ouput.apply(lambda row: gender_campaign(row), axis=1)
df_ouput = df_ouput.apply(lambda row: age_campaign(row), axis=1)

@app.route('/')
def home():
    return render_template('index.html',tables=[df_ouput.to_html(classes='data')], titles=df_ouput.columns.values)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
