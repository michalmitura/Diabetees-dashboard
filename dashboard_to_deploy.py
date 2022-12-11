import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import pickle
import matplotlib.pyplot as plt
import joblib
import numpy as np
import scikit-learn

st.set_page_config(layout='wide')

introduction = st.container()
overview = st.container()
feature_importances = st.container()
models = st.container()
predictions = st.container()

@st.cache()
def load_data():
    data = pd.read_csv('dataset_zadanie.csv')
    return data

def load_models_results():
    data = pd.read_csv('model_results/final_model_results.csv').drop('Unnamed: 0', axis=1)
    return data

def load_model(selected_model):
    models_dict = {'Decision Tree Classifier': 'DecisionTreeClassifier()', 
    'Gradient Boosting Classifier': 'GradientBoostingClassifier()',
    'Logistic Regression': 'LogisticRegression(max_iter=1000)', 
    'Random Forest Classifier': 'RandomForestClassifier()', 
    'Neural Network Classifier': 'NN_model',
    'Support Vectors Classifier': 'SVC()', 
    'k-Neighbors Classifier': 'KNeighborsClassifier()'}
    model_name = models_dict[selected_model]
    model_file = open(f'models/{model_name}.pickle', mode='rb')
    model = pickle.load(file=model_file)
    model_file.close()
    return model

def load_scaler():
    file = open(f'models/nn_scaler.pickle', mode='rb')
    scaler = pickle.load(file=file)
    file.close()
    return scaler

def load_confusion_matrix(model):
    file_name = f'model_results/confusion_matrices/{model}_confusion_matrix.csv'
    df = pd.read_csv(file_name)
    df.columns = ['', 'Predicted Negative', 'Predicted Positive']
    return df

def generate_first_plot(feature):
    plot = px.histogram(data_frame=data, x=feature, marginal='box', title=f'Distribution of {selected_feature} in sample',
    color_discrete_sequence=['green'])
    plot.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'},
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
    plot.update_layout(height=900)
    return plot

def generate_second_plot(feature):
    plot = px.histogram(data_frame = data, x=feature, color='Outcome', marginal='box', 
    title=f'Distribution of {selected_feature} for healthy patients (0) vs. diabetees (1)',
    histnorm='percent', color_discrete_sequence=['red', 'green'])
    plot.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'},
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),
            barmode='overlay')
    plot.update_layout(height=500)
    plot.update_traces(opacity=0.75)
    return plot

def load_feature_importance(selected_model):
    model_dict = {'Decision Tree Classifier': 'DecisionTreeClassifier',
    'Gradient Boosting Classifier': 'GradientBoostingClassifier',
    'Logistic Regression': 'LogisticRegression',
    'Random Forest Classifier': 'RandomForestClassifier'}
    model = model_dict[selected_model]
    file_name = f'model_results/feature_importances/{model}()_feature_importances.csv'
    data = pd.read_csv(file_name)
    data.columns = ['Feature', 'Weight']
    return data

def generate_feature_importances_chart(feature_importance_data):
    plot = px.bar(data_frame=feature_importance_data, x='Feature', y='Weight')
    plot.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'},
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),
            barmode='overlay')
    return plot

def predict(data, model):
    if model != 'Neural Network Classifier':
        loaded_model = load_model(selected_model=model)
        prediction_num = loaded_model.predict(data)
        if prediction_num == 1:
            prediction = 'Patient has diabetes'
        else:
            prediction = "Patient doesn't have diabetes" 
        return prediction
    else:
        loaded_model = load_model(selected_model=model)
        scaler = load_scaler()
        X = scaler.transform(input_data)
        prediction_num = loaded_model.predict(X)
        if prediction_num > 0.5:
            prediction = 'Patient has diabetes'
        else:
             prediction = "Patient doesn't have diabetes"   
        return prediction 
            
        # if prediction > 0.5:
        #     class_prediction = pd.DataFrame(data=[1], columns=[0])
            
        # else:
        #     class_prediction = pd.DataFrame(data=[0], columns=[0])
        # return class_prediction

data = load_data()


with introduction:
    st.header('Introduction')
    st.write('Below dashboard presents data from National Institute of Diabetes and Digestive and Kidney Diseases. In the first part I included'
    'the analysis of distribution of features in sample and comparison of feature distribution between'
    'healthy patients and patients with diabetees. In second part the feature importances for all models used in the task can be found. '
    'In next part of dashboard, performance of all models was shown along with confusion matrices. In last part of the dashboard,'
    ' I implemented several ML models which predict whether patient has diabetees or not based on given parameters.')

with overview:
    st.header('Dataset overview')
    st.write(data)
    
    selected_feature = st.selectbox('Select feature', options=data.drop('Outcome', axis=1).columns.tolist())
    plot1 = generate_first_plot(feature=selected_feature)
    plot2 = generate_second_plot(feature=selected_feature)

    col1, col2 = st.columns(2)

    with col1:
        st.write(plot1)
    with col2:
        st.write(plot2)
    
    st.write(
        'In terms of pregnancies, the positive correlation between percentage of women diagnosed with diabetees and number of pregnancies can be seen.'
        '\n High glucose concentration also was a conducive factor for diabetees, regarding blood pressure the difference between distributions in healthy patients and diabetees '
        "groups wasn't as significant."
        "\n Among patients with higher skin thickness, higher percent of diabetees can be seen, but it doesn't seem to be a decisive factor."
        "\n\n Analyzing distribution of insulin concentration, higher levels was usually associated with presence of disease. Starting at 120, percentage of diabetees tend to be two times higher than healthy people at given insulin level"
        "\n\n In terms of BMI, strong presence of diabetees was seen at the higher values of BMI. The distribution of BMI was moved to the right by 5 on average"
        "\n\n Diabetees Pedigree Function tends to be higher for the womens with diagnosed diabetees. It indicates high importance of this feature for diabetees diagnosis."
        " Analyzing distribution of age, older age groups tend to be more affected with the disease, starting around age of 30."
    )
    
with feature_importances:
    st.header('Features importances')

    col3, col4 = st.columns(2)

    with col3:
        selected_model_features = st.selectbox('Choose model', options=['Decision Tree Classifier', 'Gradient Boosting Classifier',
        'Logistic Regression', 'Random Forest Classifier'])    
        feature_importances= load_feature_importance(selected_model = selected_model_features)
        st.write(feature_importances)
    with col4:
        feature_importances_plot = generate_feature_importances_chart(feature_importance_data=feature_importances)
        st.write(feature_importances_plot)
    st.write(
        'Above table and chart shows feature weights for models from which these could be obtained. In all models except Logistic Regression'
        ' glucose level was the most important factor for predicting diabetees.'
    )


with models:
    st.header('Models')
    col5, col6 = st.columns(2)

    with col5:
        st.subheader('Models performance')
        models_performance = load_models_results()
        st.write(models_performance)

    with col6:
        st.subheader('Confusion matrices')
        selected_model_performance = st.selectbox('Choose model', options=['Decision Tree Classifier', 'Gradient Boosting Classifier',
            'Logistic Regression', 'Random Forest Classifier'], key='model_performance')    
        confusion_matrix = load_confusion_matrix(model=selected_model_performance)
        st.write(confusion_matrix)
  

with predictions:
    st.header('Predictions')
    col7, col8, col9 = st.columns(3)

    with col7:
        selected_model = st.selectbox('Choose model', options=[
            'Decision Tree Classifier', 'Gradient Boosting Classifier',
            'Logistic Regression', 'Random Forest Classifier', 'Neural Network Classifier',
            'Support Vectors Classifier', 'k-Neighbors Classifier' 
            ], key='predictions_selectbox')   
        

        selected_pregnancies = st.number_input('Choose number of pregnancies', min_value=0, step=1)
        selected_glucose = st.number_input('Choose glucose concentration', min_value=0, max_value=200, step=1)
    with col8:
        selected_blood_pressure = st.number_input('Choose blood pressure', min_value=0, max_value=200, step=1)
        selected_skin_thickness = st.number_input('Choose skin thickness', min_value=0, max_value=100, step=1)
        selected_insulin = st.number_input('Choose insulin concentration', min_value=0, max_value=1000, step=1)
    with col9:
        selected_bmi = st.number_input('Choose BMI', min_value=0.00, max_value=100.00, step=0.1)
        selected_pedrigree = st.number_input('Choose Diabetes Pedigree Function value', min_value=0.00)
        selected_age = st.number_input('Choose age', min_value=0, max_value=100, step=1)

    input_data = pd.DataFrame(data=[selected_pregnancies, selected_glucose, selected_blood_pressure,
    selected_skin_thickness, selected_insulin, selected_bmi, selected_pedrigree, selected_age]).T
    input_data.columns = data.drop('Outcome', axis=1).columns.tolist()
    st.write(input_data)

    prediction = predict(data=input_data, model=selected_model)
    st.write(prediction)
