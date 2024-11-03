import yaml
import pandas as pd
import numpy as np
from joblib import dump
import pickle
import os

# Plotting Libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
import mlflow.pyfunc

# Setting Parent Folder

current_directory = os.getcwd()
print(current_directory)

print('Libraries loaded')

with open('params.yaml') as conf_file:
    config = yaml.safe_load(conf_file)

data = pd.read_csv(config['data']['input_data'])

#Separa las variables del dataframe
X = data.drop(['date', 'Load_Type'],axis=1)
y = data['Load_Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = config['train']['test_size'], random_state = config['train']['random_state'])

# Definir variables numéricas y categóricas
numeric_features = ['Usage_kWh', 'Lagging_Current_Reactive.Power_kVarh', 'Leading_Current_Reactive_Power_kVarh',
                    'CO2(tCO2)', 'Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor', 'NSM']
categorical_features = ['WeekStatus', 'Day_of_week']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95))
        ]), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

rfc_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

rfc_model.fit(X_train, y_train)

# Evaluar el modelo en el conjunto de prueba
y_pred = rfc_model.predict(X_test)

# Calcular métricas de evaluación
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Guardar el pipeline en un archivo local
with open('./models/rfc_model.pkl', 'wb') as file:
    pickle.dump(rfc_model, file)
