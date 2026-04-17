import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'Artificial_Neural_Network_Case_Study_data.csv')

df = pd.read_csv(csv_path)

X = df[['CreditScore','Age','Balance','Tenure','EstimatedSalary']]
y = df['Exited']

# Scaling (VERY IMPORTANT for ANN)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ANN Model
model = MLPClassifier(hidden_layer_sizes=(10,10), max_iter=500)
model.fit(X_train, y_train)

# Save model + scaler
pickle.dump(model, open('model.pkl','wb'))
pickle.dump(scaler, open('scaler.pkl','wb'))

print("ANN Model created!")