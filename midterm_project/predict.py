# load the model
import pickle
import pandas as pd

df = pd.read_csv(r"C:\Users\pc\Downloads\archive (10)\heart.csv")

# lr = LogisticRegression
with open('heart_model.pkl', 'rb') as file:
    lr = pickle.load(file)

# Use it for predictions
# predictions = loaded_model.predict(X_test)

# patient_record_df = df.iloc[[9]]  

# print(patient_record_df)


patient = {
    'age': 54,
    'sex': '1',
    'cp': 0,
    'trestbps': 122,
    'chol': 286,
    'fbs': 0,
    'restecg': 0,
    'thalach': 116,
    'exang': 1,
    'oldpeak': 3.2,
    'slope': 1,
    'ca': 2,
    'thal': 2,
    'target': 0
}


from sklearn.feature_extraction import DictVectorizer

def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)

    return dv, model

from sklearn.feature_extraction import DictVectorizer

# Create and fit the DictVectorizer
dv = DictVectorizer(sparse=False)
dicts = df.drop('target', axis=1).to_dict(orient='records')  # assuming 'target' is your label
X = dv.fit_transform(dicts)  # Fit and transform training data

# Now you can transform a new patient's record
patient = dicts[9]  # 10th patient
X_patient = dv.transform([patient])

X_patient


lr.predict_proba(X)

lr.predict_proba(X)[0,1]

print('input', patient)
print('predict_proba',X_patient)

# Predict probability
probability = lr.predict_proba(X_patient)[0, 1]

# Output
print("Patient input:", patient)
print("Predicted probability of heart disease:", probability)
