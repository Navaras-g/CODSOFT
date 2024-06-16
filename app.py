import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import streamlit as st

# Loading the titanic dataset, no need for the file path as it's in the same folder
df = pd.read_csv("Titanic-Dataset.csv")

# Dropping the unnecessary columns
df.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis='columns', inplace=True)

# Map 'Sex' column to numerical values i.e. 0 and 1 for male and female respectively
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Separating the target variable
target = df['Survived']
inputs = df.drop('Survived', axis='columns')

# Filling the  missing values in 'Age' column with the average of the data
inputs['Age'] = inputs['Age'].fillna(inputs['Age'].mean())

# Splitting the data into training and testing sets, where input variables will be used to predict the target variable
x_train, x_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2, random_state=42)

# Initializing and training the model
model = GaussianNB()
model.fit(x_train, y_train)
print(model.score(x_test,y_test))

# Streamlit app
st.title('Titanic Survival Prediction')

# User inputs for prediction
in_class = st.number_input("Enter Pclass:", min_value=1, max_value=3)
sex = st.selectbox("Enter Sex:", options=['male', 'female'])
in_sex = 0 if sex == 'male' else 1
in_age = st.number_input("Enter Age:", min_value=0.0, max_value=100.0)
in_fare = st.number_input("Enter Fare:", min_value=0.0)

# Input data for prediction
tests = pd.DataFrame({
    'Pclass': [in_class],
    'Sex': [in_sex],
    'Age': [in_age],
    'Fare': [in_fare]
})

# Predicting the result
if st.button("Prediction"):
    prediction = model.predict(tests)[0]
    if prediction == 0:
        st.subheader("NOT SURVIVED")
    else:
        st.subheader("SURVIVED")

# Printing model accuracy
accuracy = model.score(x_test, y_test)
st.write(f"Model Accuracy: {accuracy:.2f}")
