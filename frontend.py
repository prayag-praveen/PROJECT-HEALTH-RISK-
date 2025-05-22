import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder


import streamlit as st  # ✅ import first




# # This MUST be the first Streamlit command
st.set_page_config(page_title="Future Health Risk Prediction", page_icon="🩺", layout="centered")


# Now you can safely use other Streamlit functions
st.title("Welcome to Health Risk Predictor")

# Add background image
# Set background image using custom CSS
def add_bg_from_local(image_file):
    with open(image_file, "rb") as img_file:
        import base64
        encoded = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("INSHOT 2.jpg")


# Load your trained model and label encoder
model = pickle.load(open('your_saved_model.pkl', 'rb'))
lab1 = pickle.load(open('Alcohol_Consumption_encoder.pkl', 'rb'))  # Load the LabelEncoder that was fitted during training
lab2 = pickle.load(open('Allergies_encoder.pkl', 'rb'))  # Load the LabelEncoder that was fitted during training
lab3 = pickle.load(open('Dietary_Habits_encoder.pkl', 'rb'))  # Load the LabelEncoder that was fitted during training
lab4 = pickle.load(open('Gender_encoder.pkl', 'rb'))  # Load the LabelEncoder that was fitted during training
lab5 = pickle.load(open('Genetic_Risk_Factor_encoder.pkl', 'rb'))  # Load the LabelEncoder that was fitted during training
lab6 = pickle.load(open('Smoking_Habit_encoder.pkl', 'rb'))  # Load the LabelEncoder that was fitted during training

# Set page title
# st.set_page_config(page_title="Future Health Risk Prediction", page_icon="🩺", layout="centered")
# st.title('🩺 Future Health Risk Prediction System')
st.write('Fill the details below carefully to predict your health risk and get personalized advice!')



st.header('📝 Enter Your Health Details')

age = st.number_input('Age', min_value=1, max_value=120, value=30)
gender = st.selectbox('Gender', ['Male', 'Female','Other'])
weight = st.number_input('Weight (kg)', min_value=30, max_value=200, value=70)
bmi = st.number_input('BMI', min_value=10.0, max_value=60.0, value=22.0)
bp_sys = st.number_input('Systolic Blood Pressure (mmHg)', min_value=80, max_value=200, value=120)
bp_dia = st.number_input('Diastolic Blood Pressure (mmHg)', min_value=50, max_value=130, value=80)
cholesterol = st.number_input('Cholesterol Level (mg/dL)', min_value=100, max_value=400, value=180)
sugar = st.number_input('Blood Sugar Level (mg/dL)', min_value=50, max_value=300, value=100)
exercise = st.slider('Exercise Frequency (days per week)', min_value=0, max_value=7, value=3)
sleep = st.slider('Average Sleep Hours', min_value=1, max_value=12, value=7)
genetic = st.selectbox('Genetic Risk Factor', ['No', 'Yes'])
allergies = st.selectbox('Allergies', ['No', 'Yes'])
alcohol = st.selectbox('Alcohol Consumption', ['None', 'Occasional', 'Regular'])
smoking = st.selectbox('Smoking Habit', ['None', 'Occasional', 'Regular'])
diet = st.selectbox('Dietary Habits', ['Healthy', 'Unhealthy'])

# User Input Dictionary
user_input = {
    'Age': age,
    'Gender': gender,
    'Weight_kg': weight,
    'BMI': bmi,
    'Blood_Pressure_Systolic': bp_sys,
    'Blood_Pressure_Diastolic': bp_dia,
    'Cholesterol_Level': cholesterol,
    'Blood_Sugar_Level': sugar,
    'Exercise_Frequency': exercise,
    'Sleep_Hours': sleep,
    'Genetic_Risk_Factor': genetic,
    'Allergies': allergies,
    'Alcohol_Consumption': alcohol,
    'Smoking_Habit': smoking,
    'Dietary_Habits': diet
}

# Function to preprocess user inputs with LabelEncoder for categorical variables
# def preprocess_input(user_input):
#     # Apply each specific LabelEncoder
#     gender_encoded = lab4.transform([user_input['Gender']])[0]
#     genetic_encoded = lab5.transform([user_input['Genetic_Risk_Factor']])[0]
#     allergies_encoded = lab2.transform([user_input['Allergies']])[0]
#     alcohol_encoded = lab1.transform([user_input['Alcohol_Consumption']])[0]
#     smoking_encoded = lab6.transform([user_input['Smoking_Habit']])[0]
#     diet_encoded = lab3.transform([user_input['Dietary_Habits']])[0]

#     return np.array([[  # Ensure order matches model training
#         user_input['Age'],
#         gender_encoded,
#         user_input['Weight_kg'],
#         user_input['BMI'],
#         user_input['Blood_Pressure_Systolic'],
#         user_input['Blood_Pressure_Diastolic'],
#         user_input['Cholesterol_Level'],
#         user_input['Blood_Sugar_Level'],
#         genetic_encoded,
#         allergies_encoded,
#         user_input['Exercise_Frequency'],
#         user_input['Sleep_Hours'],
#         alcohol_encoded,
#         smoking_encoded,
#         diet_encoded
#     ]])

def preprocess_input(user_input):
    # Ensure all categorical inputs are strings
    gender_encoded = lab4.fit_transform([str(user_input['Gender'])])[0]
    genetic_encoded = lab5.fit_transform([str(user_input['Genetic_Risk_Factor'])])[0]
    allergies_encoded = lab2.fit_transform([str(user_input['Allergies'])])[0]
    alcohol_encoded = lab1.fit_transform([str(user_input['Alcohol_Consumption'])])[0]
    smoking_encoded = lab6.fit_transform([str(user_input['Smoking_Habit'])])[0]
    diet_encoded = lab3.fit_transform([str(user_input['Dietary_Habits'])])[0]

    return np.array([[  # Ensure order matches model training
        user_input['Age'],
        gender_encoded,
        user_input['Weight_kg'],
        user_input['BMI'],
        user_input['Blood_Pressure_Systolic'],
        user_input['Blood_Pressure_Diastolic'],
        user_input['Cholesterol_Level'],
        user_input['Blood_Sugar_Level'],
        genetic_encoded,
        allergies_encoded,
        user_input['Exercise_Frequency'],
        user_input['Sleep_Hours'],
        alcohol_encoded,
        smoking_encoded,
        diet_encoded
    ]])

# Predict button
if st.button('🔎 Predict Health Risk'):
    input_data = preprocess_input(user_input)
    prediction = model.predict(input_data)[0]

    # Display prediction
    st.subheader('🩺 Prediction Result:')
    if prediction == 'High':
        st.error('🚨 High Health Risk Detected! Immediate medical attention and lifestyle changes needed.')
    elif prediction == 'Medium':
        st.warning('⚠️ Medium Health Risk. Start improving lifestyle habits.')
    else:
        st.success('✅ Low Health Risk. Keep up your healthy lifestyle!')


    # Personalized Recommendations
    st.subheader('📋 Personalized Health Recommendations:')
    recommendations = []


    if prediction == 'High':
        recommendations.append("⚠️ High Risk: Immediate lifestyle changes recommended.")
        if user_input['Smoking_Habit'] != 'None':
            recommendations.append("- 🚭 Consider quitting smoking.")
        if user_input['Alcohol_Consumption'] != 'None':
            recommendations.append("- 🍷 Reduce or eliminate alcohol consumption.")
        if user_input['Dietary_Habits'] == 'Unhealthy':
            recommendations.append("- 🥦 Improve your diet: add more vegetables, fruits, and reduce junk food.")
        if user_input['Allergies'] == 'Yes':
            recommendations.append("- 🤧 Avoid known allergens and consult an allergist if needed.")
        if user_input['Genetic_Risk_Factor'] == 'Yes':
            recommendations.append("- 🧬 Regular screenings may help detect issues early due to genetic risks.")

    elif prediction == 'Medium':
        recommendations.append("⚠️ Medium Risk: You need to be cautious. Adjust your habits.")
        if user_input['Dietary_Habits'] == 'Unhealthy':
            recommendations.append("- 🥦 Improve your diet.")
        if user_input['Smoking_Habit'] != 'None':
            recommendations.append("- 🚭 Reduce or quit smoking.")
        if user_input['Alcohol_Consumption'] != 'None':
            recommendations.append("- 🍷 Reduce alcohol consumption.")

    else:
        recommendations.append("✅ Low Risk: Maintain your healthy lifestyle!")
        recommendations.append("- 🏋️‍♂️ Stay active and exercise regularly.")
        recommendations.append("- 🥑 Keep eating a balanced diet.")
        recommendations.append("- 🛌 Maintain good sleep hygiene.")

    for rec in recommendations:
        st.write(rec)
