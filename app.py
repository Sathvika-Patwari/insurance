import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Load data
file_path = 'Insurance_Claims_Analysis.xlsx'  # Modify this path as needed
xls = pd.ExcelFile(file_path)
raw_data = pd.read_excel(xls, sheet_name='Raw Data')

# Data Cleaning
raw_data['submission_date'] = pd.to_datetime(raw_data['submission_date'], errors='coerce')
raw_data['completion_date'] = pd.to_datetime(raw_data['completion_date'], errors='coerce')
raw_data['days_to_payment'] = raw_data['days_to_payment'].fillna(raw_data['days_to_payment'].median())
raw_data['insurance_paid'] = raw_data['insurance_paid'].fillna(0)
raw_data['status'] = raw_data['status'].fillna("Unknown")
raw_data = raw_data.dropna(subset=['claim_id', 'patient_id'])

# Encoding categorical columns
label_encoder = LabelEncoder()
raw_data['insurance_carrier_encoded'] = label_encoder.fit_transform(raw_data['insurance_carrier'])
raw_data['procedure_type_encoded'] = label_encoder.fit_transform(raw_data['procedure_type'])
raw_data['status_encoded'] = label_encoder.fit_transform(raw_data['status'])

# Feature selection for prediction
features = ['claim_amount', 'insurance_carrier_encoded', 'procedure_type_encoded', 'processing_time']
target = 'days_to_payment'

X = raw_data[features]
y = raw_data[target]

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Streamlit Dashboard Layout
st.title('Dental Hospital Insurance Claims Analysis Dashboard')

# Fancy Navigation Bar (using st.markdown for clickable links)
st.markdown("""
    <style>
        .navbar {
            display: flex;
            justify-content: center;
            background-color: #1e90ff;
            padding: 15px;
            font-family: 'Arial', sans-serif;
            border-radius: 10px;
        }
        .navbar a {
            color: white;
            padding: 15px 25px;
            text-decoration: none;
            text-align: center;
            font-size: 18px;
            transition: 0.3s;
        }
        .navbar a:hover {
            background-color: #ddd;
            color: black;
        }
        .content-section {
            padding-top: 40px;
        }
        .sidebar {
            background-color: #f4f4f4;
            padding: 20px;
            border-radius: 10px;
        }
    </style>
    <div class="navbar">
        <a href="#analysis">Analysis</a>
        <a href="#prediction">Prediction</a>
    </div>
""", unsafe_allow_html=True)

# Analysis Section
st.markdown("<div class='content-section' id='analysis'></div>", unsafe_allow_html=True)
st.subheader("Reimbursement Delay Analysis")

# Create 2 columns for side-by-side plots
col1, col2 = st.columns(2)

with col1:
    # 1. Claim Amount Distribution with Plotly (Interactive)
    fig = px.histogram(raw_data, x='days_to_payment', nbins=30, title="Reimbursement Delay Distribution", labels={'days_to_payment': 'Days to Payment'})
    st.plotly_chart(fig)

with col2:
    # 2. Processing Time vs Claim Amount (Interactive Scatter Plot)
    fig = px.scatter(raw_data, x='claim_amount', y='days_to_payment', color='status', title="Processing Time vs Claim Amount", labels={'claim_amount': 'Claim Amount', 'days_to_payment': 'Days to Payment'})
    st.plotly_chart(fig)

# Create another row for additional plots
col3, col4 = st.columns(2)

with col3:
    # 3. Days to Payment vs Insurance Carrier (Box Plot)
    fig = px.box(raw_data, x='insurance_carrier', y='days_to_payment', title="Days to Payment by Insurance Carrier", labels={'insurance_carrier': 'Insurance Carrier', 'days_to_payment': 'Days to Payment'})
    st.plotly_chart(fig)

with col4:
    # 4. Correlation Heatmap (Interactive)
    numeric_data = raw_data.select_dtypes(include=[np.number])
    correlation_matrix = numeric_data.corr()
    fig = px.imshow(correlation_matrix, color_continuous_scale='Blues', title="Correlation Heatmap")
    st.plotly_chart(fig)

# 5. Claim Status Distribution Pie Chart
st.subheader("Claim Status Distribution")
claim_status_counts = raw_data['status'].value_counts()
fig = px.pie(claim_status_counts, names=claim_status_counts.index, values=claim_status_counts.values, title="Claim Status Distribution")
st.plotly_chart(fig)

# Prediction Section
st.markdown("<div class='content-section' id='prediction'></div>", unsafe_allow_html=True)
st.subheader("Predict Reimbursement Delay")

# User input form for prediction
claim_amount = st.number_input('Claim Amount', min_value=0, value=1000)
processing_time = st.number_input('Processing Time (Days)', min_value=1, value=30)
insurance_carrier = st.selectbox('Insurance Carrier', raw_data['insurance_carrier'].unique())
procedure_type = st.selectbox('Procedure Type', raw_data['procedure_type'].unique())

# Encode user inputs with error handling for unseen labels
def safe_transform(label_encoder, label):
    try:
        return label_encoder.transform([label])[0]
    except ValueError:
        # Handle unseen label by assigning a default value (e.g., -1 for unknown)
        return -1

insurance_carrier_encoded = safe_transform(label_encoder, insurance_carrier)
procedure_type_encoded = safe_transform(label_encoder, procedure_type)

# Prepare input for prediction
input_features = np.array([[claim_amount, insurance_carrier_encoded, procedure_type_encoded, processing_time]])

# Predict the reimbursement delay (days to payment)
predicted_days_to_payment = model.predict(input_features)[0]

# Display prediction result
if st.button('Predict Delay'):
    st.write(f"The predicted reimbursement delay is: **{predicted_days_to_payment:.2f} days**")

# Model Evaluation
st.subheader("Model Evaluation")
st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
