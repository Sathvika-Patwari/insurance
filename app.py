import streamlit as st
import pandas as pd
import numpy as np
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

# Sidebar filters
st.sidebar.header("Filters")
insurance_carrier_filter = st.sidebar.multiselect(
    "Select Insurance Carrier",
    options=raw_data['insurance_carrier'].unique(),
    default=raw_data['insurance_carrier'].unique()
)

procedure_type_filter = st.sidebar.multiselect(
    "Select Procedure Type",
    options=raw_data['procedure_type'].unique(),
    default=raw_data['procedure_type'].unique()
)

date_range_filter = st.sidebar.date_input(
    "Select Date Range",
    [raw_data['submission_date'].min(), raw_data['submission_date'].max()]
)

# Filter data
filtered_data = raw_data[
    (raw_data['insurance_carrier'].isin(insurance_carrier_filter)) &
    (raw_data['procedure_type'].isin(procedure_type_filter)) &
    (raw_data['submission_date'] >= pd.to_datetime(date_range_filter[0])) &
    (raw_data['submission_date'] <= pd.to_datetime(date_range_filter[1]))
]

# Dynamic Insight Computation
def compute_insights(filtered_data):
    # Calculate mean, min, and max values, handling NaN values by using .fillna(0) or a default value
    avg_claim_amount = filtered_data['claim_amount'].fillna(0).mean()
    avg_delay = filtered_data['days_to_payment'].fillna(0).mean()
    min_claim_amount = filtered_data['claim_amount'].fillna(0).min()
    max_claim_amount = filtered_data['claim_amount'].fillna(0).max()
    min_delay = filtered_data['days_to_payment'].fillna(0).min()
    max_delay = filtered_data['days_to_payment'].fillna(0).max()
    
    # Round the values for display
    avg_claim_amount = round(avg_claim_amount, 2)
    avg_delay = round(avg_delay, 2)
    min_claim_amount = round(min_claim_amount, 2)
    max_claim_amount = round(max_claim_amount, 2)
    min_delay = round(min_delay, 2)
    max_delay = round(max_delay, 2)
    
    return avg_claim_amount, avg_delay, min_claim_amount, max_claim_amount, min_delay, max_delay

# Graph and Insights function
def show_graph_with_insight(graph, explanation, insight):
    st.plotly_chart(graph)
    st.markdown(f"**Explanation:** {explanation}")
    st.markdown(f"**Insight:** {insight}")

# Safe transformation function to handle unseen labels
def safe_transform(label_encoder, label):
    try:
        return label_encoder.transform([label])[0]
    except ValueError:
        # Handle unseen label by assigning a default value (e.g., -1 for unknown)
        return -1

# Create a new page for prediction
def prediction_page():
    st.subheader("Predict Reimbursement Delay")
    
    # User input form for prediction
    claim_amount = st.number_input('Claim Amount', min_value=0, value=1000)
    processing_time = st.number_input('Processing Time (Days)', min_value=1, value=30)
    insurance_carrier = st.selectbox('Insurance Carrier', raw_data['insurance_carrier'].unique())
    procedure_type = st.selectbox('Procedure Type', raw_data['procedure_type'].unique())

    # Encode user inputs
    insurance_carrier_encoded = safe_transform(label_encoder, insurance_carrier)
    procedure_type_encoded = safe_transform(label_encoder, procedure_type)
    
    # Prepare input for prediction
    input_features = np.array([[claim_amount, insurance_carrier_encoded, procedure_type_encoded, processing_time]])

    # Predict the reimbursement delay
    predicted_days_to_payment = model.predict(input_features)[0]

    if st.button('Predict Delay'):
        st.write(f"The predicted reimbursement delay is: **{predicted_days_to_payment:.2f} days**")

# Dashboard layout and navigation
nav = st.sidebar.radio("Navigation", options=["Dashboard", "Prediction"])

if nav == "Dashboard":
    st.title("Dental Hospital Insurance Claims Analysis Dashboard")

    # Claim Amount Distribution by Insurance Carrier
    st.subheader("1. Claim Amount Distribution by Insurance Carrier")
    fig = px.box(filtered_data, x='insurance_carrier', y='claim_amount', title="Claim Amount Distribution by Insurance Carrier")
    avg_claim_amount, avg_delay, min_claim_amount, max_claim_amount, min_delay, max_delay = compute_insights(filtered_data)
    
    show_graph_with_insight(
        fig,
        "This box plot shows the distribution of claim amounts for different insurance carriers.",
        f"Average claim amount: **${avg_claim_amount}$**.\nMinimum claim amount: **${min_claim_amount}$**.\nMaximum claim amount: **${max_claim_amount}**."
    )

    # Reimbursement Delay vs Procedure Type
    st.subheader("2. Reimbursement Delay vs Procedure Type")
    fig = px.box(filtered_data, x='procedure_type', y='days_to_payment', title="Reimbursement Delay vs Procedure Type")
    show_graph_with_insight(
        fig,
        "This box plot shows the reimbursement delay for different types of procedures.",
        f"Average reimbursement delay: **{avg_delay} days**.\nMinimum delay: **{min_delay} days**.\nMaximum delay: **{max_delay} days**."
    )

    # Claim Status Distribution
    st.subheader("3. Claim Status Distribution")
    claim_status_counts = filtered_data['status'].value_counts()
    fig = px.pie(claim_status_counts, names=claim_status_counts.index, values=claim_status_counts.values, title="Claim Status Distribution")
    
    st.write(f"Total claims processed: **{len(filtered_data)}**")
    st.write(f"Approved Claims: **{claim_status_counts.get('Approved', 0)}**")
    st.write(f"Rejected Claims: **{claim_status_counts.get('Rejected', 0)}**")
    show_graph_with_insight(
        fig,
        "This pie chart shows the distribution of claims by status.",
        f"**{claim_status_counts.get('Approved', 0)}** claims were approved, while **{claim_status_counts.get('Rejected', 0)}** were rejected."
    )

    # Claim Amount vs Days to Payment
    st.subheader("4. Claim Amount vs Days to Payment")
    fig = px.scatter(filtered_data, x='claim_amount', y='days_to_payment', color='status', title="Claim Amount vs Days to Payment",
                     color_discrete_map={'Rejected': 'red', 'Approved': 'blue'}, labels={'claim_amount': 'Claim Amount', 'days_to_payment': 'Days to Payment'})
    
    show_graph_with_insight(
        fig,
        "This scatter plot shows the relationship between claim amount and reimbursement delay.",
        f"The average reimbursement delay is **{avg_delay} days**."
    )

    # Days to Payment by Insurance Carrier
    st.subheader("5. Days to Payment by Insurance Carrier")
    fig = px.box(filtered_data, x='insurance_carrier', y='days_to_payment', title="Days to Payment by Insurance Carrier")
    show_graph_with_insight(
        fig,
        "This box plot displays the variation in reimbursement delays for different insurance carriers.",
        f"Average reimbursement delay for top carriers ranges from **{min_delay} days** to **{max_delay} days**."
    )

    # Top 5 Insurance Carriers by Total Claim Amount
    st.subheader("6. Top 5 Insurance Carriers by Total Claim Amount")
    top_carriers = filtered_data.groupby('insurance_carrier')['claim_amount'].sum().nlargest(5)
    fig = px.bar(top_carriers, title="Top 5 Insurance Carriers by Total Claim Amount")
    
    show_graph_with_insight(
        fig,
        "This bar chart shows the top 5 insurance carriers based on total claim amounts.",
        f"The highest total claim amount is **${top_carriers.max()}$**, while the lowest in the top 5 is **${top_carriers.min()}**."
    )

    # Claim Amount by Provider
    st.subheader("7. Claim Amount by Provider")
    fig = px.bar(filtered_data, x='provider_name', y='claim_amount', title="Claim Amount by Provider")
    show_graph_with_insight(
        fig,
        "This bar chart shows the claim amounts grouped by provider.",
        f"Providers with the highest claims include **{filtered_data.groupby('provider_name')['claim_amount'].sum().idxmax()}**, with total claims of **${filtered_data.groupby('provider_name')['claim_amount'].sum().max()}**."
    )

    # Claim Rejection Reasons
    st.subheader("8. Claim Rejection Reasons")
    rejection_reasons = filtered_data['rejection_reason'].value_counts().dropna()
    fig = px.pie(rejection_reasons, names=rejection_reasons.index, values=rejection_reasons.values, title="Claim Rejection Reasons")
    
    st.write(f"Most common rejection reason: **{rejection_reasons.idxmax()}** with **{rejection_reasons.max()}** occurrences.")
    show_graph_with_insight(
        fig,
        "This pie chart shows the breakdown of rejection reasons for claims.",
        f"The most common rejection reason is **{rejection_reasons.idxmax()}**, with **{rejection_reasons.max()}** claims."
    )

    # Patient Responsibility vs Insurance Paid
    st.subheader("9. Patient Responsibility vs Insurance Paid")
    fig = px.scatter(filtered_data, x='patient_responsibility', y='insurance_paid', title="Patient Responsibility vs Insurance Paid")
    show_graph_with_insight(
        fig,
        "This scatter plot shows the relationship between patient responsibility and insurance paid.",
        "There appears to be a moderate inverse relationship between patient responsibility and insurance paid."
    )

    # Patient Volume Growth Over Time
    st.subheader("10. Patient Volume Growth Over Time")
    patient_volume = filtered_data.groupby(filtered_data['submission_date'].dt.to_period('M')).size()
    patient_volume.index = patient_volume.index.to_timestamp()
    fig = px.line(patient_volume, title="Patient Volume Growth Over Time")
    show_graph_with_insight(
        fig,
        "This line chart shows the number of claims submitted over time.",
        f"The total number of claims has increased steadily over the months, with a noticeable spike in **{patient_volume.idxmax().strftime('%B %Y')}**."
    )

elif nav == "Prediction":
    prediction_page()
