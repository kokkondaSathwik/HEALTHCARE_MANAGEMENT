import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the pre-trained model
model_file = "pima_diabetes_model.pkl"
model = joblib.load(model_file)

# Set Page Config
st.set_page_config(page_title="Healthcare Management", page_icon="💡", layout="wide")

# Custom Styles
st.markdown(
    """
    <style>
    .main-title {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        color: #4CAF50;
        margin-bottom: 30px;
    }
    .sub-title {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
        color: #1F618D; /* Dark Blue */
    }
    .sidebar-title {
        color: #1F618D; /* Dark Blue */
        font-size: 20px;
        font-weight: bold;
        text-align: center;
    }
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #FFEB3B; /* Yellow */
    }
    /* Sidebar text color */
    [data-testid="stSidebar"] .stSelectbox div div {
        color: #1F618D; /* Dark Blue */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main App Title
st.markdown('<div class="main-title">Healthcare Management</div>', unsafe_allow_html=True)

# Navigation Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-title">Navigation</div>', unsafe_allow_html=True)
    menu = ["🏥 Health Risk Assessment", "📊 Health Record Visualization"]
    choice = st.selectbox("Select a Feature", menu)
    st.sidebar.markdown("---")
    st.sidebar.markdown("Developed by *YourName*", unsafe_allow_html=True)
    st.sidebar.markdown("© 2025 All Rights Reserved")

# Health Risk Assessment
if choice == "🏥 Health Risk Assessment":
    st.markdown('<div class="sub-title">Health Risk Assessment Based on Lifestyle Factors</div>', unsafe_allow_html=True)
    
    # Lifestyle Questionnaire
    with st.container():
        st.markdown("### Answer the following questions:")
        col1, col2 = st.columns(2)
        with col1:
            smoking = st.radio("Do you smoke?", ["Yes", "No"], horizontal=True)
            sleep_hours = st.slider("How many hours do you sleep per night?", 0, 12, 7)
        with col2:
            physical_activity = st.slider("How many days a week do you exercise?", 0, 7, 3)
            fruits_vegetables = st.slider("How many servings of fruits and vegetables do you eat daily?", 0, 10, 3)

    # Risk Assessment Logic
    risk_score = 0
    if smoking == "Yes":
        risk_score += 2
    if physical_activity < 3:
        risk_score += 2
    if sleep_hours < 6:
        risk_score += 1
    if fruits_vegetables < 5:
        risk_score += 1

    # Display Risk Score
    st.markdown("### Your Health Risk Score:")
    if risk_score <= 2:
        st.success("✅ **Low Risk**")
    elif risk_score <= 4:
        st.warning("⚠ **Moderate Risk**")
    else:
        st.error("❌ **High Risk**")

# Health Record Visualization
elif choice == "📊 Health Record Visualization":
    st.markdown('<div class="sub-title">Health Record Visualization</div>', unsafe_allow_html=True)
    
    # File Upload
    uploaded_file = st.file_uploader("Upload your health record (CSV format):", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown("### Uploaded Data:")
        st.dataframe(df, use_container_width=True)

        # Select Column to Visualize
        column = st.selectbox("Select a column to visualize:", df.columns)

        # Visualization Tabs
        st.markdown("### Visualization:")
        tab1, tab2 = st.tabs(["📈 Line Chart", "📊 Histogram"])
        with tab1:
            if 'Time' in df.columns or 'Date' in df.columns:
                time_col = 'Time' if 'Time' in df.columns else 'Date'
                df[time_col] = pd.to_datetime(df[time_col])
                df = df.sort_values(by=time_col)
                plt.figure(figsize=(10, 6))
                plt.plot(df[time_col], df[column], marker='o', color="blue", label=column)
                plt.title(f"{column} Over Time", fontsize=16)
                plt.xlabel(time_col)
                plt.ylabel(column)
                plt.grid()
                plt.legend()
                st.pyplot(plt)
            else:
                st.warning("No time or date column found.")

        with tab2:
            plt.figure(figsize=(8, 6))
            sns.histplot(df[column], kde=True, color="green")
            plt.title(f"Distribution of {column}", fontsize=16)
            plt.xlabel(column)
            plt.ylabel("Frequency")
            st.pyplot(plt)

        # Highlight Outliers
        st.markdown("### Outlier Detection:")
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        st.write(f"Number of Outliers: *{len(outliers)}*")
        st.dataframe(outliers, use_container_width=True)
