import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np

st.set_page_config(
    page_title="Customer Engagement & Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Engagement & Churn Dashboard")
st.markdown("Streamlit Dashboard (Live Analytics)")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data.csv")
        return df
    except FileNotFoundError:
        np.random.seed(42)
        n = 5000
        data = {
            'CustomerID': range(1, n + 1),
            'Engagement_Classification': np.random.choice(
                ['Active Engaged Customer', 'Active Low-Product Customer', 'Inactive Disengaged Customer', 'Inactive High-Balance Customer'], 
                size=n, p=[0.25, 0.25, 0.25, 0.25]
            ),
            'Num Of Products': np.random.choice([1, 2, 3, 4], size=n, p=[0.5, 0.45, 0.03, 0.02]),
            'Balance': np.random.uniform(1000, 150000, n),
            'Salary': np.random.uniform(20000, 180000, n),
            'Exited': np.random.choice([0, 1], size=n, p=[0.79, 0.21]),
            'Is_Active_Member': np.random.choice([0, 1], size=n, p=[0.49, 0.51]),
            'Relationship_Strength': np.random.choice(['Moderate', 'Strong', 'Weak'], size=n, p=[0.25, 0.25, 0.5])
        }
        return pd.DataFrame(data)

df = load_data()

st.sidebar.header("🔍 User Capabilities & Filters")

eng_classes = df['Engagement_Classification'].unique().tolist()
selected_eng_class = st.sidebar.multiselect(
    "Engagement Filters", 
    options=eng_classes, 
    default=eng_classes
)

min_prod, max_prod = int(df['Num Of Products'].min()), int(df['Num Of Products'].max())
selected_products = st.sidebar.slider(
    "Product Count Sliders", 
    min_value=min_prod, 
    max_value=max_prod, 
    value=(min_prod, max_prod)
)

max_balance = float(df['Balance'].max())
selected_balance = st.sidebar.slider(
    "Balance Threshold", 
    0.0, 
    max_balance, 
    (0.0, max_balance)
)

max_salary = float(df['Salary'].max())
selected_salary = st.sidebar.slider(
    "Salary Threshold", 
    0.0, 
    max_salary, 
    (0.0, max_salary)
)

filtered_df = df[
    (df['Engagement_Classification'].isin(selected_eng_class)) &
    (df['Num Of Products'].between(selected_products[0], selected_products[1])) &
    (df['Balance'].between(selected_balance[0], selected_balance[1])) &
    (df['Salary'].between(selected_salary[0], selected_salary[1]))
]

st.header("1. Engagement vs Churn Overview")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribution of Exited / Churn")
    fig_churn = px.pie(filtered_df, names='Exited', title="Active vs Exited Customers", hole=0.4)
    st.plotly_chart(fig_churn, use_container_width=True)

with col2:
    st.subheader("Engagement Classification Breakdown")
    eng_summary = filtered_df.groupby('Engagement_Classification').agg(
        COUNT_of_Customer_ID=('CustomerID', 'count'),
        SUM_of_Exited=('Exited', 'sum')
    ).reset_index()
    eng_summary['Churn Rate (%)'] = (eng_summary['SUM_of_Exited'] / eng_summary['COUNT_of_Customer_ID']) * 100
    st.dataframe(eng_summary, use_container_width=True)

st.header("2. Product Utilization Impact Analysis")
prod_analysis = filtered_df.groupby('Num Of Products').agg(
    COUNT_of_Customer_ID=('CustomerID', 'count'),
    SUM_of_Exited=('Exited', 'sum')
).reset_index()
prod_analysis['Churn Rate (%)'] = (prod_analysis['SUM_of_Exited'] / prod_analysis['COUNT_of_Customer_ID']) * 100

col3, col4 = st.columns(2)
with col3:
    st.dataframe(prod_analysis, use_container_width=True)

with col4:
    fig_prod = px.bar(
        prod_analysis,
        x='Num Of Products',
        y='Churn Rate (%)',
        title="Churn Rate by Number Of Products",
        text='Churn Rate (%)'
    )
    st.plotly_chart(fig_prod, use_container_width=True)

st.header("3. High-Value Disengaged Customer Detector")
st.markdown("Customers with **High Balance/Salary** but **Low Engagement Score**.")

balance_median = filtered_df['Balance'].median()
filtered_df['Balance_Category'] = filtered_df['Balance'].apply(lambda x: 'High Balance' if x >= balance_median else 'Normal Balance')

high_bal_summary = filtered_df.groupby(['Balance_Category', 'Is_Active_Member']).agg(
    COUNT_of_Customer_ID=('CustomerID', 'count')
).reset_index().pivot(index='Balance_Category', columns='Is_Active_Member', values='COUNT_of_Customer_ID').fillna(0)

st.dataframe(high_bal_summary, use_container_width=True)

fig_high_bal = px.bar(
    filtered_df.groupby(['Balance_Category', 'Is_Active_Member']).size().reset_index(name='Customer_ID'),
    x='Balance_Category',
    y='Customer_ID',
    color='Is_Active_Member',
    title="High Balance Customers (Active vs Inactive)",
    barmode='group'
)
st.plotly_chart(fig_high_bal, use_container_width=True)

st.header("4. Retention Strength Scoring Panels")
col5, col6 = st.columns(2)

with col5:
    st.subheader("Relationship Strength Index")
    rel_strength = filtered_df.groupby('Relationship_Strength').agg(
        COUNT_of_Customer_ID=('CustomerID', 'count')
    ).reset_index()
    st.dataframe(rel_strength, use_container_width=True)

with col6:
    total_customers = len(filtered_df)
    churned_count = filtered_df['Exited'].sum()
    retention_rate = ((total_customers - churned_count) / total_customers) * 100 if total_customers > 0 else 0
    
    st.metric("Total Filtered Customers", total_customers)
    st.metric("Retention Rate", f"{retention_rate:.2f}%")
    st.metric("At-Risk Customers", churned_count)
