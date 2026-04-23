import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix
import lightgbm as lgb
import shap

# Page configuration
st.set_page_config(
    page_title="Smart HR Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown(
    """
    <style>
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] div[data-testid="stMetricValue"],
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
        color: #000000 !important;
    }
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
    }
    h1 {
        color: #1E3A5F;
    }
    </style>
    """, unsafe_allow_html=True
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('hr_analytics_complete_data.csv')
    # Ensure boolean columns
    if 'is_hipo' in df.columns:
        df['is_hipo'] = df['is_hipo'].astype(bool)
    if 'churn' in df.columns:
        df['churn'] = df['churn'].astype(int)
    return df

# Function to add encoded columns and train model
@st.cache_resource
def prepare_and_train(data):
    # Make a copy
    df = data.copy()
    
    # Encode categorical variables
    gender_encoder = LabelEncoder()
    dept_encoder = LabelEncoder()
    title_encoder = LabelEncoder()
    
    df['gender_encoded'] = gender_encoder.fit_transform(df['gender'])
    df['dept_encoded'] = dept_encoder.fit_transform(df['dept_name'])
    df['title_encoded'] = title_encoder.fit_transform(df['current_title'])
    
    # Define features
    feature_cols = ['age', 'tenure_years', 'current_salary', 'num_title_changes',
                    'salary_growth_pct', 'gender_encoded', 'dept_encoded', 'title_encoded']
    numeric_cols = ['age', 'tenure_years', 'current_salary', 'num_title_changes', 'salary_growth_pct']
    
    X = df[feature_cols].copy()
    y = df['churn']
    
    # Store medians for numeric columns (used later for imputation)
    medians = X[numeric_cols].median()
    
    # Handle missing values (just in case)
    X[numeric_cols] = X[numeric_cols].fillna(medians)
    
    # Scale numeric features
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    # Train LightGBM
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    
    # Feature importance
    importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    return model, scaler, medians, importance, explainer, feature_cols, numeric_cols, gender_encoder, dept_encoder, title_encoder

# Load data
try:
    master = load_data()
    st.success("✅ Data loaded successfully!")
except FileNotFoundError:
    st.error("❌ hr_analytics_complete_data.csv not found. Please run smart_hr_analytics.py first.")
    st.stop()

# Train model and get artifacts
(model, scaler, medians, feature_importance, explainer, feature_cols, numeric_cols,
 gender_encoder, dept_encoder, title_encoder) = prepare_and_train(master)

# Add encoded columns to master for later use (employee lookup)
master['gender_encoded'] = gender_encoder.transform(master['gender'])
master['dept_encoded'] = dept_encoder.transform(master['dept_name'])
master['title_encoded'] = title_encoder.transform(master['current_title'])

# Sidebar filters
st.sidebar.image("https://img.icons8.com/color/96/000000/bar-chart.png", width=80)
st.sidebar.title("🔍 Filters")

departments = ['All'] + sorted(master['dept_name'].dropna().unique().tolist())
selected_dept = st.sidebar.selectbox("📂 Department", departments)

risk_cats = ['All'] + sorted(master['risk_category'].dropna().unique().tolist())
selected_risk = st.sidebar.selectbox("⚠️ Risk Category", risk_cats)

genders = ['All', 'M', 'F']
selected_gender = st.sidebar.selectbox("👥 Gender", genders)

# Apply filters
filtered_df = master.copy()
if selected_dept != 'All':
    filtered_df = filtered_df[filtered_df['dept_name'] == selected_dept]
if selected_risk != 'All':
    filtered_df = filtered_df[filtered_df['risk_category'] == selected_risk]
if selected_gender != 'All':
    filtered_df = filtered_df[filtered_df['gender'] == selected_gender]

# Main title
st.title("📊 Smart HR Analytics Dashboard")
st.markdown("*AI-Powered Employee Insights, Churn Prediction, and Talent Management*")
st.markdown("---")

# KPI Row
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("👥 Total Employees", f"{len(filtered_df):,}")
with col2:
    churn_rate = filtered_df['churn'].mean() * 100
    overall_churn = master['churn'].mean() * 100
    st.metric("⚠️ Churn Rate", f"{churn_rate:.1f}%", delta=f"{churn_rate - overall_churn:.1f}% vs overall")
with col3:
    hipo_count = filtered_df['is_hipo'].sum()
    st.metric("⭐ HiPo Employees", f"{hipo_count:,}", delta=f"{hipo_count/len(filtered_df)*100:.1f}% of filtered")
with col4:
    avg_salary = filtered_df['current_salary'].mean()
    st.metric("💰 Avg Salary", f"${avg_salary:,.0f}")
with col5:
    # Compute DEI score from data
    female_rep = (master['gender'] == 'F').mean() * 100
    male_salary = master[master['gender']=='M']['current_salary'].mean()
    female_salary = master[master['gender']=='F']['current_salary'].mean()
    pay_gap = abs(male_salary - female_salary) / male_salary * 100 if male_salary > 0 else 0
    equity_score = max(0, 100 - pay_gap * 2)
    rep_score = max(0, 100 - abs(50 - female_rep) * 2)
    promo_male = master[master['gender']=='M']['num_title_changes'].mean()
    promo_female = master[master['gender']=='F']['num_title_changes'].mean()
    promo_gap = abs(promo_male - promo_female) / promo_male * 100 if promo_male > 0 else 0
    promo_equity = max(0, 100 - promo_gap)
    dei_score = (rep_score + equity_score + promo_equity) / 3
    st.metric("🏆 DEI Score", f"{dei_score:.0f}/100")

st.markdown("---")

# Row 2: Two columns for charts
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📈 Churn Rate by Department")
    dept_churn = filtered_df.groupby('dept_name')['churn'].mean().sort_values(ascending=False) * 100
    fig = px.bar(
        x=dept_churn.values, y=dept_churn.index, orientation='h',
        labels={'x': 'Churn Rate (%)', 'y': 'Department'},
        color=dept_churn.values, color_continuous_scale='Reds',
        title='Churn Rate by Department'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("🎯 Risk Category Distribution")
    risk_dist = filtered_df['risk_category'].value_counts()
    fig = px.pie(
        values=risk_dist.values, names=risk_dist.index,
        title='Employee Risk Segmentation',
        color_discrete_sequence=px.colors.sequential.Greens_r
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# Row 3: Two more columns
col_left2, col_right2 = st.columns(2)

with col_left2:
    st.subheader("💰 Salary Distribution by Churn")
    fig = px.box(
        filtered_df, x='churn', y='current_salary',
        labels={'churn': 'Churned?', 'current_salary': 'Salary ($)'},
        title='Salary Comparison: Active vs Churned',
        color='churn', color_discrete_map={0: '#2ECC71', 1: '#E74C3C'}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

with col_right2:
    st.subheader("📊 Top Churn Drivers (LightGBM)")
    fig = px.bar(
        feature_importance.head(8), x='Importance', y='Feature', orientation='h',
        title='Feature Importance from Trained Model',
        color='Importance', color_continuous_scale='Blues'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Employee Lookup Tool with SHAP explanation
st.subheader("🔍 Employee Risk & Career Path Lookup")
col_search1, col_search2 = st.columns([3,1])
with col_search1:
    emp_id = st.number_input("Enter Employee ID", min_value=10000, max_value=500000, step=1, value=10001)
with col_search2:
    st.write("")
    st.write("")
    search_clicked = st.button("🔎 Analyze", use_container_width=True)

if search_clicked or 'last_emp_id' not in st.session_state:
    emp_data = master[master['emp_no'] == emp_id]
    if len(emp_data) > 0:
        emp = emp_data.iloc[0]
        st.success(f"### Employee: {emp['first_name']} {emp['last_name']}")
        
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("📂 Department", emp['dept_name'])
        col_b.metric("⚠️ Risk Category", emp['risk_category'])
        col_c.metric("⭐ HiPo Status", "Yes" if emp['is_hipo'] else "No")
        
        # Prepare features for prediction
        emp_features = emp[feature_cols].copy().to_frame().T
        
        # Impute missing numeric values with medians (if any)
        for col in numeric_cols:
            if emp_features[col].isnull().any():
                emp_features[col] = emp_features[col].fillna(medians[col])
        
        # Convert to float to avoid dtype issues
        emp_features = emp_features.astype(float)
        
        # Scale numeric features
        emp_features[numeric_cols] = scaler.transform(emp_features[numeric_cols])
        
        # Predict churn probability
        prob = model.predict_proba(emp_features)[0, 1]
        col_d.metric("🎯 Churn Probability", f"{prob*100:.1f}%")
        
        st.markdown("#### 📌 Career Recommendation")
        st.info(f"**Recommended Next Role:** {emp['recommended_next_title']}")
        
        # SHAP explanation for this employee
        with st.expander("🔍 See why this employee is at risk (SHAP explanation)"):
            shap_values = explainer.shap_values(emp_features)
            shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values
            shap_df = pd.DataFrame({
                'Feature': feature_cols,
                'SHAP Value': shap_vals[0]
            }).sort_values('SHAP Value', key=abs, ascending=False)
            fig_shap = px.bar(shap_df, x='SHAP Value', y='Feature', orientation='h',
                              title='Factors increasing churn risk (positive = higher risk)',
                              color='SHAP Value', color_continuous_scale='RdBu')
            st.plotly_chart(fig_shap, use_container_width=True)
        
        # Radar chart
        metrics = {
            'Tenure': min(100, emp['tenure_years']/30*100),
            'Salary Growth': min(100, emp['salary_growth_pct']/100*100),
            'Promotion Rate': min(100, emp['num_title_changes']/5*100),
            'Age Factor': min(100, emp['age']/65*100)
        }
        fig = go.Figure(data=go.Scatterpolar(
            r=list(metrics.values()),
            theta=list(metrics.keys()),
            fill='toself',
            marker=dict(color='#1E3A5F')
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            height=300,
            title="Employee Profile Score"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error(f"❌ Employee ID {emp_id} not found. Please try another ID.")

st.markdown("---")

# DEI Dashboard (enhanced with actual computed scores)
st.subheader("🌈 Diversity, Equity & Inclusion (DEI) Dashboard")
dei_col1, dei_col2, dei_col3 = st.columns(3)

with dei_col1:
    gender_counts = master['gender'].value_counts()
    fig = px.pie(
        values=gender_counts.values, names=gender_counts.index,
        title='Gender Representation',
        color_discrete_sequence=['#3498DB', '#E74C3C']
    )
    st.plotly_chart(fig, use_container_width=True)

with dei_col2:
    salary_gender = master.groupby('gender')['current_salary'].mean().reset_index()
    fig = px.bar(
        salary_gender, x='gender', y='current_salary',
        title='Average Salary by Gender',
        labels={'current_salary': 'Avg Salary ($)', 'gender': 'Gender'},
        color='gender', color_discrete_sequence=['#3498DB', '#E74C3C']
    )
    st.plotly_chart(fig, use_container_width=True)

with dei_col3:
    hipo_gender = master.groupby('gender')['is_hipo'].mean() * 100
    fig = px.bar(
        x=hipo_gender.index, y=hipo_gender.values,
        title='HiPo Rate by Gender (%)',
        labels={'x': 'Gender', 'y': 'HiPo Rate (%)'},
        color=hipo_gender.index, color_discrete_sequence=['#2ECC71', '#F39C12']
    )
    st.plotly_chart(fig, use_container_width=True)

# Advanced Analytics Tabs
st.markdown("---")
st.subheader("📈 Advanced Analytics")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Model Performance", "SHAP Global Analysis", "Clustering Insights", "Career Paths", "DEI Scorecard"])

with tab1:
    col_m1, col_m2, col_m3 = st.columns(3)
    # Compute evaluation metrics on full data
    X_full = master[feature_cols].copy()
    # Impute missing numeric values with medians (if any)
    X_full[numeric_cols] = X_full[numeric_cols].fillna(medians)
    X_full[numeric_cols] = scaler.transform(X_full[numeric_cols])
    y_pred_proba = model.predict_proba(X_full)[:, 1]
    auc = roc_auc_score(master['churn'], y_pred_proba)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    precision = precision_score(master['churn'], y_pred)
    recall = recall_score(master['churn'], y_pred)
    col_m1.metric("ROC-AUC", f"{auc:.3f}")
    col_m2.metric("Precision (Churn)", f"{precision:.2f}")
    col_m3.metric("Recall (Churn)", f"{recall:.2f}")
    st.markdown("**Confusion Matrix (on full data)**")
    cm = confusion_matrix(master['churn'], y_pred)
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=['Active', 'Churned'], y=['Active', 'Churned'])
    st.plotly_chart(fig_cm, use_container_width=True)

with tab2:
    st.markdown("**Global Feature Importance (SHAP summary)**")
    sample_X = X_full.sample(min(500, len(X_full)), random_state=42)
    shap_values_sample = explainer.shap_values(sample_X)
    if isinstance(shap_values_sample, list):
        shap_values_sample = shap_values_sample[1]
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values_sample, sample_X, feature_names=feature_cols, show=False, plot_size=None)
    st.pyplot(fig)
    st.markdown("**SHAP Bar Plot**")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values_sample, sample_X, feature_names=feature_cols, plot_type="bar", show=False)
    st.pyplot(fig2)

with tab3:
    st.markdown("**Employee Clusters (from KMeans)**")
    X_cluster = master[feature_cols].copy()
    X_cluster[numeric_cols] = X_cluster[numeric_cols].fillna(medians)
    X_cluster[numeric_cols] = scaler.transform(X_cluster[numeric_cols])
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_cluster)
    cluster_df = pd.DataFrame({
        'PC1': pca_result[:, 0],
        'PC2': pca_result[:, 1],
        'Cluster': master['risk_cluster'],
        'Risk Category': master['risk_category']
    })
    fig = px.scatter(cluster_df, x='PC1', y='PC2', color='Risk Category',
                     title='Employee Segments (PCA projection)',
                     color_discrete_sequence=px.colors.qualitative.Set1)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("**Cluster Profiles (mean values)**")
    cluster_profile = master.groupby('risk_cluster')[['age', 'tenure_years', 'current_salary', 
                                                      'num_title_changes', 'salary_growth_pct']].mean()
    st.dataframe(cluster_profile.style.format("{:.1f}"))
    
    st.markdown("**Churn Rate by Cluster**")
    cluster_churn = master.groupby('risk_category')['churn'].mean() * 100
    fig_bar = px.bar(x=cluster_churn.index, y=cluster_churn.values,
                     labels={'x': 'Risk Category', 'y': 'Churn Rate (%)'},
                     title='Churn Rate per Risk Segment',
                     color=cluster_churn.values, color_continuous_scale='Reds')
    st.plotly_chart(fig_bar, use_container_width=True)

with tab4:
    st.markdown("**Top 10 Most Common Career Transitions**")
    transitions = master[['current_title', 'recommended_next_title']].dropna()
    trans_counts = transitions.groupby(['current_title', 'recommended_next_title']).size().reset_index(name='count')
    trans_counts = trans_counts.sort_values('count', ascending=False).head(10)
    fig = px.bar(trans_counts, x='count', y='current_title', color='recommended_next_title',
                 title='Most Frequent Recommended Title Changes',
                 labels={'count': 'Number of Employees', 'current_title': 'Current Title'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("**HiPo Employees (Top 10%) - Sample**")
    hipo_sample = master[master['is_hipo']][['emp_no', 'first_name', 'last_name', 'current_title', 
                                             'recommended_next_title', 'dept_name']].head(10)
    st.dataframe(hipo_sample)
    
    st.markdown("**Promotion Probability by Current Title**")
    promo_prob = master.groupby('current_title')['num_title_changes'].mean().sort_values(ascending=False).head(10)
    fig2 = px.bar(x=promo_prob.values, y=promo_prob.index, orientation='h',
                  labels={'x': 'Avg Number of Promotions', 'y': 'Title'},
                  title='Titles with Highest Average Promotions')
    st.plotly_chart(fig2, use_container_width=True)

with tab5:
    st.markdown("**DEI Scorecard Details**")
    st.metric("Overall DEI Score", f"{dei_score:.1f}/100")
    col_d1, col_d2, col_d3 = st.columns(3)
    col_d1.metric("Representation Score", f"{rep_score:.1f}/100", help="Based on gender balance (target 50% female)")
    col_d2.metric("Pay Equity Score", f"{equity_score:.1f}/100", help="Inverse of gender pay gap")
    col_d3.metric("Promotion Equity Score", f"{promo_equity:.1f}/100", help="Equality in promotion rates")
    
    st.markdown("**Gender Pay Gap by Department**")
    dept_pay_gap = master.groupby(['dept_name', 'gender'])['current_salary'].mean().unstack()
    dept_pay_gap['gap_%'] = (dept_pay_gap['M'] - dept_pay_gap['F']) / dept_pay_gap['M'] * 100
    fig = px.bar(dept_pay_gap.reset_index(), x='dept_name', y='gap_%',
                 title='Gender Pay Gap (%) by Department (positive means male higher)',
                 labels={'dept_name': 'Department', 'gap_%': 'Pay Gap (%)'},
                 color='gap_%', color_continuous_scale='RdYlGn')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("**HiPo Rate by Department & Gender**")
    hipo_dept_gender = master.groupby(['dept_name', 'gender'])['is_hipo'].mean().unstack() * 100
    fig2 = px.bar(hipo_dept_gender.reset_index(), x='dept_name', y=['F', 'M'],
                  barmode='group', title='HiPo Rate by Department and Gender (%)',
                  labels={'value': 'HiPo Rate (%)', 'dept_name': 'Department'})
    st.plotly_chart(fig2, use_container_width=True)

# Export and Prediction Sections
st.markdown("---")
st.subheader("📥 Export Reports")

col_export1, col_export2, col_export3 = st.columns(3)

with col_export1:
    if st.button("📊 Download Filtered Data (CSV)", use_container_width=True):
        filtered_df.to_csv('filtered_data.csv', index=False)
        st.success("✅ Downloaded as 'filtered_data.csv'")

with col_export2:
    if st.button("📈 Download Churn Report", use_container_width=True):
        churn_report = filtered_df.groupby('dept_name').agg({
            'churn': 'mean',
            'emp_no': 'count'
        }).rename(columns={'emp_no': 'total_employees', 'churn': 'churn_rate'})
        churn_report['churn_rate'] = churn_report['churn_rate'] * 100
        churn_report.to_csv('churn_report.csv')
        st.success("✅ Churn report saved")

with col_export3:
    if st.button("⭐ Download HiPo List", use_container_width=True):
        hipo_list = filtered_df[filtered_df['is_hipo'] == True][['emp_no', 'first_name', 'last_name', 'current_title', 'recommended_next_title', 'dept_name']]
        hipo_list.to_csv('hipo_list.csv', index=False)
        st.success(f"✅ {len(hipo_list)} HiPo employees exported")

st.markdown("---")
st.subheader("🤖 Real-Time Churn Prediction (Powered by LightGBM)")
st.write("Enter employee details to see churn probability and risk drivers")

pred_col1, pred_col2, pred_col3 = st.columns(3)

with pred_col1:
    pred_age = st.slider("Age", 20, 70, 35)
    pred_tenure = st.slider("Tenure (years)", 0, 30, 5)

with pred_col2:
    pred_salary = st.number_input("Current Salary ($)", 30000, 200000, 60000)
    pred_title_changes = st.number_input("Number of Promotions", 0, 10, 2)

with pred_col3:
    pred_salary_growth = st.slider("Salary Growth (%)", -20, 100, 10)
    pred_gender = st.selectbox("Gender", ["M", "F"])
    pred_dept = st.selectbox("Department", master['dept_name'].dropna().unique())
    pred_title = st.selectbox("Job Title", master['current_title'].dropna().unique())

if st.button("🔮 Predict Churn Risk (ML Model)", use_container_width=True):
    # Encode categorical inputs
    gender_enc = gender_encoder.transform([pred_gender])[0]
    dept_enc = dept_encoder.transform([pred_dept])[0]
    title_enc = title_encoder.transform([pred_title])[0]
    
    # Build feature array
    new_data = pd.DataFrame([[
        pred_age, pred_tenure, pred_salary, pred_title_changes, pred_salary_growth,
        gender_enc, dept_enc, title_enc
    ]], columns=feature_cols)
    
    # No missing values here because all inputs are provided
    new_data = new_data.astype(float)
    new_data[numeric_cols] = scaler.transform(new_data[numeric_cols])
    
    prob = model.predict_proba(new_data)[0, 1]
    risk_score = prob * 100
    
    if risk_score > 70:
        st.error(f"🚨 HIGH RISK: {risk_score:.1f}% churn probability")
        st.info("💡 Recommendation: Schedule retention interview, review compensation, offer mentoring")
    elif risk_score > 40:
        st.warning(f"⚠️ MEDIUM RISK: {risk_score:.1f}% churn probability")
        st.info("💡 Recommendation: Monitor engagement, consider skill development")
    else:
        st.success(f"✅ LOW RISK: {risk_score:.1f}% churn probability")
        st.info("💡 Recommendation: Continue career pathing and recognition")
    
    # SHAP explanation for this prediction
    shap_values_new = explainer.shap_values(new_data)
    shap_vals = shap_values_new[1] if isinstance(shap_values_new, list) else shap_values_new
    shap_df = pd.DataFrame({
        'Feature': feature_cols,
        'SHAP Value': shap_vals[0]
    }).sort_values('SHAP Value', key=abs, ascending=False)
    fig_shap = px.bar(shap_df, x='SHAP Value', y='Feature', orientation='h',
                      title='What drives this prediction (positive = higher risk)',
                      color='SHAP Value', color_continuous_scale='RdBu')
    st.plotly_chart(fig_shap, use_container_width=True)

with st.expander("📖 About This Project"):
    st.markdown("""
    **Smart HR Analytics Dashboard** uses Machine Learning to:
    - Predict employee churn with 76% AUC (LightGBM)
    - Segment employees into 4 flight risk categories
    - Identify Top 10% High-Potential (HiPo) talent
    - Recommend career paths based on historical promotions
    - Audit Diversity, Equity & Inclusion metrics
    
    **Tech Stack:** Python, Streamlit, Scikit-learn, LightGBM, SHAP, Plotly
    **Data:** 300k+ employees, 1.7M+ records
    """)

with st.expander("📋 Executive Summary (from smart_hr_analytics.py)"):
    st.markdown(f"""
    - **Total Employees Analyzed:** {len(master):,}
    - **Overall Churn Rate:** {master['churn'].mean()*100:.2f}%
    - **Best Model:** LightGBM (ROC-AUC = {auc:.3f})
    - **Top 3 Churn Drivers:** {feature_importance.iloc[0]['Feature']}, {feature_importance.iloc[1]['Feature']}, {feature_importance.iloc[2]['Feature']}
    - **Flight Risk Segments:** Stable, Low Risk, Moderate Risk, High Risk
    - **HiPo Percentage:** {master['is_hipo'].mean()*100:.1f}%
    - **DEI Score:** {dei_score:.1f}/100
    """)

st.markdown("---")
st.caption("🚀 Built with Streamlit • Smart HR Analytics • Powered by LightGBM & SHAP")