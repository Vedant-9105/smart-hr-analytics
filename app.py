import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Smart HR Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    /* Force black text in all metric cards */
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
    .css-1kyxreq {
        background-color: #1E3A5F;
    }
    h1 {
        color: #1E3A5F;
    }
    /* Force black text in metric cards */
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] div[data-testid="stMetricValue"],
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
        color: #000000 !important;
    }
    </style>
    """, unsafe_allow_html=True
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('hr_analytics_complete_data.csv')
    # Convert necessary columns
    if 'churn' in df.columns:
        df['churn'] = df['churn'].astype(int)
    if 'is_hipo' in df.columns:
        df['is_hipo'] = df['is_hipo'].astype(bool)
    return df

# Load data with error handling
try:
    master = load_data()
    st.success("✅ Data loaded successfully!")
except FileNotFoundError:
    st.error("❌ hr_analytics_complete_data.csv not found. Please run smart_hr_analytics.py first.")
    st.stop()

# Sidebar filters
st.sidebar.image("https://img.icons8.com/color/96/000000/bar-chart.png", width=80)
st.sidebar.title("🔍 Filters")

# Department filter
departments = ['All'] + sorted(master['dept_name'].dropna().unique().tolist())
selected_dept = st.sidebar.selectbox("📂 Department", departments)

# Risk category filter
risk_cats = ['All'] + sorted(master['risk_category'].dropna().unique().tolist())
selected_risk = st.sidebar.selectbox("⚠️ Risk Category", risk_cats)

# Gender filter
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
    st.metric("⚠️ Churn Rate", f"{churn_rate:.1f}%", delta=f"{churn_rate - 19.97:.1f}% vs overall")
with col3:
    hipo_count = filtered_df['is_hipo'].sum()
    st.metric("⭐ HiPo Employees", f"{hipo_count:,}", delta=f"{hipo_count/len(filtered_df)*100:.1f}% of filtered")
with col4:
    avg_salary = filtered_df['current_salary'].mean()
    st.metric("💰 Avg Salary", f"${avg_salary:,.0f}")
with col5:
    dei_score = 93.3  # from your DEI report
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
    st.subheader("📊 Top Churn Drivers (Feature Importance)")
    # Sample feature importance data (you can load from your model if saved)
    feature_imp = pd.DataFrame({
        'Feature': ['Salary Growth', 'Current Salary', 'Age', 'Tenure', 'Title Changes', 'Title', 'Dept', 'Gender'],
        'Importance': [0.216, 0.212, 0.160, 0.125, 0.100, 0.098, 0.071, 0.017]
    })
    fig = px.bar(
        feature_imp, x='Importance', y='Feature', orientation='h',
        title='Feature Importance (LightGBM)',
        color='Importance', color_continuous_scale='Blues'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Row 4: Employee Lookup Tool
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
        col_d.metric("🎯 Churn Probability", f"{emp.get('churn_probability', 0.5)*100:.1f}%" if 'churn_probability' in emp else "N/A")
        
        st.markdown("#### 📌 Career Recommendation")
        st.info(f"**Recommended Next Role:** {emp['recommended_next_title']}")
        
        # Radar chart for employee metrics
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

# Row 5: DEI Dashboard
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
    # Salary by gender
    salary_gender = master.groupby('gender')['current_salary'].mean().reset_index()
    fig = px.bar(
        salary_gender, x='gender', y='current_salary',
        title='Average Salary by Gender',
        labels={'current_salary': 'Avg Salary ($)', 'gender': 'Gender'},
        color='gender', color_discrete_sequence=['#3498DB', '#E74C3C']
    )
    st.plotly_chart(fig, use_container_width=True)

with dei_col3:
    # HiPo by gender
    hipo_gender = master.groupby('gender')['is_hipo'].mean() * 100
    fig = px.bar(
        x=hipo_gender.index, y=hipo_gender.values,
        title='HiPo Rate by Gender (%)',
        labels={'x': 'Gender', 'y': 'HiPo Rate (%)'},
        color=hipo_gender.index, color_discrete_sequence=['#2ECC71', '#F39C12']
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("🚀 Built with Streamlit • Smart HR Analytics • Powered by Machine Learning")

# Run: streamlit run app.py


# ================= ADD THIS SECTION =================
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
st.subheader("🤖 Real-Time Churn Prediction (Demo)")
st.write("Enter employee details to see churn probability")

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

# Simple prediction logic (replace with your actual model later)
if st.button("🔮 Predict Churn Risk", use_container_width=True):
    # Mock prediction (you'll replace with your LightGBM model later)
    risk_score = (
        (1 - pred_salary_growth/100) * 0.3 +
        (1 - pred_salary/100000) * 0.2 +
        (pred_tenure / 30) * 0.2 +
        (1 - pred_age/70) * 0.2 +
        (pred_title_changes / 5) * 0.1
    ) * 100
    risk_score = min(100, max(0, risk_score))
    
    if risk_score > 70:
        st.error(f"🚨 HIGH RISK: {risk_score:.1f}% churn probability")
        st.info("💡 Recommendation: Schedule retention interview, review compensation, offer mentoring")
    elif risk_score > 40:
        st.warning(f"⚠️ MEDIUM RISK: {risk_score:.1f}% churn probability")
        st.info("💡 Recommendation: Monitor engagement, consider skill development")
    else:
        st.success(f"✅ LOW RISK: {risk_score:.1f}% churn probability")
        st.info("💡 Recommendation: Continue career pathing and recognition")

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
with st.expander("🤖 Model Performance"):
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("ROC-AUC", "0.764", "LightGBM")
    col_m2.metric("Precision (Churn)", "0.52")
    col_m3.metric("Recall (Churn)", "0.74")

# Load actual feature importance if you saved it
feature_imp = pd.DataFrame({
    'Feature': ['salary_growth_pct', 'current_salary', 'age', 'tenure_years', 
                'num_title_changes', 'title_encoded', 'dept_encoded', 'gender_encoded'],
    'Importance': [0.216, 0.212, 0.160, 0.125, 0.100, 0.098, 0.071, 0.017]
})