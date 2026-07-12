# =============================================
# AUTOML DASHBOARD PRO - COMPLETE VERSION
# All 35+ Models + Deep Learning + Full Features
# Cloud Ready for Streamlit / Hugging Face
# =============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, auc, classification_report
)
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="AutoML Dashboard Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# CUSTOM CSS
# =============================================
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .sidebar-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 1rem;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #667eea;
        text-align: center;
        font-size: 13px;
        border-top: 1px solid #333;
        z-index: 999;
    }
    .status-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-success {
        background: #d4edda;
        color: #155724;
    }
    .status-warning {
        background: #fff3cd;
        color: #856404;
    }
    .step-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        height: 100%;
    }
    .step-number {
        background: #667eea;
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================
# CHECK FOR ADVANCED LIBRARIES
# =============================================
try:
    import xgboost as xgb
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGB = True
except:
    HAS_XGB = False

try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor, LGBMClassifier
    HAS_LGBM = True
except:
    HAS_LGBM = False

try:
    import catboost as cb
    from catboost import CatBoostRegressor, CatBoostClassifier
    HAS_CAT = True
except:
    HAS_CAT = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    HAS_TF = True
except:
    HAS_TF = False

# =============================================
# REGRESSION MODELS (19)
# =============================================
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge,
    HuberRegressor, RANSACRegressor, TheilSenRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor

reg_models = {
    "Linear Regression": {
        "model": LinearRegression(),
        "description": "Simple linear model for baseline",
        "category": "Traditional"
    },
    "Ridge": {
        "model": Ridge(random_state=42),
        "description": "Linear model with L2 regularization",
        "category": "Traditional"
    },
    "Lasso": {
        "model": Lasso(random_state=42),
        "description": "Linear model with L1 regularization",
        "category": "Traditional"
    },
    "ElasticNet": {
        "model": ElasticNet(random_state=42),
        "description": "Combines L1 and L2 regularization",
        "category": "Traditional"
    },
    "Bayesian Ridge": {
        "model": BayesianRidge(),
        "description": "Bayesian approach to ridge regression",
        "category": "Traditional"
    },
    "Huber": {
        "model": HuberRegressor(),
        "description": "Robust to outliers",
        "category": "Traditional"
    },
    "RANSAC": {
        "model": RANSACRegressor(random_state=42),
        "description": "Robust regression using RANSAC",
        "category": "Traditional"
    },
    "TheilSen": {
        "model": TheilSenRegressor(random_state=42),
        "description": "Robust to outliers using Theil-Sen",
        "category": "Traditional"
    },
    "Decision Tree": {
        "model": DecisionTreeRegressor(random_state=42),
        "description": "Non-linear tree-based model",
        "category": "Tree-Based"
    },
    "Random Forest": {
        "model": RandomForestRegressor(random_state=42),
        "description": "Ensemble of decision trees",
        "category": "Ensemble"
    },
    "Extra Trees": {
        "model": ExtraTreesRegressor(random_state=42),
        "description": "Extremely randomized trees",
        "category": "Ensemble"
    },
    "Gradient Boosting": {
        "model": GradientBoostingRegressor(random_state=42),
        "description": "Boosting ensemble for high performance",
        "category": "Ensemble"
    },
    "AdaBoost": {
        "model": AdaBoostRegressor(random_state=42),
        "description": "Adaptive boosting ensemble",
        "category": "Ensemble"
    },
    "Bagging": {
        "model": BaggingRegressor(random_state=42),
        "description": "Bootstrap aggregating ensemble",
        "category": "Ensemble"
    },
    "KNN": {
        "model": KNeighborsRegressor(),
        "description": "Instance-based learning",
        "category": "Traditional"
    },
    "SVR Linear": {
        "model": SVR(kernel='linear'),
        "description": "Support vector regression with linear kernel",
        "category": "SVM"
    },
    "SVR RBF": {
        "model": SVR(kernel='rbf'),
        "description": "Support vector regression with RBF kernel",
        "category": "SVM"
    },
    "Kernel Ridge": {
        "model": KernelRidge(),
        "description": "Kernelized ridge regression",
        "category": "Kernel"
    },
    "MLP": {
        "model": MLPRegressor(max_iter=1500, random_state=42),
        "description": "Neural network for regression",
        "category": "Neural Network"
    }
}

# =============================================
# CLASSIFICATION MODELS (16)
# =============================================
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

clf_models = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "description": "Fast linear classifier",
        "category": "Traditional"
    },
    "Ridge Classifier": {
        "model": RidgeClassifier(random_state=42),
        "description": "Ridge regression for classification",
        "category": "Traditional"
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=42),
        "description": "Non-linear tree-based classifier",
        "category": "Tree-Based"
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "description": "Ensemble of decision trees",
        "category": "Ensemble"
    },
    "Extra Trees": {
        "model": ExtraTreesClassifier(random_state=42),
        "description": "Extremely randomized trees",
        "category": "Ensemble"
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "description": "Boosting ensemble for high performance",
        "category": "Ensemble"
    },
    "AdaBoost": {
        "model": AdaBoostClassifier(random_state=42),
        "description": "Adaptive boosting ensemble",
        "category": "Ensemble"
    },
    "KNN": {
        "model": KNeighborsClassifier(),
        "description": "Instance-based learning",
        "category": "Traditional"
    },
    "SVC Linear": {
        "model": SVC(kernel='linear', random_state=42),
        "description": "SVM with linear kernel",
        "category": "SVM"
    },
    "SVC RBF": {
        "model": SVC(kernel='rbf', random_state=42),
        "description": "SVM with RBF kernel",
        "category": "SVM"
    },
    "Gaussian NB": {
        "model": GaussianNB(),
        "description": "Naive Bayes with Gaussian distribution",
        "category": "Bayesian"
    },
    "MLP": {
        "model": MLPClassifier(max_iter=1000, random_state=42),
        "description": "Neural network for classification",
        "category": "Neural Network"
    },
    "SGD Classifier": {
        "model": SGDClassifier(max_iter=1000, tol=1e-3, random_state=42),
        "description": "Stochastic gradient descent classifier",
        "category": "Traditional"
    },
    "Bernoulli NB": {
        "model": BernoulliNB(),
        "description": "Naive Bayes for binary features",
        "category": "Bayesian"
    },
    "LDA": {
        "model": LinearDiscriminantAnalysis(),
        "description": "Linear discriminant analysis",
        "category": "Traditional"
    },
    "QDA": {
        "model": QuadraticDiscriminantAnalysis(),
        "description": "Quadratic discriminant analysis",
        "category": "Traditional"
    }
}

# =============================================
# ADVANCED MODELS (XGBoost, LightGBM, CatBoost)
# =============================================
advanced_reg_models = {}
if HAS_XGB:
    advanced_reg_models["XGBoost Regressor"] = {
        "model": XGBRegressor(random_state=42, n_jobs=-1),
        "description": "Extreme Gradient Boosting - High performance",
        "category": "Advanced"
    }
if HAS_LGBM:
    advanced_reg_models["LightGBM Regressor"] = {
        "model": LGBMRegressor(random_state=42, n_jobs=-1),
        "description": "Light GBM - Fast and efficient",
        "category": "Advanced"
    }
if HAS_CAT:
    advanced_reg_models["CatBoost Regressor"] = {
        "model": CatBoostRegressor(random_state=42, verbose=0),
        "description": "CatBoost - Handles categorical well",
        "category": "Advanced"
    }

advanced_clf_models = {}
if HAS_XGB:
    advanced_clf_models["XGBoost Classifier"] = {
        "model": XGBClassifier(random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss'),
        "description": "Extreme Gradient Boosting - High performance",
        "category": "Advanced"
    }
if HAS_LGBM:
    advanced_clf_models["LightGBM Classifier"] = {
        "model": LGBMClassifier(random_state=42, n_jobs=-1),
        "description": "Light GBM - Fast and efficient",
        "category": "Advanced"
    }
if HAS_CAT:
    advanced_clf_models["CatBoost Classifier"] = {
        "model": CatBoostClassifier(random_state=42, verbose=0),
        "description": "CatBoost - Handles categorical well",
        "category": "Advanced"
    }

# =============================================
# HYPERPARAMETER GRIDS
# =============================================
reg_param_grids = {
    "Ridge": {'alpha': [0.1, 1.0, 10.0]},
    "Lasso": {'alpha': [0.1, 1.0, 10.0]},
    "ElasticNet": {'alpha': [0.1, 1.0], 'l1_ratio': [0.5, 0.7, 0.9]},
    "Decision Tree": {'max_depth': [5, 10, 20], 'min_samples_split': [2, 5, 10]},
    "Random Forest": {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20]},
    "Gradient Boosting": {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]},
    "KNN": {'n_neighbors': [3, 5, 7, 10]},
    "MLP": {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'alpha': [0.001, 0.01]}
}

if HAS_XGB:
    reg_param_grids["XGBoost Regressor"] = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.3]
    }
if HAS_LGBM:
    reg_param_grids["LightGBM Regressor"] = {
        'num_leaves': [31, 63, 127],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [100, 200]
    }
if HAS_CAT:
    reg_param_grids["CatBoost Regressor"] = {
        'depth': [4, 6, 8],
        'iterations': [100, 200, 300],
        'learning_rate': [0.01, 0.1]
    }

clf_param_grids = {
    "Logistic Regression": {'C': [0.1, 1.0, 10.0]},
    "Decision Tree": {'max_depth': [5, 10, 20], 'min_samples_split': [2, 5, 10]},
    "Random Forest": {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20]},
    "Gradient Boosting": {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]},
    "KNN": {'n_neighbors': [3, 5, 7, 10]},
    "SVC RBF": {'C': [0.1, 1.0, 10.0], 'gamma': ['scale', 'auto']},
    "MLP": {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'alpha': [0.001, 0.01]}
}

if HAS_XGB:
    clf_param_grids["XGBoost Classifier"] = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.3]
    }
if HAS_LGBM:
    clf_param_grids["LightGBM Classifier"] = {
        'num_leaves': [31, 63, 127],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [100, 200]
    }
if HAS_CAT:
    clf_param_grids["CatBoost Classifier"] = {
        'depth': [4, 6, 8],
        'iterations': [100, 200, 300],
        'learning_rate': [0.01, 0.1]
    }

# =============================================
# SESSION STATE
# =============================================
if 'df' not in st.session_state:
    st.session_state.df = None
if 'target' not in st.session_state:
    st.session_state.target = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None
if 'dim_reduction' not in st.session_state:
    st.session_state.dim_reduction = None

# =============================================
# SIDEBAR
# =============================================
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="font-size: 2.5rem; margin: 0;">🤖</h1>
            <h2 style="color: white; margin: 0;">AutoML Pro</h2>
            <p style="color: #667eea; margin: 0; font-size: 0.9rem;">v3.0</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "📁 Upload Data", "⚙️ Configure", "🤖 Models", 
         "🧠 Deep Learning", "📈 Visualize", "📊 Compare", "📄 Report"],
        index=0
    )
    
    st.markdown("---")
    
    # System Status
    st.markdown("### 📊 System Status")
    
    # Model Count
    total_models = len(reg_models) + len(clf_models) + len(advanced_reg_models) + len(advanced_clf_models)
    st.markdown(f"**Models:** {total_models}+")
    
    if HAS_XGB:
        st.markdown("✅ XGBoost")
    else:
        st.markdown("❌ XGBoost (install)")
    if HAS_LGBM:
        st.markdown("✅ LightGBM")
    else:
        st.markdown("❌ LightGBM (install)")
    if HAS_CAT:
        st.markdown("✅ CatBoost")
    else:
        st.markdown("❌ CatBoost (install)")
    if HAS_TF:
        st.markdown("✅ TensorFlow")
    else:
        st.markdown("❌ TensorFlow (install)")
    
    st.markdown("---")
    
    if st.session_state.df is not None:
        st.markdown(f'<span class="status-badge status-success">✅ Data Loaded</span>', unsafe_allow_html=True)
        st.markdown(f"**Rows:** {st.session_state.df.shape[0]}")
        st.markdown(f"**Columns:** {st.session_state.df.shape[1]}")
    else:
        st.markdown(f'<span class="status-badge status-warning">⏳ No Data</span>', unsafe_allow_html=True)
    
    if st.session_state.target:
        st.markdown(f"**Target:** {st.session_state.target}")
    
    st.markdown("---")
    
    # Developer Info
    st.markdown("### 👨‍💻 About")
    st.markdown("© Developed by **Md Shoaib Ali**")
    st.markdown("National Institute of Technology Durgapur")
    st.markdown("---")
    st.markdown("📧 Contact: [GitHub](https://github.com)")

# =============================================
# PAGE: DASHBOARD
# =============================================
if page == "🏠 Dashboard":
    st.markdown("""
        <div class="main-header">
            <h1>🚀 AutoML Dashboard Pro</h1>
            <p>End-to-End Machine Learning with 35+ Models</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div class="step-card">
                <div class="step-number">1</div>
                <h4>📁 Upload</h4>
                <p style="color: #7f8c8d; font-size: 0.9rem;">Load your dataset</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="step-card">
                <div class="step-number">2</div>
                <h4>⚙️ Configure</h4>
                <p style="color: #7f8c8d; font-size: 0.9rem;">Set target & problem type</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="step-card">
                <div class="step-number">3</div>
                <h4>🤖 Train</h4>
                <p style="color: #7f8c8d; font-size: 0.9rem;">Train ML models</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div class="step-card">
                <div class="step-number">4</div>
                <h4>📊 Analyze</h4>
                <p style="color: #7f8c8d; font-size: 0.9rem;">View results</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Stats if data loaded
    if st.session_state.df is not None:
        df = st.session_state.df
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Rows", df.shape[0])
        with col2:
            st.metric("📑 Total Columns", df.shape[1])
        with col3:
            numeric = len(df.select_dtypes(include=[np.number]).columns)
            st.metric("🔢 Numeric", numeric)
        with col4:
            missing = df.isnull().sum().sum()
            st.metric("❌ Missing", missing, delta="⚠️" if missing > 0 else "✅")
        
        with st.expander("🔍 View Sample Data"):
            st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown("---")
    
    # Features
    st.subheader("🎯 Key Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="metric-card">
                <h3>🤖 35+ Models</h3>
                <p>Traditional, Ensemble, Advanced, and Deep Learning</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="metric-card">
                <h3>📈 Auto-Visualization</h3>
                <p>Interactive plots and real-time analytics</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="metric-card">
                <h3>🎯 One-Click Compare</h3>
                <p>Find the best model for your data</p>
            </div>
        """, unsafe_allow_html=True)

# =============================================
# PAGE: UPLOAD DATA
# =============================================
elif page == "📁 Upload Data":
    st.markdown("## 📁 Upload Your Dataset")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx'],
        help="Upload your dataset to get started"
    )
    
    if uploaded_file is not None:
        try:
            with st.spinner("Loading file..."):
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.df = df
                
                st.success(f"✅ Successfully loaded: {uploaded_file.name}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Rows", df.shape[0])
                with col2:
                    st.metric("📑 Columns", df.shape[1])
                with col3:
                    missing = df.isnull().sum().sum()
                    st.metric("❌ Missing", missing)
                
                st.markdown("---")
                
                tab1, tab2, tab3 = st.tabs(["📄 Data Preview", "📊 Statistics", "📈 Distributions"])
                
                with tab1:
                    st.dataframe(df.head(20), use_container_width=True)
                
                with tab2:
                    st.dataframe(df.describe(), use_container_width=True)
                
                with tab3:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        col = st.selectbox("Select column", numeric_cols)
                        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                        fig.update_layout(template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)
                        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# =============================================
# PAGE: CONFIGURE
# =============================================
elif page == "⚙️ Configure":
    st.markdown("## ⚙️ Configure Your Project")
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first")
        st.stop()
    
    df = st.session_state.df
    
    col1, col2 = st.columns(2)
    
    with col1:
        target = st.selectbox(
            "🎯 Select Target Column",
            options=df.columns.tolist()
        )
        st.session_state.target = target
    
    with col2:
        problem_type = st.selectbox(
            "📊 Select Problem Type",
            options=["Regression", "Classification"]
        )
        st.session_state.problem_type = problem_type
    
    st.markdown("---")
    
    # Target Analysis
    st.subheader("📊 Target Column Analysis")
    target_data = df[target]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Data Type", str(target_data.dtype))
    with col2:
        st.metric("Unique Values", target_data.nunique())
    with col3:
        st.metric("Missing Values", target_data.isnull().sum())
    
    fig = px.histogram(target_data, title=f"Distribution of {target}")
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    
    if st.button("✅ Apply Configuration", type="primary", use_container_width=True):
        st.success("✅ Configuration applied successfully!")

# =============================================
# PAGE: MODELS
# =============================================
elif page == "🤖 Models":
    st.markdown("## 🤖 Model Training")
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first")
        st.stop()
    
    if st.session_state.target is None:
        st.warning("⚠️ Please configure settings first")
        st.stop()
    
    df = st.session_state.df
    target = st.session_state.target
    problem_type = st.session_state.problem_type
    
    # Model Selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Combine all models
        if problem_type == "Regression":
            all_models = {**reg_models, **advanced_reg_models}
            model_names = list(all_models.keys())
            categories = [all_models[m]["category"] for m in model_names]
            selected_model = st.selectbox("🤖 Select Model", model_names)
            st.caption(all_models[selected_model]["description"])
        else:
            all_models = {**clf_models, **advanced_clf_models}
            model_names = list(all_models.keys())
            selected_model = st.selectbox("🤖 Select Model", model_names)
            st.caption(all_models[selected_model]["description"])
    
    with col2:
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
    
    # Advanced Settings
    with st.expander("⚙️ Advanced Settings", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            enable_tuning = st.checkbox("Hyperparameter Tuning", value=False)
            if enable_tuning:
                tune_method = st.radio("Method", ["Grid Search", "Random Search"])
        with col2:
            enable_cv = st.checkbox("Cross-Validation", value=False)
            if enable_cv:
                cv_folds = st.slider("CV Folds", 3, 10, 5)
        with col3:
            enable_scaling = st.checkbox("Feature Scaling", value=True)
            enable_pca = st.checkbox("PCA (Dimensionality Reduction)", value=False)
            if enable_pca:
                pca_components = st.slider("Components", 2, 20, 5)
    
    st.markdown("---")
    
    # Train Button
    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        with st.spinner("Training model..."):
            try:
                # Prepare data
                X = df.drop(columns=[target])
                y = df[target]
                
                # Encode categorical
                for col in X.columns:
                    if X[col].dtype == 'object':
                        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
                
                X = X.fillna(X.mean())
                
                if y.dtype == 'object':
                    y = LabelEncoder().fit_transform(y)
                
                # Feature Scaling
                if enable_scaling:
                    scaler = StandardScaler()
                    X = scaler.fit_transform(X)
                
                # PCA
                if enable_pca:
                    pca = PCA(n_components=min(pca_components, X.shape[1]))
                    X = pca.fit_transform(X)
                    st.session_state.dim_reduction = pca
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42,
                    stratify=y if problem_type == "Classification" and len(np.unique(y)) > 1 else None
                )
                
                # Get model
                model = all_models[selected_model]["model"]
                
                # Hyperparameter Tuning
                if enable_tuning:
                    param_grid = {}
                    if problem_type == "Regression":
                        if selected_model in reg_param_grids:
                            param_grid = reg_param_grids[selected_model]
                    else:
                        if selected_model in clf_param_grids:
                            param_grid = clf_param_grids[selected_model]
                    
                    if param_grid:
                        scoring = 'r2' if problem_type == "Regression" else 'accuracy'
                        if tune_method == "Grid Search":
                            search = GridSearchCV(model, param_grid, cv=3, scoring=scoring, n_jobs=-1)
                        else:
                            search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=3, scoring=scoring, random_state=42, n_jobs=-1)
                        search.fit(X_train, y_train)
                        model = search.best_estimator_
                        st.info(f"✅ Best parameters: {search.best_params_}")
                
                # Cross-Validation
                if enable_cv:
                    cv = StratifiedKFold(n_splits=cv_folds) if problem_type == "Classification" else KFold(n_splits=cv_folds)
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, 
                                               scoring='r2' if problem_type == "Regression" else 'accuracy')
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                else:
                    cv_mean = None
                    cv_std = None
                
                # Train
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Store results
                st.session_state.model_results = {
                    'model': model,
                    'model_name': selected_model,
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'X_test': X_test,
                    'problem_type': problem_type,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'feature_names': df.drop(columns=[target]).columns if not enable_pca else [f'PC{i+1}' for i in range(X.shape[1])]
                }
                
                # Display metrics
                st.markdown("### 📊 Model Performance")
                
                if problem_type == "Regression":
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("R² Score", f"{r2:.4f}")
                    col2.metric("RMSE", f"{rmse:.4f}")
                    col3.metric("MAE", f"{mae:.4f}")
                    if cv_mean:
                        col4.metric("CV Score", f"{cv_mean:.4f} ± {cv_std:.4f}")
                    
                    # Actual vs Predicted
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=y_test, mode='lines+markers', name='Actual',
                                            line=dict(color='royalblue', width=2)))
                    fig.add_trace(go.Scatter(y=y_pred, mode='lines+markers', name='Predicted',
                                            line=dict(color='tomato', width=2)))
                    fig.update_layout(title=f"Actual vs Predicted - {selected_model}",
                                     xaxis_title="Sample Index", yaxis_title="Target",
                                     template="plotly_white", hovermode='x')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Residual Plot
                    residuals = y_test - y_pred
                    fig2 = px.scatter(x=y_pred, y=residuals, title="Residual Plot")
                    fig2.add_hline(y=0, line_dash="dash", line_color="red")
                    fig2.update_layout(xaxis_title="Predicted", yaxis_title="Residual")
                    st.plotly_chart(fig2, use_container_width=True)
                    
                else:
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{acc:.4f}")
                    col2.metric("Precision", f"{prec:.4f}")
                    col3.metric("Recall", f"{rec:.4f}")
                    col4.metric("F1 Score", f"{f1:.4f}")
                    
                    # Confusion Matrix
                    cm = confusion_matrix(y_test, y_pred)
                    class_labels = [str(cls) for cls in np.unique(np.concatenate([y_test, y_pred]))]
                    
                    fig = px.imshow(cm, text_auto=True, x=class_labels, y=class_labels,
                                   color_continuous_scale='Blues', title=f"Confusion Matrix - {selected_model}")
                    fig.update_layout(template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Classification Report
                    st.subheader("📋 Classification Report")
                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df, use_container_width=True)
                
                st.success("✅ Model training completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# =============================================
# PAGE: DEEP LEARNING
# =============================================
elif page == "🧠 Deep Learning":
    st.markdown("## 🧠 Deep Learning")
    
    if not HAS_TF:
        st.warning("⚠️ TensorFlow is not installed. Please install: pip install tensorflow")
        st.stop()
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first")
        st.stop()
    
    if st.session_state.target is None:
        st.warning("⚠️ Please configure settings first")
        st.stop()
    
    df = st.session_state.df
    target = st.session_state.target
    problem_type = st.session_state.problem_type
    
    col1, col2 = st.columns(2)
    
    with col1:
        architecture = st.selectbox(
            "🧠 Architecture",
            ["ANN (Dense Layers)", "CNN (Convolutional)", "RNN (Recurrent)", "LSTM (Long Short-Term Memory)"]
        )
    
    with col2:
        epochs = st.slider("Epochs", 10, 200, 50, 10)
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
    
    with st.expander("⚙️ Advanced Settings"):
        col1, col2 = st.columns(2)
        with col1:
            hidden_layers = st.number_input("Hidden Layers", 1, 5, 2)
            neurons = st.number_input("Neurons per Layer", 16, 256, 64, 16)
        with col2:
            dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)
            learning_rate = st.select_slider("Learning Rate", [0.001, 0.01, 0.1, 0.2], value=0.001)
    
    if st.button("🧠 Train Deep Model", type="primary", use_container_width=True):
        with st.spinner("Training deep learning model..."):
            try:
                # Prepare data
                X = df.drop(columns=[target])
                y = df[target]
                
                for col in X.columns:
                    if X[col].dtype == 'object':
                        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
                
                X = X.fillna(X.mean())
                
                if y.dtype == 'object':
                    y = LabelEncoder().fit_transform(y)
                
                # Scale data
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42,
                    stratify=y if problem_type == "Classification" and len(np.unique(y)) > 1 else None
                )
                
                # Build model based on architecture
                if architecture == "ANN (Dense Layers)":
                    model = models.Sequential()
                    model.add(layers.Dense(neurons, activation='relu', input_shape=(X.shape[1],)))
                    for _ in range(hidden_layers - 1):
                        model.add(layers.Dense(neurons, activation='relu'))
                        model.add(layers.Dropout(dropout_rate))
                    
                    if problem_type == "Classification":
                        model.add(layers.Dense(len(np.unique(y)), activation='softmax'))
                        loss = 'sparse_categorical_crossentropy'
                        metrics = ['accuracy']
                    else:
                        model.add(layers.Dense(1, activation='linear'))
                        loss = 'mse'
                        metrics = ['mae']
                
                elif architecture == "CNN (Convolutional)":
                    model = models.Sequential()
                    # Reshape for CNN (assuming 1D data)
                    model.add(layers.Reshape((X.shape[1], 1), input_shape=(X.shape[1],)))
                    model.add(layers.Conv1D(32, 3, activation='relu'))
                    model.add(layers.MaxPooling1D(2))
                    model.add(layers.Flatten())
                    model.add(layers.Dense(64, activation='relu'))
                    model.add(layers.Dropout(dropout_rate))
                    
                    if problem_type == "Classification":
                        model.add(layers.Dense(len(np.unique(y)), activation='softmax'))
                        loss = 'sparse_categorical_crossentropy'
                        metrics = ['accuracy']
                    else:
                        model.add(layers.Dense(1, activation='linear'))
                        loss = 'mse'
                        metrics = ['mae']
                
                elif architecture in ["RNN (Recurrent)", "LSTM (Long Short-Term Memory)"]:
                    model = models.Sequential()
                    model.add(layers.Reshape((X.shape[1], 1), input_shape=(X.shape[1],)))
                    
                    if architecture == "LSTM (Long Short-Term Memory)":
                        model.add(layers.LSTM(neurons, return_sequences=True))
                        model.add(layers.LSTM(neurons))
                    else:
                        model.add(layers.SimpleRNN(neurons))
                    
                    model.add(layers.Dense(64, activation='relu'))
                    model.add(layers.Dropout(dropout_rate))
                    
                    if problem_type == "Classification":
                        model.add(layers.Dense(len(np.unique(y)), activation='softmax'))
                        loss = 'sparse_categorical_crossentropy'
                        metrics = ['accuracy']
                    else:
                        model.add(layers.Dense(1, activation='linear'))
                        loss = 'mse'
                        metrics = ['mae']
                
                # Compile model
                optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
                model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
                
                # Display model summary
                st.subheader("📋 Model Architecture")
                summary = []
                model.summary(print_fn=lambda x: summary.append(x))
                st.code('\n'.join(summary))
                
                # Train model
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                class ProgressCallback(keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        progress = (epoch + 1) / epochs
                        progress_bar.progress(progress)
                        status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {logs.get('loss', 0):.4f}")
                
                history = model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.2,
                    callbacks=[ProgressCallback()],
                    verbose=0
                )
                
                # Evaluate
                y_pred = model.predict(X_test)
                if problem_type == "Classification":
                    y_pred_class = np.argmax(y_pred, axis=1)
                else:
                    y_pred_class = y_pred.flatten()
                
                # Display results
                st.markdown("### 📊 Model Performance")
                
                if problem_type == "Regression":
                    r2 = r2_score(y_test, y_pred_class)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred_class))
                    mae = mean_absolute_error(y_test, y_pred_class)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("R² Score", f"{r2:.4f}")
                    col2.metric("RMSE", f"{rmse:.4f}")
                    col3.metric("MAE", f"{mae:.4f}")
                    
                    # Plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=y_test, mode='lines+markers', name='Actual'))
                    fig.add_trace(go.Scatter(y=y_pred_class, mode='lines+markers', name='Predicted'))
                    fig.update_layout(title="Actual vs Predicted", template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    acc = accuracy_score(y_test, y_pred_class)
                    prec = precision_score(y_test, y_pred_class, average='weighted', zero_division=0)
                    rec = recall_score(y_test, y_pred_class, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred_class, average='weighted', zero_division=0)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{acc:.4f}")
                    col2.metric("Precision", f"{prec:.4f}")
                    col3.metric("Recall", f"{rec:.4f}")
                    col4.metric("F1 Score", f"{f1:.4f}")
                    
                    # Confusion Matrix
                    cm = confusion_matrix(y_test, y_pred_class)
                    fig = px.imshow(cm, text_auto=True, title="Confusion Matrix", color_continuous_scale='Blues')
                    fig.update_layout(template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Training History
                st.subheader("📈 Training History")
                fig = make_subplots(rows=1, cols=2)
                
                fig.add_trace(go.Scatter(y=history.history['loss'], name='Training Loss'), row=1, col=1)
                fig.add_trace(go.Scatter(y=history.history['val_loss'], name='Validation Loss'), row=1, col=1)
                
                if 'accuracy' in history.history:
                    fig.add_trace(go.Scatter(y=history.history['accuracy'], name='Training Accuracy'), row=1, col=2)
                    fig.add_trace(go.Scatter(y=history.history['val_accuracy'], name='Validation Accuracy'), row=1, col=2)
                
                fig.update_layout(template="plotly_white", height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                st.success("✅ Deep Learning training completed!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# =============================================
# PAGE: VISUALIZE
# =============================================
elif page == "📈 Visualize":
    st.markdown("## 📈 Advanced Visualizations")
    
    if st.session_state.model_results is None:
        st.warning("⚠️ Please train a model first (Go to 'Models')")
        st.stop()
    
    results = st.session_state.model_results
    
    tabs = st.tabs(["📊 Main Plot", "📈 Feature Importance", "📉 Distribution", "📊 Correlation", "🔬 PCA"])
    
    with tabs[0]:
        if results['problem_type'] == "Regression":
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=results['y_test'], mode='lines+markers', name='Actual'))
            fig.add_trace(go.Scatter(y=results['y_pred'], mode='lines+markers', name='Predicted'))
            fig.update_layout(title=f"Actual vs Predicted - {results['model_name']}",
                             template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            cm = confusion_matrix(results['y_test'], results['y_pred'])
            class_labels = [str(cls) for cls in np.unique(np.concatenate([results['y_test'], results['y_pred']]))]
            fig = px.imshow(cm, text_auto=True, x=class_labels, y=class_labels,
                           color_continuous_scale='Blues', title=f"Confusion Matrix - {results['model_name']}")
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        if hasattr(results['model'], 'feature_importances_'):
            importances = results['model'].feature_importances_
            feature_names = results['feature_names'] if hasattr(results, 'feature_names') else [f'F{i}' for i in range(len(importances))]
            fi_df = pd.DataFrame({'Feature': feature_names[:len(importances)], 'Importance': importances})
            fi_df = fi_df.sort_values('Importance', ascending=True).tail(20)
            fig = px.bar(fi_df, y='Feature', x='Importance', title='Feature Importance (Top 20)',
                        orientation='h', color='Importance', color_continuous_scale='Viridis')
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance not available for this model")
    
    with tabs[2]:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=results['y_test'], name='Actual', opacity=0.7))
        fig.add_trace(go.Histogram(x=results['y_pred'], name='Predicted', opacity=0.7))
        fig.update_layout(title="Actual vs Predicted Distribution", barmode='overlay', template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:
        if st.session_state.df is not None:
            df = st.session_state.df
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr()
                fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap",
                               color_continuous_scale='RdBu', aspect='auto')
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
    
    with tabs[4]:
        if st.session_state.df is not None and st.session_state.target:
            df = st.session_state.df
            X = df.drop(columns=[st.session_state.target])
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 2:
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(X[numeric_cols])
                pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
                pca_df['Target'] = df[st.session_state.target]
                
                fig = px.scatter(pca_df, x='PC1', y='PC2', color='Target',
                                title=f'PCA: {pca.explained_variance_ratio_[0]:.2%} + {pca.explained_variance_ratio_[1]:.2%} variance explained')
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
                
                st.caption(f"Total explained variance: {sum(pca.explained_variance_ratio_):.2%}")

# =============================================
# PAGE: COMPARE
# =============================================
elif page == "📊 Compare":
    st.markdown("## 📊 Model Comparison")
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first")
        st.stop()
    
    if st.session_state.target is None:
        st.warning("⚠️ Please configure settings first")
        st.stop()
    
    df = st.session_state.df
    target = st.session_state.target
    problem_type = st.session_state.problem_type
    
    # Select models to compare
    if problem_type == "Regression":
        all_models = {**reg_models, **advanced_reg_models}
    else:
        all_models = {**clf_models, **advanced_clf_models}
    
    selected_models = st.multiselect(
        "Select models to compare",
        options=list(all_models.keys()),
        default=list(all_models.keys())[:5]
    )
    
    if st.button("📊 Compare Selected Models", type="primary", use_container_width=True):
        with st.spinner("Comparing models..."):
            try:
                # Prepare data
                X = df.drop(columns=[target])
                y = df[target]
                
                for col in X.columns:
                    if X[col].dtype == 'object':
                        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
                
                X = X.fillna(X.mean())
                if y.dtype == 'object':
                    y = LabelEncoder().fit_transform(y)
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, name in enumerate(selected_models):
                    status_text.text(f"Training {name}... ({i+1}/{len(selected_models)})")
                    try:
                        model = all_models[name]["model"]
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        if problem_type == "Regression":
                            score = r2_score(y_test, y_pred)
                            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                            mae = mean_absolute_error(y_test, y_pred)
                            results.append({
                                "Model": name,
                                "R² Score": f"{score:.4f}",
                                "RMSE": f"{rmse:.4f}",
                                "MAE": f"{mae:.4f}",
                                "Category": all_models[name]["category"],
                                "Status": "✅"
                            })
                        else:
                            acc = accuracy_score(y_test, y_pred)
                            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                            results.append({
                                "Model": name,
                                "Accuracy": f"{acc:.4f}",
                                "Precision": f"{prec:.4f}",
                                "Recall": f"{rec:.4f}",
                                "F1 Score": f"{f1:.4f}",
                                "Category": all_models[name]["category"],
                                "Status": "✅"
                            })
                    except Exception as e:
                        results.append({
                            "Model": name,
                            "Status": f"❌ {str(e)[:30]}"
                        })
                    
                    progress_bar.progress((i + 1) / len(selected_models))
                
                status_text.text("✅ Comparison complete!")
                
                df_results = pd.DataFrame(results)
                
                st.markdown("### 📊 Comparison Results")
                st.dataframe(df_results, use_container_width=True)
                
                # Bar chart
                if problem_type == "Regression":
                    metric = "R² Score"
                else:
                    metric = "Accuracy"
                
                df_success = df_results[df_results['Status'] == '✅'].copy()
                if not df_success.empty:
                    df_success[metric] = pd.to_numeric(df_success[metric], errors='coerce')
                    df_success = df_success.sort_values(metric, ascending=False)
                    
                    fig = px.bar(df_success, x='Model', y=metric, color='Category',
                                title=f"Model Comparison - {metric}",
                                text_auto='.4f', color_discrete_sequence=px.colors.qualitative.Set3)
                    fig.update_layout(xaxis_tickangle=-45, template="plotly_white", height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Best model
                    best = df_success.iloc[0]
                    st.success(f"""
                        🏆 **Best Model: {best['Model']}**
                        - **{metric}:** {best[metric]}
                        - **Category:** {best['Category']}
                    """)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# =============================================
# PAGE: REPORT
# =============================================
elif page == "📄 Report":
    st.markdown("## 📄 Generate Report")
    
    if st.session_state.model_results is None:
        st.warning("⚠️ Please train a model first (Go to 'Models')")
        st.stop()
    
    results = st.session_state.model_results
    df = st.session_state.df
    
    st.markdown("### 📋 Report Sections")
    
    sections = st.multiselect(
        "Select sections to include",
        ["Executive Summary", "Data Description", "Model Performance", 
         "Visualizations", "Feature Importance", "Recommendations", "Model Architecture"],
        default=["Executive Summary", "Model Performance", "Visualizations"]
    )
    
    if st.button("📄 Generate Report", type="primary", use_container_width=True):
        st.markdown("---")
        st.markdown("# 📊 AutoML Dashboard Pro Report")
        st.markdown(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.markdown("---")
        
        if "Executive Summary" in sections:
            st.markdown("## 📋 Executive Summary")
            st.markdown(f"""
                | Metric | Value |
                |--------|-------|
                | **Model Used** | {results['model_name']} |
                | **Problem Type** | {results['problem_type']} |
                | **Dataset Size** | {df.shape[0]} rows, {df.shape[1]} columns |
                | **Target Variable** | {st.session_state.target} |
            """)
            st.markdown("---")
        
        if "Data Description" in sections:
            st.markdown("## 📊 Data Description")
            st.dataframe(df.describe(), use_container_width=True)
            st.markdown("---")
        
        if "Model Performance" in sections:
            st.markdown("## 📈 Model Performance")
            
            if results['problem_type'] == "Regression":
                r2 = r2_score(results['y_test'], results['y_pred'])
                rmse = np.sqrt(mean_squared_error(results['y_test'], results['y_pred']))
                mae = mean_absolute_error(results['y_test'], results['y_pred'])
                
                col1, col2, col3 = st.columns(3)
                col1.metric("R² Score", f"{r2:.4f}")
                col2.metric("RMSE", f"{rmse:.4f}")
                col3.metric("MAE", f"{mae:.4f}")
            else:
                acc = accuracy_score(results['y_test'], results['y_pred'])
                prec = precision_score(results['y_test'], results['y_pred'], average='weighted', zero_division=0)
                rec = recall_score(results['y_test'], results['y_pred'], average='weighted', zero_division=0)
                f1 = f1_score(results['y_test'], results['y_pred'], average='weighted', zero_division=0)
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{acc:.4f}")
                col2.metric("Precision", f"{prec:.4f}")
                col3.metric("Recall", f"{rec:.4f}")
                col4.metric("F1 Score", f"{f1:.4f}")
            st.markdown("---")
        
        if "Visualizations" in sections:
            st.markdown("## 📊 Visualizations")
            
            if results['problem_type'] == "Regression":
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=results['y_test'], mode='lines+markers', name='Actual'))
                fig.add_trace(go.Scatter(y=results['y_pred'], mode='lines+markers', name='Predicted'))
                fig.update_layout(title="Actual vs Predicted", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            else:
                cm = confusion_matrix(results['y_test'], results['y_pred'])
                class_labels = [str(cls) for cls in np.unique(np.concatenate([results['y_test'], results['y_pred']]))]
                fig = px.imshow(cm, text_auto=True, x=class_labels, y=class_labels,
                               color_continuous_scale='Blues', title="Confusion Matrix")
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("---")
        
        if "Feature Importance" in sections:
            if hasattr(results['model'], 'feature_importances_'):
                st.markdown("## 📈 Feature Importance")
                importances = results['model'].feature_importances_
                feature_names = results.get('feature_names', [f'F{i}' for i in range(len(importances))])
                fi_df = pd.DataFrame({'Feature': feature_names[:len(importances)], 'Importance': importances})
                fi_df = fi_df.sort_values('Importance', ascending=True).tail(20)
                fig = px.bar(fi_df, y='Feature', x='Importance', title='Feature Importance',
                            orientation='h', color='Importance', color_continuous_scale='Viridis')
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("---")
        
        if "Recommendations" in sections:
            st.markdown("## 📋 Recommendations")
            
            if results['problem_type'] == "Regression":
                r2 = r2_score(results['y_test'], results['y_pred'])
                if r2 > 0.8:
                    st.success("✅ **Excellent Performance!** Consider deploying this model to production.")
                elif r2 > 0.6:
                    st.info("📊 **Good Performance.** Consider feature engineering for further improvement.")
                else:
                    st.warning("⚠️ **Moderate Performance.** Consider trying other algorithms or adding more data.")
            else:
                acc = accuracy_score(results['y_test'], results['y_pred'])
                if acc > 0.85:
                    st.success("✅ **Excellent Performance!** Consider deploying this model to production.")
                elif acc > 0.7:
                    st.info("📊 **Good Performance.** Consider feature engineering for further improvement.")
                else:
                    st.warning("⚠️ **Moderate Performance.** Consider trying other algorithms or adding more data.")
            st.markdown("---")
        
        if "Model Architecture" in sections:
            if hasattr(results['model'], 'get_params'):
                st.markdown("## 🏗️ Model Architecture")
                st.json(results['model'].get_params())
            else:
                st.info("Model architecture details not available for this model type")
        
        st.success("✅ Report generated successfully!")

# =============================================
# FOOTER
# =============================================
st.markdown("""
    <div class="sidebar-footer">
        © 2026 Developed by <strong>Md Shoaib Ali</strong> | 
        National Institute of Technology Durgapur | 
        🤖 AutoML Dashboard Pro v3.0
    </div>
""", unsafe_allow_html=True)