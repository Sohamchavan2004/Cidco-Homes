# CIDCO_Homes_Dashboard.py
# Streamlit app for CIDCO Housing Price Prediction with Responsible AI

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import io
import shap
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Optional: XGBoost
try:
    from xgboost import XGBRegressor
    has_xgb = True
except Exception:
    has_xgb = False

# LIME
try:
    from lime.lime_tabular import LimeTabularExplainer
    has_lime = True
except Exception:
    has_lime = False

# Fairlearn
try:
    from fairlearn.metrics import MetricFrame
    from sklearn.metrics import mean_absolute_error as mae
    has_fairlearn = True
except Exception:
    has_fairlearn = False

st.set_page_config(page_title="CIDCO Homes: Price Prediction & Responsible AI", layout="wide")
st.title("🏘️ CIDCO Housing Price Prediction Dashboard")

# Problem Statement Section
st.markdown("""
### 📋 Problem Statement

The City and Industrial Development Corporation (CIDCO) aims to develop an intelligent system to predict housing prices for properties across the Navi Mumbai region. The main objective is to build a machine learning solution that can accurately estimate property prices based on key features such as location, carpet area, distance from railway stations, and property categorization.  

In addition to accurate predictions, the system should provide transparency by using interpretable AI techniques like SHAP and LIME to explain individual price predictions. It should also ensure fairness across different applicant categories, including General, SC, ST, and Religious Minorities, as well as across various locations. By doing so, the system can assist decision-makers in setting pricing strategies and formulating policies for affordable housing schemes.  

This dashboard presents an analysis of CIDCO housing data using state-of-the-art machine learning models. It focuses on multiple aspects: predictive accuracy through models such as Ridge Regression, Random Forest, and XGBoost; explainability using SHAP for global feature importance and LIME for local, instance-level explanations; fairness audits to detect and mitigate potential biases across sensitive demographic attributes; and readiness for deployment with API code generation for production integration.
""")


st.markdown("---")

# -----------------------
# Utilities & Example data
# -----------------------
@st.cache_data
def load_example_data():
    """Load sample CIDCO data structure"""
    np.random.seed(42)
    n = 200
    
    locations = ['Kharghar Station', 'Panvel (W) Bus Terminus', 'Kalamboli Bus Depot', 
                 'Bamandongri', 'Kharkopar Plot 3 Sector 16']
    categories = ['General', 'SC (SCHEDULED CASTE)', 'ST (SCHEDULED TRIBES)', 
                  'RM (RELIGIOUS MINORITIES)', 'MATHADI KAMGAR', 'PHYSICALLY DISABLED']
    types = ['Ews', 'Lig', 'Lig A', 'Lig B']
    
    df = pd.DataFrame({
        'Distance_from_Railway_Station_km': np.random.uniform(0.5, 8, n),
        'Carpet_Area_sqft': np.random.choice([322, 398, 540], n),
        'No_of_Towers': np.random.randint(2, 45, n),
        'category': np.random.choice(categories, n),
        'Location': np.random.choice(locations, n),
        'Type': np.random.choice(types, n),
        'has_rera': np.random.choice([0, 1], n, p=[0.05, 0.95]),
        'location_popularity': np.random.choice([58, 59, 60, 62, 64, 65, 67, 180], n),
        'price_per_sqft': np.random.uniform(7000, 24000, n),
        'distance_category': np.random.choice(['near', 'moderate', 'far'], n),
    })
    
    # Generate price with realistic relationships
    base_price = 35
    df['Price_lakhs'] = (
        base_price + 
        -2 * df['Distance_from_Railway_Station_km'] +
        0.05 * df['Carpet_Area_sqft'] +
        0.3 * df['No_of_Towers'] +
        0.001 * df['price_per_sqft'] +
        np.random.normal(0, 5, n)
    ).clip(20, 100)
    
    return df

def build_preprocessor(numeric_features, categorical_features):
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', LabelEncoder())
    ])
    
    # For multiple categorical features, we'll handle encoding manually
    return numeric_features, categorical_features

def encode_categorical(df, cat_cols):
    """Encode categorical columns using LabelEncoder"""
    df_encoded = df.copy()
    encoders = {}
    for col in cat_cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le
    return df_encoded, encoders

# -----------------------
# Sidebar: Upload / Config
# -----------------------
st.sidebar.header("1️⃣ Data & Settings")
upload_format = st.sidebar.radio("Data source", ['Use example data', 'Upload CSV file'])

if upload_format == 'Use example data':
    df = load_example_data()
    st.sidebar.success("✅ Using example CIDCO data")
else:
    uploaded_file = st.sidebar.file_uploader("Upload CIDCO dataset CSV", type=['csv'])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"✅ Loaded {df.shape[0]} records")
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()
    else:
        st.info("📤 No file uploaded — using example dataset.")
        df = load_example_data()

st.sidebar.write(f"**Dataset:** {df.shape[0]} rows × {df.shape[1]} columns")

# -----------------------
# EDA Section
# -----------------------
st.header("📊 Exploratory Data Analysis")

st.markdown("""
Understanding the data distribution and relationships is crucial for building accurate predictive models. 
The following visualizations provide insights into property prices, geographical factors, and feature correlations 
within the CIDCO housing dataset.
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Dataset Preview")
    st.markdown("*First 10 rows of the CIDCO housing dataset showing key features*")
    st.dataframe(df.head(10))

with col2:
    st.subheader("Summary Statistics")
    st.markdown("*Statistical measures (mean, median, standard deviation) for numerical features*")
    st.dataframe(df.describe().T)

# Distribution plots
st.subheader("🔍 Feature Distributions")
st.markdown("*Visual analysis of key features affecting property prices in CIDCO projects*")

tab1, tab2, tab3 = st.tabs(["Price Distribution", "Distance Analysis", "Location Popularity"])

with tab1:
    fig, ax = plt.subplots(figsize=(10, 5))
    if 'Price_lakhs' in df.columns:
        sns.histplot(df['Price_lakhs'], bins=30, kde=True, ax=ax, color='steelblue')
        ax.set_title("Distribution of Property Prices in CIDCO Housing Projects", fontsize=14, fontweight='bold')
        ax.set_xlabel("Price (Lakhs INR)", fontsize=12)
        ax.set_ylabel("Frequency (Number of Properties)", fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        # Add descriptive text
        mean_price = df['Price_lakhs'].mean()
        median_price = df['Price_lakhs'].median()
        ax.text(0.98, 0.98, f'Mean: ₹{mean_price:.2f}L\nMedian: ₹{median_price:.2f}L', 
                transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=10)
        
        st.pyplot(fig)
        st.caption("📊 This histogram shows the frequency distribution of property prices. The curve (KDE) represents the probability density, helping identify the most common price ranges.")

with tab2:
    fig, ax = plt.subplots(figsize=(10, 5))
    if 'Distance_from_Railway_Station_km' in df.columns and 'Price_lakhs' in df.columns:
        scatter = ax.scatter(df['Distance_from_Railway_Station_km'], df['Price_lakhs'], 
                           alpha=0.6, c=df['Price_lakhs'], cmap='viridis', s=50)
        ax.set_xlabel("Distance from Railway Station (kilometers)", fontsize=12)
        ax.set_ylabel("Price (Lakhs INR)", fontsize=12)
        ax.set_title("Impact of Railway Station Proximity on Property Prices", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Price (Lakhs)', rotation=270, labelpad=20)
        
        st.pyplot(fig)
        st.caption("🚉 This scatter plot reveals the relationship between distance from railway stations and property prices. Proximity to transport hubs typically correlates with higher property values due to better connectivity.")

with tab3:
    if 'location_popularity' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        popularity_counts = df['location_popularity'].value_counts().sort_index()
        bars = popularity_counts.plot(kind='bar', ax=ax, color='coral', edgecolor='black')
        ax.set_title("Distribution of Location Popularity Scores Across CIDCO Projects", fontsize=14, fontweight='bold')
        ax.set_xlabel("Popularity Score", fontsize=12)
        ax.set_ylabel("Number of Properties", fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fontsize=9)
        
        st.pyplot(fig)
        st.caption("📍 Location popularity score indicates demand and desirability. Higher scores typically reflect areas with better amenities, infrastructure, and social factors.")

# Correlation heatmap
if st.checkbox("Show Correlation Heatmap"):
    st.subheader("🔥 Correlation Matrix")
    st.markdown("*Correlation coefficients between numerical features - values closer to +1 or -1 indicate stronger relationships*")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        fig, ax = plt.subplots(figsize=(12, 8))
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=ax, square=True, linewidths=1)
        ax.set_title("Feature Correlation Heatmap - CIDCO Housing Dataset", fontsize=14, fontweight='bold', pad=20)
        st.pyplot(fig)
        st.caption("🔢 Positive correlations (red) suggest features move together, while negative correlations (blue) indicate inverse relationships. This helps identify multicollinearity and feature importance.")

# -----------------------
# Feature Engineering
# -----------------------
st.header("⚙️ Feature Configuration")

st.markdown("""
The model uses both **numerical** and **categorical** features. Numerical features are standardized, 
while categorical features are encoded to numerical representations for machine learning algorithms.
""")

# Define features based on CIDCO dataset
available_numeric = ['Distance_from_Railway_Station_km', 'Carpet_Area_sqft', 
                     'No_of_Towers', 'price_per_sqft', 'location_popularity']
available_categorical = ['category', 'Location', 'Type', 'distance_category']

# Filter available columns
numeric_features = [col for col in available_numeric if col in df.columns]
categorical_features = [col for col in available_categorical if col in df.columns]

st.write(f"**Numeric features ({len(numeric_features)}):** {', '.join(numeric_features)}")
st.write(f"**Categorical features ({len(categorical_features)}):** {', '.join(categorical_features)}")

# Target variable
target_col = 'Price_lakhs'
if target_col not in df.columns:
    st.error(f"❌ Target column '{target_col}' not found in dataset!")
    st.stop()

# -----------------------
# Model Testing Section
# -----------------------
st.sidebar.header("2️⃣ Model Testing")
train_size = st.sidebar.slider('Test set size (%)', 50, 90, 80) / 100
random_state = st.sidebar.number_input('Random seed', value=42, step=1)
run_train = st.sidebar.button('🚀 Test Model', type="primary")

# Prepare data
X = df[numeric_features + categorical_features].copy()
y = df[target_col].copy()

# Handle missing values in target
valid_mask = y.notna()
X = X[valid_mask]
y = y[valid_mask]

# Encode categorical features
X_encoded, encoders = encode_categorical(X, categorical_features)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, train_size=train_size, random_state=int(random_state)
)

models = None
results_df = None

if run_train:
    st.header("🤖 Model Test Results")
    
    st.markdown("""
    Three regression models are trained and compared: **Ridge Regression** (linear with regularization), 
    **Random Forest** (ensemble of decision trees), and **XGBoost** (gradient boosting). 
    Performance is evaluated using MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), and R² (coefficient of determination).
    """)
    
    with st.spinner('Testing models... Please wait'):
        try:
            models = {}
            results = []
            
            # Ridge Regression
            with st.spinner('Testing Ridge Regression...'):
                ridge = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', RidgeCV(alphas=np.logspace(-3, 3, 7), cv=5))
                ])
                ridge.fit(X_train, y_train)
                ridge_pred = ridge.predict(X_test)
                models['Ridge'] = ridge
                results.append({
                    'Model': 'Ridge Regression',
                    'MAE': mean_absolute_error(y_test, ridge_pred),
                    'RMSE': np.sqrt(mean_squared_error(y_test, ridge_pred)),
                    'R²': r2_score(y_test, ridge_pred)
                })
            
            # Random Forest
            with st.spinner('Testing Random Forest...'):
                rf = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
                ])
                rf.fit(X_train, y_train)
                rf_pred = rf.predict(X_test)
                models['RandomForest'] = rf
                results.append({
                    'Model': 'Random Forest',
                    'MAE': mean_absolute_error(y_test, rf_pred),
                    'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
                    'R²': r2_score(y_test, rf_pred)
                })
            
            # XGBoost (if available)
            if has_xgb:
                with st.spinner('Testing XGBoost...'):
                    xgb = Pipeline([
                        ('scaler', StandardScaler()),
                        ('model', XGBRegressor(
                            n_estimators=200, learning_rate=0.1, 
                            random_state=42, verbosity=0
                        ))
                    ])
                    xgb.fit(X_train, y_train)
                    xgb_pred = xgb.predict(X_test)
                    models['XGBoost'] = xgb
                    results.append({
                        'Model': 'XGBoost',
                        'MAE': mean_absolute_error(y_test, xgb_pred),
                        'RMSE': np.sqrt(mean_squared_error(y_test, xgb_pred)),
                        'R²': r2_score(y_test, xgb_pred)
                    })
            
            results_df = pd.DataFrame(results).sort_values('R²', ascending=False)
            st.success("✅ Testing completed successfully!")
            
        except Exception as e:
            import traceback
            st.error(f"❌ Testing failed: {e}")
            st.code(traceback.format_exc())
            st.stop()

# -----------------------
# Results & Explanations
# -----------------------
if models is not None and results_df is not None:
    
    st.subheader("📈 Model Performance Comparison")
    st.markdown("*Lower MAE and RMSE values indicate better accuracy, while higher R² indicates better fit*")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(results_df.set_index('Model').style.highlight_max(axis=0, color='lightgreen'))
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 5))
        results_df.plot(x='Model', y=['MAE', 'RMSE'], kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4'])
        ax.set_title("Model Error Comparison - Lower is Better", fontsize=14, fontweight='bold')
        ax.set_ylabel("Error (Lakhs INR)", fontsize=12)
        ax.set_xlabel("Machine Learning Model", fontsize=12)
        ax.legend(title="Error Metrics", fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=0)
        st.pyplot(fig)
        st.caption("📊 Comparison of prediction errors across models. MAE represents average absolute error, while RMSE penalizes larger errors more heavily.")
    
    # Model selection for explanations
    st.header("🔍 Model Interpretability")
    st.markdown("""
    Understanding *why* the model makes certain predictions is crucial for trust and accountability. 
    We use **SHAP** (global explanations) and **LIME** (local explanations) to interpret model decisions.
    """)
    
    model_choice = st.selectbox('Select model for detailed analysis', options=list(models.keys()))
    selected_model = models[model_choice]
    
    # Predictions sample
    st.subheader("🎯 Sample Predictions")
    st.markdown("*Comparing actual vs predicted prices on test data to assess model accuracy*")
    
    n_samples = min(10, len(X_test))
    sample_preds = selected_model.predict(X_test.iloc[:n_samples])
    
    pred_df = pd.DataFrame({
        'Actual Price': y_test.iloc[:n_samples].values,
        'Predicted Price': sample_preds,
        'Error': y_test.iloc[:n_samples].values - sample_preds
    })
    pred_df['Error %'] = (pred_df['Error'] / pred_df['Actual Price'] * 100).round(2)
    st.dataframe(pred_df.style.background_gradient(cmap='RdYlGn', subset=['Error %']))
    st.caption("🎯 Green indicates underestimation, red indicates overestimation. Percentage error shows relative accuracy.")

    # SHAP Explanations
    st.subheader("🌟 SHAP Feature Importance")
    st.markdown("""
    **SHAP (SHapley Additive exPlanations)** values show how much each feature contributes to predictions. 
    Positive SHAP values increase the prediction, while negative values decrease it. 
    This provides a unified measure of global feature importance across the entire dataset.
    """)

    with st.expander("Show SHAP Analysis", expanded=True):
        try:
            # Get the actual model from pipeline
            actual_model = selected_model.named_steps['model']
            
            # For tree-based models, use TreeExplainer with transformed data
            if 'RandomForest' in model_choice or 'XGBoost' in model_choice:
                # Transform the data using the scaler from the pipeline
                scaler = selected_model.named_steps['scaler']
                X_test_scaled = scaler.transform(X_test.iloc[:100])
                
                explainer = shap.TreeExplainer(actual_model)
                shap_values = explainer.shap_values(X_test_scaled)
                
                # Summary plot
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X_test_scaled, 
                                feature_names=X_test.columns.tolist(), show=False)
                plt.title("SHAP Summary Plot - Feature Impact on Price Predictions", fontsize=14, fontweight='bold', pad=20)
                st.pyplot(fig)
                plt.close()
                st.caption("🎨 Each dot represents a property. Color indicates feature value (red=high, blue=low). Position shows impact on prediction.")
                
                # Feature importance bar plot
                st.write("**Top Features by Importance**")
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X_test_scaled, 
                                plot_type="bar", 
                                feature_names=X_test.columns.tolist(), 
                                show=False)
                plt.title("Mean Absolute SHAP Values - Overall Feature Importance", fontsize=14, fontweight='bold', pad=20)
                st.pyplot(fig2)
                plt.close()
                st.caption("📊 Bar length represents average impact magnitude. Longer bars indicate more influential features.")
                
            else:  # Ridge or other linear models
                st.info("Using Linear SHAP for Ridge regression...")
                
                # Extract components from pipeline
                scaler = selected_model.named_steps['scaler']
                ridge_model = selected_model.named_steps['model']
                
                # Transform data manually
                X_test_scaled = scaler.transform(X_test.iloc[:100])
                X_train_scaled = scaler.transform(X_train)
                
                # Use LinearExplainer (much faster and designed for linear models)
                try:
                    # For Ridge with multiple alphas tested, get the best estimator
                    if hasattr(ridge_model, 'coef_'):
                        explainer = shap.LinearExplainer(
                            ridge_model, 
                            X_train_scaled,
                            feature_names=X_test.columns.tolist()
                        )
                        shap_values = explainer.shap_values(X_test_scaled)
                        
                        # Summary plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        shap.summary_plot(shap_values, X_test_scaled, 
                                        feature_names=X_test.columns.tolist(), 
                                        show=False)
                        plt.title("SHAP Summary Plot - Linear Model Feature Contributions", fontsize=14, fontweight='bold', pad=20)
                        st.pyplot(fig)
                        plt.close()
                        st.caption("📈 Linear model SHAP values show consistent directional impact of each feature.")
                    else:
                        raise AttributeError("Model doesn't have coefficients")
                        
                except Exception as e:
                    # Fallback: Manual feature importance from coefficients
                    st.info(f"Using coefficient-based importance (LinearExplainer failed: {e})")
                    
                    coefficients = ridge_model.coef_
                    feature_importance = pd.DataFrame({
                        'Feature': X_test.columns.tolist(),
                        'Coefficient': coefficients,
                        'Abs_Coefficient': np.abs(coefficients)
                    }).sort_values('Abs_Coefficient', ascending=True)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    feature_importance.plot(x='Feature', y='Coefficient', 
                                        kind='barh', ax=ax, legend=False, color='steelblue')
                    ax.set_xlabel('Ridge Coefficient Value', fontsize=12)
                    ax.set_title('Feature Importance - Ridge Model Coefficients', fontsize=14, fontweight='bold')
                    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
                    ax.grid(axis='x', alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
                    st.caption("📊 Positive coefficients increase price, negative decrease it. Magnitude shows strength of effect.")
                
        except Exception as e:
            import traceback
            st.warning(f"SHAP visualization failed: {str(e)}")
            st.info("Try installing/updating required packages:\n```\npip install shap matplotlib numpy --upgrade\n```")
            if st.checkbox("Show error details"):
                st.code(traceback.format_exc())
    
    # LIME Local Explanations
    st.subheader("🧪 LIME - Local Explanations")
    st.markdown("""
    **LIME (Local Interpretable Model-agnostic Explanations)** explains individual predictions by fitting 
    a simple interpretable model around a specific instance. This shows which features most influenced 
    a particular price prediction, helping stakeholders understand individual cases.
    """)
    
    if has_lime:
        with st.expander("Explain Individual Predictions"):
            st.markdown("*Analyzing a random test instance to understand factors driving its price prediction*")
            
            # Automatically select a random instance instead of slider
            instance_idx = np.random.randint(0, len(X_test))
            st.info(f"Explaining randomly selected test instance #{instance_idx}")
            
            try:
                explainer = LimeTabularExplainer(
                    X_train.values,
                    feature_names=X_train.columns.tolist(),
                    mode='regression',
                    verbose=False
                )
                
                exp = explainer.explain_instance(
                    X_test.iloc[instance_idx].values,
                    selected_model.predict,
                    num_features=8
                )
                
                # Display explanation
                st.write(f"**Prediction Explanation for Instance {instance_idx}**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Actual Price", f"₹{y_test.iloc[instance_idx]:.2f} Lakhs")
                with col2:
                    predicted = selected_model.predict(X_test.iloc[[instance_idx]])[0]
                    st.metric("Predicted Price", f"₹{predicted:.2f} Lakhs")
                
                fig = exp.as_pyplot_figure()
                fig.set_size_inches(10, 6)
                plt.title(f"LIME Explanation - Feature Contributions for Instance {instance_idx}", 
                         fontsize=14, fontweight='bold', pad=20)
                st.pyplot(fig)
                st.caption("🔍 Orange bars push prediction higher, blue bars lower. Length shows contribution magnitude.")
                
                if st.button("🔄 Explain Another Random Instance"):
                    st.rerun()
                
            except Exception as e:
                st.error(f"LIME explanation failed: {e}")
    else:
        st.info("LIME not installed. Install with: `pip install lime`")
    
    # Fairness Audit
    st.header("⚖️ Fairness Audit")
    st.markdown("""
    **Fairness analysis** ensures the model performs equitably across different demographic groups. 
    We examine prediction accuracy across applicant categories (General, SC, ST, etc.) to detect 
    potential bias. Significant disparities may indicate the need for model adjustment or additional fairness constraints.
    """)
    
    if has_fairlearn and 'category' in X_test.columns:
        with st.expander("Fairness Analysis by Applicant Category", expanded=True):
            try:
                predictions = selected_model.predict(X_test)
                sensitive_feature = X_test['category'] if 'category' in X_test.columns else X_test['Location']
                
                # Create MetricFrame
                mf = MetricFrame(
                    metrics={
                        'MAE': mean_absolute_error,
                        'R²': r2_score
                    },
                    y_true=y_test,
                    y_pred=predictions,
                    sensitive_features=sensitive_feature
                )
                
                st.write("**Overall Model Performance:**")
                col1, col2 = st.columns(2)
                col1.metric("Mean Absolute Error", f"{mean_absolute_error(y_test, predictions):.2f} Lakhs")
                col2.metric("R² Score", f"{r2_score(y_test, predictions):.3f}")
                
                st.write("**Group-wise Performance Breakdown:**")
                st.markdown("*Performance metrics segmented by applicant category to identify disparities*")
                st.dataframe(mf.by_group.style.background_gradient(cmap='RdYlGn'))
                
                # Visualize disparities
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                mf.by_group['MAE'].plot(kind='bar', ax=ax1, color='coral', edgecolor='black')
                ax1.set_title('Mean Absolute Error by Applicant Category', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Mean Absolute Error (Lakhs INR)', fontsize=11)
                ax1.set_xlabel('Applicant Category', fontsize=11)
                ax1.tick_params(axis='x', rotation=45)
                ax1.grid(axis='y', alpha=0.3)
                
                mf.by_group['R²'].plot(kind='bar', ax=ax2, color='skyblue', edgecolor='black')
                ax2.set_title('R² Score by Applicant Category', fontsize=12, fontweight='bold')
                ax2.set_ylabel('R² Score (Higher is Better)', fontsize=11)
                ax2.set_xlabel('Applicant Category', fontsize=11)
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(axis='y', alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                st.caption("📊 Fairness metrics across demographic groups. Consistent bars indicate equitable performance; large variations suggest potential bias.")
                
                # Disparity metrics
                st.write("**Disparity Analysis:**")
                st.markdown("*Measuring performance variation across groups - lower values indicate more equitable predictions*")
                
                mae_diff = mf.difference()['MAE']
                r2_diff = mf.difference()['R²']
                
                col1, col2 = st.columns(2)
                col1.metric("MAE Difference (max-min)", f"{mae_diff:.3f} Lakhs", 
                           help="Maximum difference in MAE between any two groups")
                col2.metric("R² Difference (max-min)", f"{r2_diff:.3f}",
                           help="Maximum difference in R² between any two groups")
                
                if mae_diff > 5:
                    st.warning("⚠️ Significant MAE disparity detected across groups! Consider bias mitigation techniques.")
                else:
                    st.success("✅ MAE disparity is within acceptable range - model shows fair performance across groups")
                
            except Exception as e:
                st.error(f"Fairness audit failed: {e}")
    else:
        st.info("Fairlearn not installed or 'category' column not available. Install with: `pip install fairlearn`")
    
    # Model Deployment
    st.header("💾 Model Deployment")
    st.markdown("""
    Deploy your trained model to production using the tools below. Download the model file for local use, 
    or generate API code for cloud deployment with FastAPI and Docker.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Save Trained Model")
        model_filename = st.text_input('Model filename', value=f'cidco_{model_choice.lower()}_model.pkl')
        
        if st.button('💾 Download Model'):
            try:
                buffer = io.BytesIO()
                pickle.dump(selected_model, buffer)
                buffer.seek(0)
                st.download_button(
                    label="⬇️ Download Model File",
                    data=buffer,
                    file_name=model_filename,
                    mime='application/octet-stream'
                )
                st.success("Model ready for download!")
            except Exception as e:
                st.error(f"Failed to save model: {e}")
    
    with col2:
        st.subheader("API Deployment Code")
        if st.button('📝 Generate FastAPI Code'):
            api_code = f"""
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np

class PredictionRequest(BaseModel):
    Distance_from_Railway_Station_km: float
    Carpet_Area_sqft: float
    No_of_Towers: int
    price_per_sqft: float
    location_popularity: int
    category: str
    Location: str
    Type: str
    distance_category: str

app = FastAPI(title="CIDCO Home Price Prediction API")

# Load model
model = pickle.load(open('{model_filename}', 'rb'))

@app.get('/health')
def health_check():
    return {{"status": "healthy", "model": "{model_choice}"}}

@app.post('/predict')
def predict_price(request: PredictionRequest):
    # Convert request to DataFrame
    data = pd.DataFrame([request.dict()])
    
    # Make prediction
    prediction = model.predict(data)[0]
    
    return {{
        "predicted_price_lakhs": round(float(prediction), 2),
        "model_used": "{model_choice}"
    }}

# Run with: uvicorn api:app --reload
"""
            st.download_button(
                label="⬇️ Download api.py",
                data=api_code,
                file_name='cidco_api.py',
                mime='text/plain'
            )
            
            dockerfile = """
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY api.py .
COPY *.pkl .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
"""
            st.download_button(
                label="⬇️ Download Dockerfile",
                data=dockerfile,
                file_name='Dockerfile',
                mime='text/plain'
            )
            
            requirements = """
fastapi==0.104.1
uvicorn==0.24.0
pandas==2.1.3
numpy==1.26.2
scikit-learn==1.3.2
pydantic==2.5.0
"""
            st.download_button(
                label="⬇️ Download requirements.txt",
                data=requirements,
                file_name='requirements.txt',
                mime='text/plain'
            )

else:
    st.info("👈 Configure settings in the sidebar and click 'Train Models' to begin analysis")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📚 About
This dashboard implements:
- **EDA** from Experiment 3
- **ML Pipeline** from Experiment 4
- **Responsible AI** from Experiment 5

**Features:**
- Price prediction for CIDCO homes
- Model interpretability (SHAP & LIME)
- Fairness audit across demographics
- Ready-to-deploy API code

**Tech Stack:** Scikit-learn, XGBoost, SHAP, LIME, Fairlearn

---
**Accessibility:** All visualizations include descriptive text and captions for screen readers and assistive technologies.
""")
