import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import time
from PIL import Image

# Set custom theme
st.set_page_config(
    page_title="Wine Quality Classification",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional Color Palette
COLORS = {
    "night_bordeaux": "#590d22",
    "dark_amaranth": "#800f2f",
    "cherry_rose": "#a4133c",
    "rosewood": "#c9184a",
    "bubblegum_pink": "#ff4d6d",
    "bubblegum_pink_2": "#ff758f",
    "cotton_candy": "#ff8fa3",
    "cherry_blossom": "#ffb3c1",
    "pastel_pink": "#ffccd5",
    "lavender_blush": "#fff0f3",
    "white": "#FFFFFF",
    "light_gray": "#f4f3ee",
    "text_dark": "#2C3E50",
    "text_light": "#7F8C8D"
}

# Enhanced Custom CSS
st.markdown(f"""
    <style>
    * {{
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }}
    
    html, body, .main {{
        background: {COLORS['light_gray']} !important;
    }}
    
    .main {{
        padding: 30px 40px;
        background: {COLORS['light_gray']};
        min-height: 100vh;
    }}
    
    /* Header Container */
    .header {{
        display: flex;
        align-items: center;
        gap: 30px;
        margin-bottom: 40px;
        padding: 35px 40px;
        background: linear-gradient(135deg, {COLORS['night_bordeaux']} 0%, {COLORS['cherry_rose']} 100%);
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(89, 13, 34, 0.2);
    }}
    
    .header-content {{
        flex: 1;
    }}
    
    .header-title {{
        color: white;
        font-size: 2.8rem;
        font-weight: 900;
        margin: 0;
        letter-spacing: 1px;
    }}
    
    .header-subtitle {{
        color: {COLORS['cotton_candy']};
        font-size: 1.2rem;
        font-weight: 700;
        margin: 10px 0 0 0;
    }}
    
    /* Top Navigation Tabs */
    .nav-container {{
        display: flex;
        gap: 20px;
        margin-bottom: 30px;
        border-bottom: 3px solid {COLORS['pastel_pink']};
        padding-bottom: 0;
    }}
    
    .nav-tab {{
        padding: 16px 28px;
        background: transparent;
        border: none;
        color: {COLORS['text_light']};
        font-size: 1rem;
        font-weight: 800;
        cursor: pointer;
        border-bottom: 4px solid transparent;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }}
    
    .nav-tab:hover {{
        color: {COLORS['cherry_rose']};
        border-bottom-color: {COLORS['bubblegum_pink']};
    }}
    
    .nav-tab.active {{
        color: {COLORS['night_bordeaux']};
        border-bottom-color: {COLORS['cherry_rose']};
    }}
    
    /* Material Design Cards */
    .material-card {{
        background: {COLORS['white']};
        border-radius: 16px;
        padding: 28px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(0,0,0,0.04);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
        min-height: 240px;
    }}
    
    .material-card:hover {{
        box-shadow: 0 12px 24px rgba(89, 13, 34, 0.1);
        transform: translateY(-6px);
    }}
    
    .card-icon {{
        font-size: 3.5rem;
        margin-bottom: 15px;
        line-height: 1;
    }}
    
    .card-title {{
        color: {COLORS['night_bordeaux']};
        font-size: 1.3rem;
        font-weight: 800;
        margin-bottom: 12px;
    }}
    
    .card-description {{
        color: {COLORS['text_light']};
        font-size: 0.95rem;
        line-height: 1.6;
        font-weight: 500;
    }}
    
    /* Metric Cards - Dark */
    .metric-card {{
        background: linear-gradient(135deg, {COLORS['night_bordeaux']} 0%, {COLORS['cherry_rose']} 100%);
        color: white;
        padding: 28px 24px;
        border-radius: 14px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(89, 13, 34, 0.15);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 10px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(255,255,255,0.1);
    }}
    
    .metric-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 10px 24px rgba(89, 13, 34, 0.25);
    }}
    
    .metric-label {{
        font-size: 0.8rem;
        opacity: 0.95;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }}
    
    .metric-value {{
        font-size: 2.4rem;
        font-weight: 900;
        letter-spacing: -1px;
    }}
    
    .metric-subtext {{
        font-size: 0.75rem;
        opacity: 0.85;
        font-weight: 600;
    }}
    
    /* Metric Cards - Light */
    .metric-card-light {{
        background: linear-gradient(135deg, {COLORS['cotton_candy']} 0%, {COLORS['cherry_blossom']} 100%);
        color: {COLORS['night_bordeaux']};
    }}
    
    /* Upload Container */
    .upload-container {{
        background: linear-gradient(135deg, {COLORS['lavender_blush']} 0%, {COLORS['pastel_pink']} 100%);
        border: 2px dashed {COLORS['cherry_rose']};
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-bottom: 24px;
    }}
    
    .upload-title {{
        color: {COLORS['night_bordeaux']};
        font-size: 1.1rem;
        font-weight: 800;
        margin-bottom: 8px;
    }}
    
    .upload-subtitle {{
        color: {COLORS['text_light']};
        font-size: 0.9rem;
        margin-bottom: 16px;
    }}
    
    .quick-stats {{
        display: flex;
        gap: 16px;
        justify-content: center;
        margin-top: 16px;
        flex-wrap: wrap;
    }}
    
    .stat-badge {{
        background: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        color: {COLORS['cherry_rose']};
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }}
    
    /* Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS['night_bordeaux']} 0%, {COLORS['cherry_rose']} 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 14px 32px;
        font-size: 0.95rem;
        font-weight: 800;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 12px rgba(89, 13, 34, 0.15);
        width: 100%;
        letter-spacing: 0.5px;
    }}
    
    .stButton > button:hover {{
        box-shadow: 0 8px 20px rgba(89, 13, 34, 0.3);
        transform: translateY(-2px);
    }}
    
    /* Section titles */
    .section-title {{
        color: {COLORS['night_bordeaux']};
        font-size: 1.8rem;
        font-weight: 900;
        margin: 0px 0 5px 0;
        letter-spacing: 0.5px;
    }}
    
    /* Messages */
    .success-message {{
        background: linear-gradient(135deg, #27AE60 0%, #1E8449 100%);
        color: white;
        padding: 18px 24px;
        border-radius: 10px;
        font-weight: 700;
        margin-bottom: 24px;
        border-left: 5px solid #0B4C27;
    }}
    
    .warning-message {{
        background: linear-gradient(135deg, #F39C12 0%, #D68910 100%);
        color: white;
        padding: 18px 24px;
        border-radius: 10px;
        font-weight: 700;
        margin-bottom: 24px;
        border-left: 5px solid #7D4507;
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] button {{
        background: transparent;
        color: {COLORS['text_light']};
        border-bottom: 3px solid transparent;
        border-radius: 0;
        transition: all 0.3s ease;
        font-weight: 800;
        padding: 12px 20px;
    }}
    
    .stTabs [data-baseweb="tab-list"] button:hover {{
        color: {COLORS['cherry_rose']};
        border-bottom-color: {COLORS['bubblegum_pink']};
    }}
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
        color: {COLORS['night_bordeaux']};
        border-bottom-color: {COLORS['cherry_rose']};
    }}
    
    /* Expander */
    .streamlit-expanderHeader {{
        background: {COLORS['white']};
        color: {COLORS['night_bordeaux']};
        font-weight: 800;
        border-radius: 10px;
        border: 2px solid {COLORS['pastel_pink']};
    }}
    
    .streamlit-expanderHeader:hover {{
        background: {COLORS['lavender_blush']};
        border-color: {COLORS['cherry_rose']};
    }}
    
    /* Sidebar */
    .sidebar {{
        background: {COLORS['light_gray']} !important;
    }}
    
    /* Footer */
    .footer {{
        text-align: center;
        color: {COLORS['text_light']};
        font-size: 0.9rem;
        margin-top: 50px;
        padding: 30px;
        border-top: 2px solid {COLORS['pastel_pink']};
        font-weight: 600;
        background: {COLORS['light_gray']};
    }}
    
    .footer-title {{
        color: {COLORS['night_bordeaux']};
        font-weight: 900;
        font-size: 1.1rem;
        margin-bottom: 8px;
    }}
    
    .footer-subtitle {{
        color: {COLORS['cherry_rose']};
        font-weight: 700;
        margin-bottom: 4px;
    }}
    
    /* Selectbox and Input styling */
    .stSelectbox {{
        background: {COLORS['white']};
    }}
    
    .stFileUploader {{
        background: {COLORS['white']};
    }}
    
    /* Data frame */
    .dataframe {{
        border-radius: 8px;
        overflow: hidden;
        background: {COLORS['white']};
    }}
    
    </style>
""", unsafe_allow_html=True)

# Load Models
@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load("models/logistic_model.pkl"),
        "Decision Tree": joblib.load("models/decision_tree_model.pkl"),
        "KNN": joblib.load("models/knn_model.pkl"),
        "Naive Bayes": joblib.load("models/naive_bayes_model.pkl"),
        "Random Forest": joblib.load("models/random_forest_model.pkl"),
        "XGBoost": joblib.load("models/xgboost_model.pkl")
    }
    scaler = joblib.load("models/scaler.pkl")
    return models, scaler

models, scaler = load_models()

# Load wine image
try:
    wine_image = Image.open('wine_image.jpg')
except:
    wine_image = None

# Header
def show_header():
    if wine_image:
        col_img, col_text = st.columns([0.08, 0.92], gap="large")
        with col_img:
            st.image(wine_image, width=100)
        with col_text:
            st.markdown(f"""
            <div style='padding: 20px 0px 5px 0px;'>
                <h1 class='header-title'>Wine Quality Classification</h1>
                <p class='header-subtitle'>Premium Machine Learning Prediction System</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class='header'>
                <div style='width: 100px; height: 100px; border-radius: 16px; background: linear-gradient(135deg, {COLORS['rosewood']} 0%, {COLORS['bubblegum_pink']} 100%); display: flex; align-items: center; justify-content: center; font-size: 3rem; border: 4px solid white;'>🍷</div>
                <div class='header-content'>
                    <h1 class='header-title'>Wine Quality Classification</h1>
                    <p class='header-subtitle'>Premium Machine Learning Prediction System</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

# Top Navigation with Tabs
show_header()

tab1, tab2 = st.tabs(["🚀 Predict", "📚 Know More"])

# ==================== TAB 1: PREDICT ====================
with tab1:
    
    st.markdown(f"<div class='section-title'>Upload & Predict</div>", unsafe_allow_html=True)
    
    # Upload container - Compact design
    st.markdown(f"""
        <div class='upload-subtitle'>CSV format only • Include 'target' column for metrics</div>
    """, unsafe_allow_html=True)
    
    # Compact upload section
    col_upload, col_model = st.columns([2, 1], gap="large")
    
    with col_upload:
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            label_visibility="collapsed",
            key="file_uploader"
        )
    
    with col_model:
        st.markdown(f"<div style='margin-top: 4px;'></div>", unsafe_allow_html=True)
        model_name = st.selectbox(
            "Select Model",
            list(models.keys()),
            label_visibility="collapsed",
            key="model_select"
        )

# If file is uploaded, show file info and predict button in same row
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        has_target = "target" in data.columns
        
        # Custom styled button
        st.markdown(f"""
        <style>
        .predict-button-container {{
            display: flex;
            flex-direction: row;
            justify-content: flex-end;
        }}
        .stTooltipHoverTarget{{
            margin-top: -6rem;
            margin-left: 74rem;
            width: 22rem;        
        }}
        .predict-button-container button {{
            color: {COLORS['night_bordeaux']};
            font-weight: 900;
        }}
        </style>
        <div class="predict-button-container">
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Predict Now", use_container_width=True, key="predict_btn", help="Click to predict wine quality"):
                with st.spinner('🔄 Processing your data...'):
                    time.sleep(0.5)
                    
                    model = models[model_name]
                    X = data.copy()
                    
                    if "target" in X.columns:
                        X = X.drop("target", axis=1)

                    if model_name in ["Logistic Regression", "KNN", "Naive Bayes"]:
                        X_scaled = scaler.transform(X)
                    else:
                        X_scaled = X.values

                    y_pred = model.predict(X_scaled)
                    y_prob = model.predict_proba(X_scaled)[:, 1]

                    results_df = data.copy()
                    results_df['Prediction'] = y_pred
                    results_df['Probability'] = np.round(y_prob, 4)
                
                st.session_state.results_df = results_df
                st.session_state.y_pred = y_pred
                st.session_state.y_prob = y_prob
                st.session_state.has_target = has_target
                st.session_state.model_name = model_name
                st.session_state.data = data

        if not has_target:
            st.markdown(f"<div class='warning-message'>ℹ️ No 'target' column detected. You'll see predictions without evaluation metrics.</div>", unsafe_allow_html=True)
    
    # Show results if they exist
    if "results_df" in st.session_state:
        results_df = st.session_state.results_df
        y_pred = st.session_state.y_pred
        y_prob = st.session_state.y_prob
        has_target = st.session_state.has_target
        model_name = st.session_state.model_name
        data = st.session_state.data
        
        st.markdown(f"<div class='success-message'>✅ Prediction completed with <b>{model_name}</b>!</div>", unsafe_allow_html=True)
        
        # Results tabs
        res_tab1, res_tab2, res_tab3 = st.tabs(["📋 Results", "📊 Summary", "📥 Download"])
        
        with res_tab1:
            st.dataframe(results_df, use_container_width=True, height=350)
        
        with res_tab2:
            stat_col1, stat_col2, stat_col3 = st.columns(3, gap="large")
            
            bad_count = (results_df['Prediction'] == 0).sum()
            good_count = (results_df['Prediction'] == 1).sum()
            avg_prob = results_df['Probability'].mean()
            
            with stat_col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>🔴 Bad Wines</div>
                    <div class='metric-value'>{bad_count}</div>
                    <div class='metric-subtext'>{bad_count/len(results_df)*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with stat_col2:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>🟢 Good Wines</div>
                    <div class='metric-value'>{good_count}</div>
                    <div class='metric-subtext'>{good_count/len(results_df)*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with stat_col3:
                st.markdown(f"""
                <div class='metric-card-light'>
                    <div class='metric-label'>📈 Confidence</div>
                    <div class='metric-value' style='color: {COLORS["night_bordeaux"]};'>{avg_prob:.2%}</div>
                </div>
                """, unsafe_allow_html=True)
        
        with res_tab3:
            csv = results_df.to_csv(index=False)
            st.download_button(
                "📥 Download Predictions as CSV",
                csv,
                f"wine_predictions_{model_name.replace(' ', '_')}.csv",
                "text/csv",
                use_container_width=True
            )

        # Evaluation metrics
        if has_target:
            st.markdown(f"<h2 class='section-title' style='margin-top: 35px;'>Performance Metrics</h2>", unsafe_allow_html=True)
            
            y_true = data["target"]
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_true, y_pred)
            
            try:
                roc_auc = roc_auc_score(y_true, y_prob)
            except:
                roc_auc = None

            # Metrics in one row
            metric_col1, metric_col2, metric_col3, metric_col4, metric_col5, metric_col6 = st.columns(6, gap="medium")
            
            with metric_col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Accuracy</div>
                    <div class='metric-value'>{accuracy:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col2:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Precision</div>
                    <div class='metric-value'>{precision:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col3:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Recall</div>
                    <div class='metric-value'>{recall:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col4:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>F1-Score</div>
                    <div class='metric-value'>{f1:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col5:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>MCC</div>
                    <div class='metric-value'>{mcc:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col6:
                roc_val = f"{roc_auc:.3f}" if roc_auc else "N/A"
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>ROC-AUC</div>
                    <div class='metric-value'>{roc_val}</div>
                </div>
                """, unsafe_allow_html=True)

            # Confusion Matrix and Report
            st.markdown(f"<h2 class='section-title' style='margin-top: 35px;'>Detailed Analysis</h2>", unsafe_allow_html=True)
            
            cm_col, report_col = st.columns([0.5, 1.2], gap="large")
            
            cm = confusion_matrix(y_true, y_pred)
            
            with cm_col:
                fig, ax = plt.subplots(figsize=(1, 1))
                cmap = sns.color_palette([COLORS['cherry_blossom'], COLORS['bubblegum_pink'], COLORS['cherry_rose']], as_cmap=True)
                sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                            xticklabels=['Bad', 'Good'],
                            yticklabels=['Bad', 'Good'],
                            cbar=False, annot_kws={'size': 3, 'color': 'white'})
                ax.set_xlabel('Predicted', fontsize=3, color=COLORS['night_bordeaux'])
                ax.set_ylabel('Actual', fontsize=3, color=COLORS['night_bordeaux'])
                ax.set_title('Confusion Matrix', fontsize=3, color=COLORS['night_bordeaux'], pad=10)
                # Change font size for tick labels (Bad, Good)
                ax.tick_params(axis='x', labelsize=3)
                ax.tick_params(axis='y', labelsize=3)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
            
            with report_col:
                report = classification_report(y_true, y_pred, 
                                              target_names=['Bad', 'Good'],
                                              output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.background_gradient(cmap='RdYlGn', axis=0), 
                           use_container_width=True, height=450)

# ==================== TAB 2: KNOW MORE ====================
with tab2:
    
    st.markdown(f"<h2 class='section-title'>How to Use</h2>", unsafe_allow_html=True)
    
    guide_col1, guide_col2 = st.columns(2, gap="large")
    
    with guide_col1:
        st.markdown(f"""
        <div class='material-card'>
            <div class='card-icon'>📤</div>
            <div class='card-title'>Step 1: Prepare Data</div>
            <div class='card-description'>
            Create a CSV file with wine features. Include a 'target' column for evaluation (1=Good, 0=Bad).
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with guide_col2:
        st.markdown(f"""
        <div class='material-card'>
            <div class='card-icon'>📁</div>
            <div class='card-title'>Step 2: Upload File</div>
            <div class='card-description'>
            Go to the 'Predict' tab and upload your CSV file using the file uploader.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    guide_col3, guide_col4 = st.columns(2, gap="large")
    
    with guide_col3:
        st.markdown(f"""
        <div class='material-card'>
            <div class='card-icon'>🤖</div>
            <div class='card-title'>Step 3: Select Model</div>
            <div class='card-description'>
            Choose from 6 machine learning models. Each has different strengths and performance characteristics.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with guide_col4:
        st.markdown(f"""
        <div class='material-card'>
            <div class='card-icon'>📊</div>
            <div class='card-title'>Step 4: View Results</div>
            <div class='card-description'>
            Get predictions, detailed metrics, confusion matrix, and classification reports.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"<h2 class='section-title' style='margin-top: 40px;'>Available Models</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3, gap="large")
    
    with col1:
        st.markdown(f"""
        <div class='material-card'>
            <div class='card-icon'>🔵</div>
            <div class='card-title'>Logistic Regression</div>
            <div class='card-description'>Fast linear classification model. Best for quick predictions and baseline comparisons.</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='material-card'>
            <div class='card-icon'>🌳</div>
            <div class='card-title'>Decision Tree</div>
            <div class='card-description'>Interpretable tree-based model. Good for understanding decision patterns.</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='material-card'>
            <div class='card-icon'>👥</div>
            <div class='card-title'>K-Nearest Neighbors</div>
            <div class='card-description'>Instance-based learning. Works well with similar data points.</div>
        </div>
        """, unsafe_allow_html=True)
    
    col4, col5, col6 = st.columns(3, gap="large")
    
    with col4:
        st.markdown(f"""
        <div class='material-card'>
            <div class='card-icon'>🎲</div>
            <div class='card-title'>Naive Bayes</div>
            <div class='card-description'>Probabilistic model. Fast training and prediction.</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class='material-card'>
            <div class='card-icon'>🌲</div>
            <div class='card-title'>Random Forest</div>
            <div class='card-description'>Ensemble of trees. Robust and handles non-linear patterns.</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        st.markdown(f"""
        <div class='material-card'>
            <div class='card-icon'>⚡</div>
            <div class='card-title'>XGBoost</div>
            <div class='card-description'>Gradient boosting. Highest accuracy and complex pattern detection.</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"<h2 class='section-title' style='margin-top: 40px;'>About Wine Quality</h2>", unsafe_allow_html=True)
    
    about_col1, about_col2 = st.columns(2, gap="large")
    
    with about_col1:
        st.markdown(f"""
        <div class='material-card'>
            <div class='card-icon'>🍷</div>
            <div class='card-title'>What is Quality Score?</div>
            <div class='card-description'>
            Wine quality is rated on a scale of 0-10 based on sensory evaluation by experts. In this app, we classify wines as:
            • <b>Good (≥7):</b> High quality wines
            • <b>Bad (<7):</b> Lower quality wines
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with about_col2:
        st.markdown(f"""
        <div class='material-card'>
            <div class='card-icon'>📈</div>
            <div class='card-title'>Key Metrics Explained</div>
            <div class='card-description'>
            <b>Accuracy:</b> Overall correctness
            <b>Precision:</b> Of predicted goods, how many are correct
            <b>Recall:</b> Of actual goods, how many were found
            <b>F1-Score:</b> Balance between precision and recall
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"<h2 class='section-title' style='margin-top: 40px;'>Dataset Features</h2>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='material-card'>
        <div class='card-title'>Wine Physicochemical Properties</div>
        <div class='card-description'>
        <b>Acidity Measures:</b> Fixed acidity, volatile acidity, citric acid
        <br><br>
        <b>Chemical Properties:</b> Residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide
        <br><br>
        <b>Quality Indicators:</b> Density, pH, sulphates
        <br><br>
        <b>Alcohol Content:</b> Alcohol percentage by volume
        <br><br>
        These features are used by ML models to predict whether a wine is of good or bad quality.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown(f"""
        <div class='footer'>
            <div class='footer-title'>🍷 Wine Quality Classification System</div>
            <div class='footer-subtitle'>Developed by Anandhalakshmi</div>
            <p style='margin-top: 12px; color: {COLORS['text_light']};'>
            Advanced Machine Learning Solution for Wine Quality Prediction<br>
            Built with Streamlit, Scikit-learn & Python | © 2026
            </p>
        </div>
    """, unsafe_allow_html=True)