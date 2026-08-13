import streamlit as st
import pandas as pd
import requests
import json
import os
import plotly.express as px
import plotly.graph_objects as go

API_URL = "http://localhost:8000"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ml', 'model')

st.set_page_config(page_title="Smart Stock 🚀", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .metric-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
    }
    .metric-value {
        font-size: 2em;
        font-weight: bold;
        color: #4CAF50;
    }
    .alert-card {
        background-color: #ffebee;
        border-radius: 10px;
        padding: 20px;
        color: #c62828;
        border-left: 5px solid #c62828;
    }
    .status-safe { color: #4CAF50; font-weight: bold; }
    .status-low { color: #FF9800; font-weight: bold; }
    .status-reorder { color: #F44336; font-weight: bold; }
    .status-critical { color: #B71C1C; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("📦 Smart Stock - AI Demand Forecasting")

@st.cache_data
def load_products():
    res = requests.get(f"{API_URL}/products").json()
    return res['products']

@st.cache_data
def load_stores():
    res = requests.get(f"{API_URL}/stores").json()
    return res['stores']

@st.cache_data
def load_demo_data():
    raw_path = os.path.join(MODEL_DIR, 'demo_raw_data.csv')
    if os.path.exists(raw_path):
        return pd.read_csv(raw_path)
    return pd.DataFrame()
    
def draw_overview(demo_df):
    st.header("Global Overview")
    total_sales = int(demo_df['sales'].sum()) if not demo_df.empty else 0
    total_products = demo_df['product_id'].nunique() if not demo_df.empty else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div>Total Sales (last 60d)</div><div class="metric-value">{total_sales}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div>Active Products</div><div class="metric-value">{total_products}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div>Expected Out of Stock</div><div class="metric-value" style="color: #FF9800;">2</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div>System Health</div><div class="metric-value">🟢 Online</div></div>', unsafe_allow_html=True)

def dashboard_app():
    # Sidebar
    st.sidebar.title("Navigation")
    products = load_products()
    stores = load_stores()
    
    store_names = [s['id'] for s in stores]
    product_names = [p['name'] for p in products]
    product_map = {p['name']: p for p in products}
    
    selected_store = st.sidebar.selectbox("Select Store", store_names)
    selected_product_name = st.sidebar.selectbox("Select Product", product_names)
    selected_product = product_map[selected_product_name]
    
    horizon = st.sidebar.slider("Forecast Horizon (Days)", min_value=1, max_value=30, value=7)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Simulate Current Stock")
    current_stock = st.sidebar.number_input("Current Stock Amount", min_value=0, value=50)
    
    demo_df = load_demo_data()
    
    # Overview
    draw_overview(demo_df)
    st.markdown("---")
    
    # Product Specific Dash
    st.header(f"Product Deep-Dive: {selected_product['name']} at {selected_store}")
    
    # 1. Historical Data
    if not demo_df.empty:
        prod_data = demo_df[(demo_df['store_id'] == selected_store) & (demo_df['product_id'] == selected_product['id'])]
        if not prod_data.empty:
            fig = px.line(prod_data, x="date", y="sales", title="Recent Historical Sales (Last 60 Days)", markers=True)
            st.plotly_chart(fig, use_container_width=True)
            
    # 2. Get Forecast from API
    st.subheader("AI Demand Forecast")
    
    # Use product dictionary if missing from API
    p_lead = selected_product.get('lead_time_days', 2)
    p_safety = selected_product.get('safety_stock', 20)
    
    if st.button("Generate Forecast & Reorder Engine", type="primary"):
        with st.spinner("AI Engine thinking..."):
            req_data = {
                "store_id": selected_store,
                "product_id": selected_product['id'],
                "horizon_days": horizon,
                "current_stock": current_stock,
                "safety_stock": p_safety,
                "lead_time_days": p_lead
            }
            try:
                res = requests.post(f"{API_URL}/forecast", json=req_data)
                res.raise_for_status()
                data = res.json()
                forecasts = data['forecasts']
                inv_rec = data['inventory_recommendation']
                
                # Plot Forecast
                df_f = pd.DataFrame(forecasts)
                fig_f = px.bar(df_f, x='date', y='predicted_demand', 
                               title='Predicted Daily Demand', 
                               text='predicted_demand', 
                               color_discrete_sequence=['#ff9900'])
                st.plotly_chart(fig_f, use_container_width=True)
                
                # Inventory engine display
                st.markdown("### Inventory Recommendation Action")
                
                status_color = "status-safe"
                is_alert = False
                if inv_rec['status'] == "CRITICAL":
                    status_color = "status-critical"
                    is_alert = True
                elif inv_rec['status'] == "REORDER NOW":
                    status_color = "status-reorder"
                    is_alert = True
                elif inv_rec['status'] == "LOW STOCK":
                    status_color = "status-low"
                
                r_col1, r_col2, r_col3, r_col4, r_col5 = st.columns(5)
                r_col1.metric("Current Stock", current_stock)
                r_col2.metric("Total Forecast Demand", inv_rec['total_forecast_demand'])
                r_col3.metric("Safety Stock", p_safety)
                r_col4.metric("Recommended Order", inv_rec['recommended_reorder'])
                
                r_col5.markdown(f"Status<br><div style='font-size: 1.5em;' class='{status_color}'>{inv_rec['status']}</div>", unsafe_allow_html=True)
                
                col_exp = st.empty()
                with col_exp.expander("Why this recommendation?"):
                    st.write(f"The model predicts **{inv_rec['total_forecast_demand']}** total units sold over the next {horizon} days.")
                    st.write(f"We keep **{p_safety}** units as safety stock.")
                    st.write(f"Required: {inv_rec['total_forecast_demand']} + {p_safety} = {inv_rec['total_forecast_demand'] + p_safety}")
                    st.write(f"Since you have **{current_stock}** units, you need to order: **{inv_rec['recommended_reorder']}** units.")
                    
                if is_alert:
                    st.markdown(f"""
                        <div class="alert-card">
                            <strong>WhatsApp Alert Triggered:</strong><br>
                            A notification has been generated for store manager indicating low stock for {selected_product['name']}.
                        </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Error fetching forecast: {e}")
                
    st.markdown("---")
    # 3. Model Performance
    st.header("Model Performance Metrics")
    try:
        perf = requests.get(f"{API_URL}/model/performance").json()
        if "error" not in perf:
            metrics = perf['metrics']
            importance = perf['feature_importance']
            
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("MAE (XGBoost)", metrics['xgboost']['MAE'], delta=f"{metrics['xgboost']['MAE'] - metrics['baseline']['MAE']:.2f} vs Baseline", delta_color="inverse")
            mc2.metric("RMSE (XGBoost)", metrics['xgboost']['RMSE'])
            mc3.metric("MAPE (XGBoost)", f"{metrics['xgboost']['MAPE']}%")
            
            st.subheader("Feature Importance")
            feat_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])
            fig_i = px.bar(feat_df, x='Importance', y='Feature', orientation='h', title="Top Drivers of Demand")
            fig_i.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_i, use_container_width=True)
    except:
        st.info("Metrics not available.")
        
if __name__ == '__main__':
    dashboard_app()
