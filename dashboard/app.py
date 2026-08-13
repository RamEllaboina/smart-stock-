import streamlit as st
import pandas as pd
import requests
import json
import os
import plotly.express as px
import plotly.graph_objects as go

API_URL = "http://localhost:8001"
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

def refactor_dashboard(demo_df, products, stores, store_names, product_names, product_map):
    st.sidebar.title("Navigation")
    
    selected_store = st.sidebar.selectbox("Select Store", store_names)
    selected_product_name = st.sidebar.selectbox("Select Product", product_names)
    selected_product = product_map[selected_product_name]
    
    horizon = st.sidebar.slider("Forecast Horizon (Days)", min_value=1, max_value=30, value=7)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Simulate Current Stock")
    current_stock = st.sidebar.number_input("Current Stock Amount", min_value=0, value=50)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Global Overview", "📦 Product Deep-Dive", "⚠️ Data Quality & Anomalies", "⚙️ MLOps & Model Health"])
    
    with tab1:
        draw_overview(demo_df)
        
    with tab4:
        st.header(f"MLOps Health: {selected_product['name']} at {selected_store}")
        col_btn, _ = st.columns([1, 4])
        with col_btn:
            if st.button("Trigger Retraining Pipeline", type="secondary"):
                try:
                    res_train = requests.post(f"{API_URL}/monitoring/retrain", json={"store_id": selected_store, "product_id": selected_product['id']})
                    st.info(str(res_train.json()))
                except Exception as e:
                    st.error(str(e))
                    
        try:
            res_ops = requests.get(f"{API_URL}/monitoring/health?store_id={selected_store}&product_id={selected_product['id']}")
            if res_ops.status_code == 200:
                ops_report = res_ops.json()
                
                ops_c1, ops_c2, ops_c3 = st.columns(3)
                ops_c1.markdown(f'<div class="metric-card"><div>System Status</div><div class="metric-value">{ops_report["model_status"]}</div></div>', unsafe_allow_html=True)
                ops_c2.markdown(f'<div class="metric-card"><div>Active Model</div><div class="metric-value" style="font-size: 1.2em;">{ops_report["production_model"]}</div></div>', unsafe_allow_html=True)
                ops_c3.markdown(f'<div class="metric-card"><div>Retraining Suggestion</div><div class="metric-value" style="color: #FF9800;">{ops_report["retraining_decision"]}</div></div>', unsafe_allow_html=True)
                
                st.markdown("### Performance Degradation")
                perf = ops_report["performance"]
                p_c1, p_c2, p_c3, p_c4 = st.columns(4)
                p_c1.metric("WAPE", f"{perf['wape']*100:.2f}%", delta=f"{perf['degradation']*100:.2f}% vs Baseline", delta_color="inverse")
                p_c2.metric("MAE", perf['mae'])
                p_c3.metric("Bias", f"{perf['bias']:.2f}")
                p_c4.metric("Status", perf['status'])
                
                st.markdown("### Feature Drift")
                drift = ops_report["drift"]
                d_c1, d_c2, d_c3, d_c4 = st.columns(4)
                d_c1.metric("Total Monitored", drift['total_features'])
                d_c2.metric("Features Drifted", drift['features_with_drift'])
                d_c3.metric("Percentage", f"{drift['percentage_drifted']}%")
                d_c4.metric("Highest Drift", drift['highest_drift_feature'])
                
                if drift['feature_details']:
                    st.dataframe(pd.DataFrame(drift['feature_details']), use_container_width=True)
            else:
                st.info("No MLOps report found for this item.")
        except Exception as e:
            st.error(f"Cannot connect to MLOps API: {e}")
            
    with tab3:
        st.header("Production Data Quality & Anomaly Monitor")
        try:
            res_anom = requests.get(f"{API_URL}/monitoring/anomalies")
            if res_anom.status_code == 200:
                report = res_anom.json()
                
                qa_col1, qa_col2, qa_col3, qa_col4 = st.columns(4)
                
                status_clr = "#4CAF50" if report['data_quality_score'] >= 90 else "#FF9800"
                if report['data_quality_score'] < 70: status_clr = "#F44336"
                
                qa_col1.markdown(f'<div class="metric-card"><div>Data Quality Score</div><div class="metric-value" style="color: {status_clr}">{report["data_quality_score"]}/100</div></div>', unsafe_allow_html=True)
                qa_col2.markdown(f'<div class="metric-card"><div>Total Anomalies</div><div class="metric-value" style="color: #FF9800;">{report["anomalies"]}</div></div>', unsafe_allow_html=True)
                qa_col3.markdown(f'<div class="metric-card"><div>Critical Anomalies</div><div class="metric-value" style="color: #F44336;">{report["critical_anomalies"]}</div></div>', unsafe_allow_html=True)
                qa_col4.markdown(f'<div class="metric-card"><div>Records Scanned</div><div class="metric-value">{report["total_records"]}</div></div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Failed to load anomalies: {e}")
                
def login_screen():
    st.subheader("Login to Smart Stock Backend")
    with st.form("login"):
        user = st.text_input("Email")
        pwd = st.text_input("Password", type="password")
        sub = st.form_submit_button("Login")
        if sub:
            try:
                res = requests.post(f"{API_URL}/auth/login", data={"username": user, "password": pwd})
                if res.status_code == 200:
                    st.session_state['token'] = res.json()['access_token']
                    st.session_state['user'] = res.json()['user']
                    st.rerun()
                else:
                    st.error("Invalid credentials")
            except Exception as e:
                st.error(f"Backend cannot be reached: {e}")

def dashboard_app():
    if 'token' not in st.session_state:
        login_screen()
        return
        
    token = st.session_state['token']
    headers = {"Authorization": f"Bearer {token}"}
    
    st.sidebar.title(f"Logged in as {st.session_state['user']['email']}")
    
    # Load dynamic values from the API
    try:
        products = requests.get(f"{API_URL}/products").json()['products']
        stores = requests.get(f"{API_URL}/stores").json()['stores']
        
        # Streamlit requires a "name" key in products for the map
        for p in products:
            if 'name' not in p:
                p['name'] = p['id']
            if 'safety_stock' not in p:
                p['safety_stock'] = 20
    except Exception:
        products = [{"id": "P001", "name": "Milk", "safety_stock": 20}, {"id": "P002", "name": "Bread", "safety_stock": 10}]
        stores = [{"id": "store_01"}, {"id": "store_02"}]
    
    store_names = [s['id'] for s in stores]
    product_names = [p['name'] for p in products]
    product_map = {p['name']: p for p in products}
    
    demo_df = load_demo_data()
    
    selected_store = st.sidebar.selectbox("Select Store", store_names)
    selected_product_name = st.sidebar.selectbox("Select Product", product_names)
    selected_product = product_map[selected_product_name]
    
    horizon = st.sidebar.slider("Forecast Horizon (Days)", min_value=1, max_value=30, value=7)
    
    st.sidebar.markdown("---")
    current_stock = st.sidebar.number_input("Current Stock Amount", min_value=0, value=50)
    
    if st.sidebar.button("Logout"):
        del st.session_state['token']
        del st.session_state['user']
        st.rerun()
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Global Overview", "📦 Product Deep-Dive", "⚠️ Data Quality & Anomalies", "⚙️ MLOps & Model Health"])
    
    with tab1:
        draw_overview(demo_df)
        
    with tab4:
        st.header(f"MLOps Health: {selected_product['name']} at {selected_store}")
        col_btn, _ = st.columns([1, 4])
        with col_btn:
            if st.button("Trigger Retraining Pipeline", type="secondary"):
                try:
                    res_train = requests.post(f"{API_URL}/monitoring/retrain", json={"store_id": selected_store, "product_id": selected_product['id']}, headers=headers)
                    if res_train.status_code == 200:
                        st.info(str(res_train.json()))
                    else:
                        st.error(res_train.text)
                except Exception as e:
                    st.error(str(e))
                    
        try:
            res_ops = requests.get(f"{API_URL}/monitoring/health?store_id={selected_store}&product_id={selected_product['id']}", headers=headers)
            if res_ops.status_code == 200:
                ops_report = res_ops.json()
                
                ops_c1, ops_c2, ops_c3 = st.columns(3)
                ops_c1.markdown(f'<div class="metric-card"><div>System Status</div><div class="metric-value">{ops_report["model_status"]}</div></div>', unsafe_allow_html=True)
                ops_c2.markdown(f'<div class="metric-card"><div>Active Model</div><div class="metric-value" style="font-size: 1.2em;">{ops_report["production_model"]}</div></div>', unsafe_allow_html=True)
                ops_c3.markdown(f'<div class="metric-card"><div>Retraining Suggestion</div><div class="metric-value" style="color: #FF9800;">{ops_report["retraining_decision"]}</div></div>', unsafe_allow_html=True)
                
                perf = ops_report["performance"]
                st.markdown(f"**WAPE:** {perf['wape']*100:.2f}% | **Degradation:** {perf['degradation']*100:.2f}% | **Drift Features:** {ops_report['drift']['features_with_drift']}")
            else:
                st.info(res_ops.text)
        except Exception as e:
            st.error(f"Cannot connect to API: {e}")
            
    with tab3:
        st.header("Production Data Quality & Anomaly Monitor")
        try:
            res_anom = requests.get(f"{API_URL}/monitoring/anomalies", headers=headers)
            if res_anom.status_code == 200:
                report = res_anom.json()
                st.write(f"Total Anomalies Filtered: {report['total']}")
                if report['items']:
                    df_an = pd.DataFrame(report['items'])
                    st.dataframe(df_an[['date', 'product_id', 'original_value', 'anomaly_type', 'severity', 'reason']], use_container_width=True)
            else:
                st.error(res_anom.text)
        except Exception as e:
            st.error(f"Error fetching anomalies: {e}")

    with tab2:
        st.header(f"{selected_product['name']} at {selected_store}")
        
        if not demo_df.empty:
            prod_data = demo_df[(demo_df['store_id'] == selected_store) & (demo_df['product_id'] == selected_product['id'])]
            if not prod_data.empty:
                fig = px.line(prod_data, x="date", y="sales", title="Recent Historical Sales (Last 60 Days)", markers=True)
                st.plotly_chart(fig, use_container_width=True)
                
        if st.button("Generate Forecast & Reorder Engine", type="primary"):
            req_data = {
                "store_id": selected_store,
                "product_id": selected_product['id'],
                "horizon_days": horizon,
                "current_stock": current_stock,
                "safety_stock": selected_product.get('safety_stock', 20),
                "lead_time_days": 2
            }
            res = requests.post(f"{API_URL}/inventory/recommendations", json=req_data, headers=headers)
            if res.status_code == 200:
                data = res.json()
                forecasts = data['forecasts']
                inv_rec = data['inventory_recommendation']
                
                df_f = pd.DataFrame(forecasts)
                fig_f = px.bar(df_f, x='date', y='predicted_demand', title='Predicted Demand')
                st.plotly_chart(fig_f, use_container_width=True)
                
                st.write("### Target Recommendation", inv_rec)
            else:
                st.error(res.text)

if __name__ == '__main__':
    dashboard_app()
