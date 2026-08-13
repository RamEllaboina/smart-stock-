# Smart Stock

**Smart Stock** is an AI-powered demand forecasting and inventory management MVP built for small retailers and e-grocery businesses. It leverages historical sales data to intelligently predict future product demand, preventing both overstocking and stockouts. 

It is designed for a fast 24-hour hackathon implementation providing end-to-end functionality right from Data Generation & ML to an interactive AI Dashboard and WhatsApp notifications.

## 🚀 Features
- **Exploratory Data Generation**: Generates synthetic, realistic retail sales data featuring trends, seasonality, and random promotions.
- **Data Engineering**: Data cleaning, handling missing dates, detecting outliers.
- **Feature Engineering**: Built-in rolling aggregations and lag features for chronological modeling.
- **Machine Learning**: Predicts future demand using an XGBoost Regressor.
- **Decision Engine**: Recommends automated restocking quantities factoring in safety stock and lead times.
- **Dashboard**: Simple, interactive dashboard (Streamlit) showing current health, specific product forecasts, and performance metrics.
- **Automated Alerts**: Real-time WhatsApp notifications (via Twilio) for critically low inventory.

## 📁 Repository Structure
```
smart-stock/
├── data/
│   ├── raw/
│   └── generate_data.py   <- Run this first to build the dataset
├── ml/
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py           <- Trains XGBoost and Baseline
│   ├── evaluate.py
│   ├── predict.py
│   └── model/             <- Output metrics and trained models 
├── backend/
│   └── main.py            <- FastAPI backend
├── dashboard/
│   └── app.py             <- Streamlit Dashboard UI
├── alerts/
│   └── whatsapp.py        <- Twilio alerts implementation
├── .env.example
├── requirements.txt
└── README.md
```

## 🛠 Tech Stack
- **Backend API**: Python 3, FastAPI, Uvicorn
- **Machine Learning**: XGBoost, Scikit-Learn, Pandas, NumPy
- **Frontend UI**: Streamlit, Plotly
- **Alerts**: Twilio API

## 🚦 Quick Start & Demo Instructions

1. **Install Dependencies**
   Navigate to the project root and install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Data**
   ```bash
   python data/generate_data.py
   ```

3. **Train the AI Model**
   ```bash
   python -m ml.train
   ```

4. **Start the API Backend**
   ```bash
   uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```
   
5. **Start the Interactive Dashboard**
   Open a new terminal window:
   ```bash
   streamlit run dashboard/app.py
   ```

6. **View the Magic**
   Streamlit will open the dashboard at `http://localhost:8501`. 
   Use the sidebar to simulate different products, current stock, and forecast horizons!

## 🔐 Environment Variables
Optional: Add Twilio keys for live WhatsApp alerts instead of mock ones.
Create a `.env` file in the root folder using `.env.example` as a template and provide your actual Twilio SID and Auth Token.

## ⚠️ Limitations & Future Scope
- The forecasting pipeline uses a simplistic rolling approach for future iterative predictions in this MVP.
- Can be extended to support real-time POS data streaming.
- Hyperparameter tuning for XGBoost can be integrated.
- Expanding into deep learning based sequence models like LSTMs or Transformers.
