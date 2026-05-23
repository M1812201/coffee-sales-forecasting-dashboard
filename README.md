# ☕ Coffee Sales Forecasting & Demand Intelligence Dashboard

## 📌 Project Overview
This project is an end-to-end, production-level coffee sales forecasting and demand intelligence system developed using Python and Streamlit. By combining interactive, Power BI-style historical data analytics with robust Machine Learning regression models, this system transforms raw transaction data into actionable business strategies.

The dashboard analyzes historical retail footprints to:
* **Forecast future sales & demand trends** with statistical confidence windows.
* **Isolate operational vulnerabilities** by tracking hourly peak-demand metrics.
* **Optimize supply chains and staffing matrices** via granular, store-level analytics.

---

# 🚀 Features

### 📊 Power BI-Style Interactive Analytics
* **Dynamic KPI Ribbon:** Real-time tracking of *Total Revenue*, *Total Items Sold*, and custom calculations like *Avg Price Per Item*.
* **Global Metric Toggle:** A master selector switch to dynamically shift the entire app view between **Volume (Quantity)** and **Value (Revenue)**.
* **Granular Store Slicing:** Multi-location cross-filtering (*Astoria*, *Hell's Kitchen*, *Lower Manhattan*) that instantly recalculates metrics and trend charts.
* **Operational Density Grids:** High-fidelity Seaborn heatmaps tracking transactional volume across operating hours versus locations.

### 🤖 Predictive Machine Learning Engine
* **Advanced Feature Pipelines:** Automated creation of temporal features, store dummy flags, and historical data offsets (**1-hour lag, 24-hour lag, 168-hour weekly lag, and 3-hour rolling averages**).
* **Dual-Model Processing:** Implements both a baseline **Linear Regression** model and an ensemble **Gradient Boosting Regressor**.
* **Confidence Interval Bounds:** Forecasts map out a 95% statistical safety net ($\pm 1.96 \times \text{std}$) to help store managers avoid stockouts or over-handling waste.
* **Live Prediction Simulator:** An interactive playground allowing operators to plug in a simulated hour and instantly view model expectations.

---

# 🛠️ Technologies Used
* **UI/UX Framework:** Streamlit
* **Data Pipelines & Processing:** Pandas, NumPy, OpenPyXL
* **Statistical Visualizations:** Matplotlib, Seaborn
* **Machine Learning Algorithms:** Scikit-Learn (`LinearRegression`, `GradientBoostingRegressor`)

---

# 📂 Dataset 
The system processes structured retail transaction records containing features critical to demand planning:
* `transaction_time` / `hour` (Temporal anchors)
* `transaction_qty` (Volume marker)
* `unit_price` / `revenue` (Financial performance variables)
* `store_location` (Spatial categorization)
* `product_category` / `product_type` (Inventory hierarchy items)

---

# 📉 Evaluation Metrics
Models undergo split validation partitions (80/20 Train/Test) and track performance across standard regression error profiles:

| Machine Learning Model | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) |
| :--- | :--- | :--- |
| **Linear Regression** | *~1097.81* | *~1282.31* |
| **Gradient Boosting** | *~1365.49* | *~1834.99* |

*Note: Linear Regression acts as the structural engine for our live scenario simulator due to its tight error margins on this baseline matrix.*

---

# ▶️ How to Run Locally

### 1. Install Requirements
Ensure you have Python installed, then pull down dependencies:
```bash
pip install pandas streamlit numpy seaborn matplotlib scikit-learn openpyxl

## Configure File Path
df = pd.read_excel(r'Local Drive:\Your_Folder_Name\Afficionado Coffee Roasters.xlsx')


### 2. Run Streamlit App

```bash
streamlit run app.py
```

### 3. 🔗 Live App
https://coffee-sales-forecasting-dashboard.streamlit.app/


# 📈 Future Improvements

- ARIMA / SARIMA forecasting
- Facebook Prophet integration
- Real-time forecasting
- Cloud deployment
- Automated model retraining



# 👩‍💻 Author

Akshata Gurav Data Science & Business Intelligence Developer Project Portfolio Work: Retail Demand Forecasting using Machine Learning and Streamlit.
