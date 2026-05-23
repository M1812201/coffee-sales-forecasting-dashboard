import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Coffee Demand Intelligence", layout="wide")
st.title("Coffee Sales and Demand Prediction Dashboard")

# -------------------------------
# LOAD DATA
# -------------------------------
# Using your local path
df = pd.read_excel(r'D:\Project1\Afficionado Coffee Roasters.xlsx')

# -------------------------------
# DATA CLEANING
# -------------------------------
df["transaction_time"] = pd.to_datetime(
    df["transaction_time"].astype(str).str.strip(),
    format='%H:%M:%S',
    errors='coerce'
)

df["hour"] = df["transaction_time"].dt.hour
df["dayofweek"] = pd.to_datetime(df["year"].astype(str) + "-01-01").dt.dayofweek
df["revenue"] = df["transaction_qty"] * df["unit_price"]

st.success("Data Loaded Successfully")

# -------------------------------
# SIDEBAR CONTROLS
# -------------------------------
st.sidebar.header("Controls")

forecast_horizon = st.sidebar.slider(
    "Forecast Horizon (Days Simulation)",
    1, 30, 7
)

metric = st.sidebar.radio(
    "Select Metric",
    ["Quantity", "Revenue"]
)

target_col = "revenue" if metric == "Revenue" else "transaction_qty"

store_list = df["store_location"].unique()
selected_store = st.sidebar.selectbox("Select Store", store_list)

# -------------------------------
# TABS (PRODUCTION UI)
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard",
    "📈 Forecast",
    "📉 Model Evaluation",
    "⚙ KPIs"
])

# =========================================================
# TAB 1 - POWER BI STYLE DASHBOARD (FILTERED BY STORE)
# =========================================================
with tab1:
    # Filter the dataset based on the sidebar store selector
    df_filtered = df[df["store_location"] == selected_store]

    # Custom styling header matching your red Power BI title banner
    st.markdown(
        f"<h3 style='text-align: center; color: #1E3A8A;'>Historical Performance Analysis — {selected_store}</h3>", 
        unsafe_allow_html=True
    )
    st.markdown("---")

    # Recreating your 3 Power BI KPI Cards side-by-side
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_rev = df_filtered["revenue"].sum()
        st.metric(label="Total Revenue", value=f"${total_rev:,.2f}")
        
    with col2:
        total_items = df_filtered["transaction_qty"].sum()
        st.metric(label="Total Items Sold", value=f"{total_items:,}")
        
    with col3:
        # Custom DAX: Revenue divided by Total Items Sold
        avg_price = total_rev / total_items if total_items > 0 else 0
        st.metric(label="Avg Price Per Item", value=f"${avg_price:.2f}")

    st.markdown("---")

    # Charts Layout - Split into Two Columns
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Revenue by Category")
        # Horizontal bar chart matching your Power BI product categories
        category_data = df_filtered.groupby("product_category")["revenue"].sum().sort_values(ascending=True)
        st.bar_chart(category_data, horizontal=True)

    with chart_col2:
        st.subheader("Hourly Sales Trend")
        # Line chart tracking customer volume peaks by hour
        hourly_data = df_filtered.groupby("hour")[target_col].sum()
        st.line_chart(hourly_data)

    st.markdown("---")
    st.subheader("Store Density Heatmap (Hour vs Location)")
    # Keeps your advanced heat grid representation intact across all data
    heatmap_data = df.groupby(["hour", "store_location"])[target_col].sum().unstack()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    sns.heatmap(heatmap_data, cmap="YlGnBu", ax=ax)
    st.pyplot(fig)

# =========================================================
# FEATURE ENGINEERING (ML PREPARATION)
# =========================================================
df_model = df.copy()
df_model = df_model.groupby(["hour", "store_location"])[target_col].sum().reset_index()

df_model["lag_1"] = df_model[target_col].shift(1)
df_model["lag_24"] = df_model[target_col].shift(24)
df_model["lag_168"] = df_model[target_col].shift(168)
df_model["rolling_3"] = df_model[target_col].rolling(3).mean()

df_model = pd.get_dummies(df_model, columns=["store_location"])
df_model = df_model.fillna(0)

X = df_model.drop(target_col, axis=1)
y = df_model[target_col]

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# =========================================================
# MACHINE LEARNING MODELS
# =========================================================
lr = LinearRegression()
lr.fit(X_train, y_train)

gb = GradientBoostingRegressor()
gb.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
gb_pred = gb.predict(X_test)

# =========================================================
# CONFIDENCE INTERVAL CALCULATIONS
# =========================================================
residuals = y_test - lr_pred
std = np.std(residuals)
z = 1.96

upper = lr_pred + z * std
lower = lr_pred - z * std

# =========================================================
# TAB 2 - FORECAST
# =========================================================
with tab2:
    st.subheader("Forecast with Confidence Interval")

    store_col = "store_location_" + selected_store

    if store_col in X_test.columns:
        plot_df = pd.DataFrame({
            "Actual": y_test.values,
            "Predicted": lr_pred,
            "Upper": upper,
            "Lower": lower,
            "Store": X_test[store_col].values
        }).reset_index(drop=True)

        # FILTER SELECTED STORE
        plot_df = plot_df[plot_df["Store"] == 1]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(plot_df["Actual"].values, label="Actual", color="blue")
        ax.plot(plot_df["Predicted"].values, label="Predicted", color="green")
        ax.fill_between(
            range(len(plot_df)),
            plot_df["Lower"],
            plot_df["Upper"],
            alpha=0.3,
            color="gray",
            label="Confidence Interval"
        )
        ax.legend()
        st.pyplot(fig)

# =========================================================
# TAB 3 - MODEL EVALUATION
# =========================================================
with tab3:
    st.subheader("Model Performance Evaluation")

    def show(name, y_true, y_pred):
        st.write(f"### {name}")
        st.write("MAE:", round(mean_absolute_error(y_true, y_pred), 2))
        st.write("RMSE:", round(np.sqrt(mean_squared_error(y_true, y_pred)), 2))
        st.write("---")

    show("Linear Regression", y_test, lr_pred)
    show("Gradient Boosting (Ensemble)", y_test, gb_pred)

# =========================================================
# TAB 4 - KPI DASHBOARD
# =========================================================
with tab4:
    st.subheader("Key Performance Indicators")

    st.metric("Forecast MAE", round(mean_absolute_error(y_test, lr_pred), 2))
    st.metric("Forecast RMSE", round(np.sqrt(mean_squared_error(y_test, lr_pred)), 2))

    peak_hour = df.groupby("hour")[target_col].sum().idxmax()
    st.metric("Peak Demand Hour", f"{peak_hour}:00 AM/PM")

    best_store = df.groupby("store_location")[target_col].sum().idxmax()
    st.metric("Best Performing Store", best_store)

# =========================================================
# LIVE INTERACTIVE PREDICTION WIDGET
# =========================================================
st.markdown("---")
st.subheader("Live Model Prediction Playground")

input_hour = st.slider("Hour of Day Simulation", 0, 23, 10)

input_data = {
    "hour": input_hour,
    "lag_1": 0,
    "lag_24": 0,
    "lag_168": 0,
    "rolling_3": 0
}

for col in X.columns:
    if col.startswith("store_location_"):
        input_data[col] = 1 if col == "store_location_" + selected_store else 0

input_df = pd.DataFrame([input_data])
input_df = input_df.reindex(columns=X.columns, fill_value=0)

pred = lr.predict(input_df)[0]
# Prevent negative forecasting simulations
pred_final = max(0, pred)

st.success(f"Predicted {metric} Demand for {selected_store} at {input_hour}:00 is: {round(pred_final, 2)}")
