import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------------
# PAGE SETUP
# -------------------------------
st.set_page_config(page_title="Coffee Sales Prediction", layout="wide")
st.title("Coffee Sales Prediction Dashboard")

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_excel(r'Afficionado Coffee Roasters.xlsx')

# -------------------------------
# CLEANING
# -------------------------------
df["transaction_time"] = pd.to_datetime(
    df["transaction_time"].astype(str).str.strip(),
    format='%H:%M:%S',
    errors='coerce'
)

df["hour"] = df["transaction_time"].dt.hour
df["revenue"] = df["transaction_qty"] * df["unit_price"]

st.success("Data Loaded Successfully")

# -------------------------------
# SIDEBAR CONTROLS
# -------------------------------
st.sidebar.header("Controls")

forecast_horizon = st.sidebar.slider(
    "Forecast Horizon (Hours)",
    1, 72, 24
)

metric = st.sidebar.radio(
    "Select Metric",
    ["Quantity", "Revenue"]
)

if metric == "Revenue":
    target_col = "revenue"
else:
    target_col = "transaction_qty"

# -------------------------------
# BASIC METRICS
# -------------------------------
st.subheader("Overall Metrics")
st.write("Total Quantity:", df["transaction_qty"].sum())
st.write("Total Revenue:", df["revenue"].sum())

# -------------------------------
# ANALYSIS
# -------------------------------
st.subheader("Peak Hour Analysis")

hourly_sales = df.groupby("hour")[target_col].sum()
st.line_chart(hourly_sales)

st.subheader("Store-wise Sales")
store_sales = df.groupby("store_location")[target_col].sum()
st.bar_chart(store_sales)

st.subheader("Category Analysis")
category_sales = df.groupby("product_category")[target_col].sum()
st.bar_chart(category_sales)

# -------------------------------
# BUSINESS INSIGHTS
# -------------------------------
st.subheader("Business Insights")

peak_store = df.groupby(["store_location", "hour"])[target_col].sum().reset_index()

peak_store_final = peak_store.loc[
    peak_store.groupby("store_location")[target_col].idxmax()
].sort_values(by=target_col, ascending=False)

peak_store_final.columns = ["Store", "Peak Hour", "Sales"]

st.dataframe(peak_store_final)

best_store = df.groupby("store_location")[target_col].sum().idxmax()
low_store = df.groupby("store_location")[target_col].sum().idxmin()

st.write("Best Store:", best_store)
st.write("Lowest Store:", low_store)

# -------------------------------
# MACHINE LEARNING
# -------------------------------
st.subheader("Machine Learning Model")

hourly_data = df.groupby(["hour", "store_location"])[target_col].sum().reset_index()

hourly_data["lag_1"] = hourly_data[target_col].shift(1)
hourly_data["rolling_3"] = hourly_data[target_col].rolling(3).mean()
hourly_data = hourly_data.fillna(0)

hourly_data = pd.get_dummies(hourly_data, columns=["store_location"])

X = hourly_data.drop(target_col, axis=1)
y = hourly_data[target_col]

split_index = int(len(X) * 0.8)

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# -------------------------------
# MODELS
# -------------------------------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

gb_model = GradientBoostingRegressor()
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

# -------------------------------
# BASELINES
# -------------------------------
naive_pred = y_test.shift(1).bfill()
moving_avg_pred = y_test.rolling(3).mean().bfill()

# -------------------------------
# CONFIDENCE INTERVAL (ADDED)
# -------------------------------
residuals = y_test - lr_pred
std_error = np.std(residuals)
z = 1.96  # 95% confidence

upper_bound = lr_pred + z * std_error
lower_bound = lr_pred - z * std_error

# -------------------------------
# EVALUATION
# -------------------------------
st.subheader("Model Evaluation")

def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    st.write(f"{name} MAE: {round(mae,2)}")
    st.write(f"{name} RMSE: {round(rmse,2)}")
    st.write("---")

evaluate("Linear Regression", y_test, lr_pred)
evaluate("Gradient Boosting", y_test, gb_pred)
evaluate("Naive", y_test, naive_pred)
evaluate("Moving Average", y_test, moving_avg_pred)

# -------------------------------
# FORECAST + CONFIDENCE INTERVAL
# -------------------------------
st.subheader("Forecast Visualization (with Confidence Interval)")

store_list = df["store_location"].unique()
selected_store = st.selectbox("Select Store", store_list)

store_col = "store_location_" + selected_store

if store_col in X_test.columns:

    # -------------------------------
    # FIX: reset index to avoid mismatch error
    # -------------------------------
    test_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": lr_pred,
        "Upper": upper_bound,
        "Lower": lower_bound,
        store_col: X_test[store_col].values
    }).reset_index(drop=True)

    # safe filtering
    store_data = test_df[test_df[store_col] == 1]

    if len(store_data) > 0:

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(store_data["Actual"].values, label="Actual", color="blue")
        ax.plot(store_data["Predicted"].values, label="Predicted", color="green")

        ax.fill_between(
            range(len(store_data)),
            store_data["Lower"],
            store_data["Upper"],
            color="gray",
            alpha=0.3,
            label="Confidence Interval"
        )

        ax.set_title("Forecast with Confidence Interval")
        ax.legend()

        st.pyplot(fig)

    else:
        st.warning("No data available for selected store")
# -------------------------------
# HEATMAP
# -------------------------------
st.subheader("Hourly Demand Heatmap")

heatmap_data = df.groupby(["hour", "store_location"])[target_col].sum().unstack()

fig, ax = plt.subplots(figsize=(8, 4))

sns.heatmap(
    heatmap_data,
    ax=ax,
    cmap="YlGnBu",
    annot=False,
    linewidths=0.3
)

st.pyplot(fig)

# -------------------------------
# PREDICTION UI
# -------------------------------
st.subheader("Predict Sales")

input_hour = st.slider("Select Hour", 0, 23, 10)
input_store = st.selectbox("Select Store Location (Prediction)", store_list)

input_data = {
    "hour": input_hour,
    "lag_1": 0,
    "rolling_3": 0
}

for col in X.columns:
    if col.startswith("store_location_"):
        input_data[col] = 1 if col == "store_location_" + input_store else 0

input_df = pd.DataFrame([input_data])
input_df = input_df.reindex(columns=X.columns, fill_value=0)

predicted_sales = lr_model.predict(input_df)[0]

st.success(f"Estimated {metric}: {round(predicted_sales,2)}")
