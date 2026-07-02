import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Page Configuration
st.set_page_config(
    page_title="Salary Prediction",
    page_icon="💰",
    layout="centered"
)

# Title
st.title("💰 Salary Prediction App")
st.write("Predict salary based on employee level using Polynomial Regression (degree = 4).")

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("dataset_SALARYPREDICTION.csv")
    return df

df = load_data()

# Prepare Data (same as notebook: x = Level, y = Salary)
x = df[["Level"]]
y = df[["Salary"]]

# Train Model (Polynomial Regression, degree = 4 — matches notebook)
poly_fet = PolynomialFeatures(degree=4)
x_poly = poly_fet.fit_transform(x)

model = LinearRegression()
model.fit(x_poly, y)

# Model performance on training data (same metrics as notebook)
y_pred_train = model.predict(x_poly)
mae = mean_absolute_error(y, y_pred_train)
mse = mean_squared_error(y, y_pred_train)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred_train)

# Sidebar
st.sidebar.header("Enter Employee Details")

level = st.sidebar.slider(
    "Employee Level",
    min_value=float(df["Level"].min()),
    max_value=float(df["Level"].max()),
    value=6.5,
    step=0.1
)

# Prediction (same approach as notebook's my_level example)
level_poly = poly_fet.transform([[level]])
prediction = model.predict(level_poly)[0][0]

# Result
st.subheader("Prediction Result")
st.success(f"Predicted Salary for Level {level}: ₹ {prediction:,.2f}")

# Model performance
with st.expander("View Model Performance (Degree 4, trained on full dataset)"):
    st.write(f"**MAE:** {mae:,.2f}")
    st.write(f"**MSE:** {mse:,.2f}")
    st.write(f"**RMSE:** {rmse:,.2f}")
    st.write(f"**R² Score:** {r2:.4f}")

# Dataset
with st.expander("View Dataset"):
    st.dataframe(df)

# Footer
st.markdown("---")
st.write("Built with ❤️ using Streamlit")
