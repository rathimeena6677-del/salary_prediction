import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Salary Prediction",
    page_icon="💰",
    layout="centered"
)

st.title("💰 Salary Prediction App")
st.write(
    "Predict salary based on employee level using **Polynomial Regression**. "
    "Adjust the degree and level in the sidebar to see how the model responds."
)

# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("dataset_SALARYPREDICTION.csv")

df = load_data()

x = df[["Level"]]
y = df[["Salary"]]

# ---------------------------------------------------------
# Sidebar - user input
# ---------------------------------------------------------
st.sidebar.header("Enter Employee Details")

degree = st.sidebar.selectbox(
    "Polynomial Degree",
    options=[1, 2, 3, 4, 5, 6],
    index=3,  # defaults to 4, matching the notebook
    help="Degree 4 matches the original notebook. Try other degrees to compare fit."
)

level = st.sidebar.slider(
    "Employee Level",
    min_value=float(df["Level"].min()),
    max_value=float(df["Level"].max()),
    value=float(df["Level"].min()),
    step=0.1
)

predict_clicked = st.sidebar.button("Predict Salary", use_container_width=True)

st.sidebar.caption(
    "ℹ️ The dataset only contains whole-number levels (1–10). "
    "Picking a decimal level (e.g. 2.8) makes the model **interpolate** "
    "a value between the nearest known points — so it won't exactly match "
    "any row in the dataset."
)

# ---------------------------------------------------------
# Train Model (cached per degree so switching is fast)
# ---------------------------------------------------------
@st.cache_resource
def train_model(_x, _y, deg):
    poly = PolynomialFeatures(degree=deg)
    x_poly = poly.fit_transform(_x)
    m = LinearRegression()
    m.fit(x_poly, _y)
    return poly, m

poly_fet, model = train_model(x, y, degree)

x_poly_all = poly_fet.transform(x)
y_pred_train = model.predict(x_poly_all)

mae = mean_absolute_error(y, y_pred_train)
mse = mean_squared_error(y, y_pred_train)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred_train)

# ---------------------------------------------------------
# Prediction Result
# ---------------------------------------------------------
st.subheader("Prediction Result")

if predict_clicked:
    level_poly = poly_fet.transform([[level]])
    prediction = model.predict(level_poly)[0][0]
    st.success(f"Predicted Salary for Level **{level}** (degree {degree}): ₹ {prediction:,.2f}")

    # Show nearest actual data points for context
    lower = df[df["Level"] <= level].tail(1)
    upper = df[df["Level"] >= level].head(1)
    if not lower.empty and not upper.empty and lower["Level"].values[0] != upper["Level"].values[0]:
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                f"Nearest lower level ({lower['Level'].values[0]})",
                f"₹ {lower['Salary'].values[0]:,.0f}"
            )
        with col2:
            st.metric(
                f"Nearest upper level ({upper['Level'].values[0]})",
                f"₹ {upper['Salary'].values[0]:,.0f}"
            )
    elif not lower.empty and lower["Level"].values[0] == level:
        actual_salary = lower["Salary"].values[0]
        diff = prediction - actual_salary
        st.caption(
            f"Level {level} exists in the dataset with an **actual** salary of ₹ {actual_salary:,.0f}. "
            f"The model's prediction differs by ₹ {diff:,.0f} — this is normal: with only "
            f"{len(df)} data points, a degree-{degree} polynomial (R² = {r2:.4f}) fits closely "
            f"but not perfectly through every point."
        )
else:
    st.info("Set the degree and employee level in the sidebar, then click **Predict Salary**.")

# ---------------------------------------------------------
# Regression Curve Plot
# ---------------------------------------------------------
st.subheader("Regression Curve")

fig, ax = plt.subplots(figsize=(7, 4.5))

# smooth curve for a clean plot
x_grid = np.linspace(df["Level"].min(), df["Level"].max(), 200).reshape(-1, 1)
y_grid = model.predict(poly_fet.transform(x_grid))

ax.scatter(x, y, color="red", label="Actual data", zorder=3)
ax.plot(x_grid, y_grid, color="blue", label=f"Polynomial fit (degree {degree})", zorder=2)

if predict_clicked:
    ax.scatter([level], [prediction], color="green", s=100, marker="*",
               label=f"Prediction (Level {level})", zorder=4)

ax.set_xlabel("Level")
ax.set_ylabel("Salary")
ax.set_title("Level vs Salary")
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)

# ---------------------------------------------------------
# Model Performance
# ---------------------------------------------------------
with st.expander("View Model Performance (on training data)"):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", f"{mae:,.0f}")
    c2.metric("MSE", f"{mse:,.0f}")
    c3.metric("RMSE", f"{rmse:,.0f}")
    c4.metric("R² Score", f"{r2:.4f}")

    st.caption("Compare across degrees:")
    comparison_rows = []
    for d in range(1, 7):
        p, m = train_model(x, y, d)
        yp = m.predict(p.transform(x))
        comparison_rows.append({
            "Degree": d,
            "R²": round(r2_score(y, yp), 4),
            "RMSE": round(np.sqrt(mean_squared_error(y, yp)), 2)
        })
    st.dataframe(pd.DataFrame(comparison_rows), hide_index=True, use_container_width=True)

# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------
with st.expander("View Dataset"):
    st.dataframe(df, use_container_width=True)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.write("Built with ❤️ using Streamlit")
