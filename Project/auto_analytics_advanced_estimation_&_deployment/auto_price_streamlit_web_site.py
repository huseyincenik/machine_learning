import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
import pickle

st.markdown("---")

# Initialize session_state if it's the first run
if 'data_button_clicked' not in st.session_state:
    st.session_state.data_button_clicked = False

# Data button
data_button, _, _ = st.columns(3)
data_button_key = "data_button"  # Unique key
if data_button.button("Data", key=data_button_key):
    st.session_state.data_button_clicked = not st.session_state.data_button_clicked

# Initialize 'info_button_clicked' and 'links_button_clicked' if it's the first run
if 'info_button_clicked' not in st.session_state:
    st.session_state.info_button_clicked = False
if 'links_button_clicked' not in st.session_state:
    st.session_state.links_button_clicked = False

# HTML template
html_temp = """
<div style="background-color:tomato;padding:10px">
<h2 style="color:white;text-align:center;">Streamlit ML Cloud App </h2>
"""
st.markdown(html_temp, unsafe_allow_html=True)
st.markdown("")

# Data display
if st.session_state.data_button_clicked:
    # Data descriptions
    data_descriptions = {
        'make_model': 'Car model',
        'gearbox': 'Gearbox type',
        'drivetrain': 'Drivetrain type',
        'power_kW': 'Engine power (kW)',
        'age': 'Age',
        'empty_weight': 'Empty weight',
        'mileage': 'Mileage',
        'gears': 'Number of gears',
        'cons_avg': 'Average consumption',
        'co_emissions': 'CO emissions',
    }

    # Data as a DataFrame
    data_df = pd.DataFrame(list(data_descriptions.items()), columns=['Variable', 'Description'])

    # Show data descriptions
    st.markdown("---")
    st.dataframe(data_df)
    st.markdown("\n\n")

# Buttons
info_button, links_button, _ = st.columns(3)

# Information button
info_button_key = "info_button"  # Unique key
if info_button.button("Information", key=info_button_key):
    st.session_state.info_button_clicked = not st.session_state.info_button_clicked

# Links button
links_button_key = "links_button"  # Unique key
if links_button.button("Links", key=links_button_key):
    st.session_state.links_button_clicked = not st.session_state.links_button_clicked

# Information markdown
info_markdown = """
This project revolves around using machine learning algorithms to estimate car prices. 

The following regression algorithms were implemented:
- Linear Regression
- Lasso Regression
- Ridge Regression
- Decision Tree
- Random Forest
- XGBoost

Model evaluation, grid-search and cross-validation were performed, resulting in the following scores:

| Model             | R2    | MAE    | RMSE   | MAPE   |
|-------------------|-------|--------|--------|--------|
| XGBoost           | 0.921 | 2123.94| 3373.07| 0.132  |
| Random Forest     | 0.921 | 2252.57| 3374.97| 0.150  |
| Lasso             | 0.831 | 2818.00| 4954.25| 0.192  |
| Linear Regression | 0.830 | 2818.65| 4957.25| 0.192  |
| ElasticNet        | 0.830 | 2817.18| 4959.12| 0.192  |
| Decision Tree     | 0.816 | 3467.44| 5157.75| 0.221  |
"""

# Links HTML
links_html = """
<div style="margin-bottom: 20px;">
    <h3 style="background-color: #FF6961; color: white; padding: 10px; border-radius: 5px;">🚗📈 Auto Analytics: Advanced Estimation & Deployment 🛠️</h3>
    <ul style="list-style-type: none; padding: 0;">
        <li style="margin-bottom: 10px;">
            <a style="color: red;" href="https://github.com/huseyincenik/deep_learning/tree/main/Projects/churn_prediction_in_banking" target="_blank">
                <b>Github Notebook Link</b>
            </a>
        </li>
        <li style="margin-bottom: 10px;">
            <a style="color: blue;" href="https://www.kaggle.com/huseyincenik/churn-prediction-in-banking-deep-learning-approach" target="_blank">
                <b>Kaggle Notebook Link</b>
            </a>
        </li>
        <li style="margin-bottom: 10px;">
            <a style="color: yellow;" href="https://nbviewer.org/github/huseyincenik/nlp_natural_language_processing/blob/main/sentiment_analysis_predicting_product_recommendations_via_text_analysis/sentiment_analysis_predicting_product_recommendations_via_text_analysis.ipynb" target="_blank">
                <b>Nbviewer Notebook</b>
            </a>
        </li>
        <li style="margin-bottom: 10px;">
            <a style="color: green;" href="https://www.linkedin.com/in/huseyincenik/" target="_blank">
                <b>My Linkedin Account</b>
            </a>
        </li>
        <li style="margin-bottom: 10px;">
           <a style="color: orange;" href="https://www.linkedin.com/in/huseyincenik/" target="_blank">
                <b>Streamlit Live</b>
            </a>
        </li>
    </ul>
    <a href="https://www.google.com.tr" target="_blank">
    </a>
</div>
"""

# Show Information
if st.session_state.info_button_clicked:
    st.markdown("---")
    st.markdown(f"<div style='text-align: center; padding: 10px; border: 1px solid #e0e0e0; border-radius: 10px;'>{info_markdown}</div>", unsafe_allow_html=True)
    st.markdown("\n\n")

# Show Links
if st.session_state.links_button_clicked:
    st.markdown("---")
    st.markdown(f"<div style='text-align: center; padding: 10px; border: 1px solid #e0e0e0; border-radius: 10px;'>{links_html}</div>", unsafe_allow_html=True)
    st.markdown("\n\n")

html_temp_2 = """
<div style ="margin-top:20px"> <img src="https://miro.medium.com/v2/resize:fit:1000/1*GDjVt1eUGYVOxn1d04g7uw.jpeg" alt ="Car Image" style = "display:block;margin:auto; width:500px;height:auto;"> </div>
"""
st.markdown(html_temp_2,unsafe_allow_html=True)



st.markdown("---")

# title of the sidebar
html_temp = """
<div style="background-color:green;padding:10px">
<h2 style="color:white;text-align:center;">Car Price Prediction </h2>
</div>"""

st.sidebar.markdown(html_temp,unsafe_allow_html=True)


selected_algorithm = st.sidebar.selectbox("Select Algorithm", ["Random Forest", "XGBoost"], index = 0)

# Load the appropriate CSV file for the selected algorithm
if selected_algorithm == "Random Forest":
    df = pd.read_csv("./rf_data.csv")
    # data_filename = "rf_data.csv"
elif selected_algorithm == "XGBoost":
    df = pd.read_csv("./xgb_data.csv")
    
else:
    st.error("Invalid Selection!")


st.header("Training Dataframe is below:")
st.markdown("---")
st.write(df.sample(5))


make_model = st.sidebar.selectbox("Select Auto Brand - Model", df["make_model"].unique(), index = 28)
gearbox = st.sidebar.selectbox("Select Gearbox", df["gearbox"].unique(), index = 1)
drivetrain = st.sidebar.selectbox("Select Drivetrain", df["drivetrain"].unique(), index = 1)
power_kw = st.sidebar.number_input("Enter Power (in kW)", min_value = df["power_kW"].min(), max_value = df["power_kW"].max(), value = df["power_kW"].mode().iloc[0], step = 1.0)
age = st.sidebar.number_input("Enter Age", min_value = df["age"].min(), max_value = df["age"].max(), value = df["age"].mode().iloc[0], step = 1.0)
empty_weight = st.sidebar.number_input("Enter Empty Weight", min_value = df["empty_weight"].min(), max_value = df["empty_weight"].max(), value = df["empty_weight"].mode().iloc[0], step = 1.0)
mileage = st.sidebar.number_input("Enter the Mileage", min_value = df["mileage"].min(), max_value = df["mileage"].max(), value = df["mileage"].mode().iloc[0], step = 1.0)
gears = st.sidebar.number_input("Enter Gears", min_value = df["gears"].min(), max_value = df["gears"].max(), value = df["gears"].mode().iloc[0], step = 1.0)
cons_avg = st.sidebar.number_input("Enter Consumption", min_value = df["cons_avg"].min(), max_value = df["cons_avg"].max(), value = df["cons_avg"].mode().iloc[0], step = 1.0)
co_emissions = st.sidebar.number_input("Enter Average CO Emissions", min_value = df["co_emissions"].min(), max_value = df["co_emissions"].max(), value = df["co_emissions"].mode().iloc[0], step = 1.0)

# To load machine learning model


model_xgb = pickle.load(open("xgb_pipe_model", "rb"))
model_rf = pickle.load(open("rf_pipe_model", "rb"))


my_dict = {"power_kW":power_kw,
           "age":age,
           "empty_weight": empty_weight,
           "mileage": mileage,
           "gears": gears,
           "cons_avg": cons_avg,
           "co_emissions": co_emissions,
           "make_model": make_model,
           "gearbox": gearbox,
           "drivetrain":drivetrain}

st.header("The values you selected is below")
st.markdown("---")
# Dictionary'i DataFrame'e çevirme
df_input = pd.DataFrame.from_dict([my_dict])

# Sıralama
df_input = df_input[["make_model", "gearbox", "drivetrain", "power_kW", "age", "empty_weight", "mileage", "gears", "cons_avg", "co_emissions"]]

# Tabloyu görüntüleme
st.table(df_input)



st.title("Car Prediction")

# Seçilen make_model'den ilk kelimeyi al ve küçük harfe çevir
make_model_lower = make_model.split()[0].lower()

# Resimleri içeren klasör yolu
pictures_folder = "Picture"

# make_model'e ait resmi bul
png_image_path = os.path.join(pictures_folder, f"{make_model_lower}.png")
jpg_image_path = os.path.join(pictures_folder, f"{make_model_lower}.jpg")

# Resmi görüntüle
try:
    # Önce PNG resmi dene
    image = Image.open(png_image_path)
except FileNotFoundError:
    try:
        # PNG bulunamazsa JPG resmi dene
        image = Image.open(jpg_image_path)
    except FileNotFoundError:
        st.warning(f"Resim bulunamadı: {png_image_path} veya {jpg_image_path}")
        st.stop()

# Resmi 60x60 piksel boyutunda göster
image = image.resize((256, 256))

st.image(image, caption=make_model, use_column_width=80)
# defining the function which will make the prediction using the data
def prediction(model, input_data):
	prediction = model.predict(input_data)
	return prediction

# Making prediction and displaying results
if st.button("Predict"):
    if selected_algorithm == "Random Forest":
        result = prediction(model_rf, df_input)[0]
    else :
        result = prediction(model_xgb, df_input)[0]

try:
    st.success(f"With {selected_algorithm}, Car Price is **{round(result,0)}**")
except NameError:
    st.write("Please press **Predict** button to display the result!")