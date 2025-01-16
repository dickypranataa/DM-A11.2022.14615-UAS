import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Fungsi untuk memuat model
@st.cache_resource
def load_model(file_name):
    with open(file_name, 'rb') as f:
        model = pickle.load(f)
    return model

# Fungsi utama aplikasi
def main():
    st.title("Prediksi Keterlambatan Pengiriman Produk")

    # Pilihan model
    st.header("Pilih Model")
    model_option = st.selectbox("Pilih Model", ["LightGBM", "XGBoost"])
    model_file = "lightgbm_model.pkl" if model_option == "LightGBM" else "xgboost_model.pkl"
    model = load_model(model_file)

    # Input data pengguna
    st.header("Masukkan Data")
    warehouse_block = st.selectbox('Warehouse Block (Pilih Gudang)', ['A', 'B', 'C', 'D', 'F'])
    mode_of_shipment = st.selectbox('Mode of Shipment ', ['Flight', 'Ship', 'Road'])
    customer_care_calls = st.slider('Customer Care Calls (Jumlah panggilan ke layanan pelanggan.)', 0, 10, 5)
    customer_rating = st.slider('Customer Rating', 1, 5, 3)
    cost_of_the_product = st.number_input('Cost of the Product', min_value=0.0, step=1.0)
    prior_purchases = st.slider('Prior Purchases (Jumlah pembelian sebelumnya)', 0, 10, 2)
    product_importance = st.selectbox('Product Importance (Tingkat kepentingan produk)', ['low', 'medium', 'high'])
    gender = st.selectbox('Gender', ['F', 'M'])
    discount_offered = st.number_input('Discount Offered', min_value=0.0, step=1.0)
    weight_in_gms = st.number_input('Weight (grams)', min_value=0.0, step=1.0)

    # Validasi input untuk mencegah pembagian dengan nol
    if weight_in_gms == 0:
        st.warning("Weight in grams tidak boleh bernilai nol. Masukkan nilai yang valid.")
    if cost_of_the_product == 0:
        st.warning("Cost of the Product tidak boleh bernilai nol. Masukkan nilai yang valid.")

    # Mapping untuk encoding variabel kategori
    encoder = {
        'Warehouse_block': {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'F': 4},
        'Mode_of_Shipment': {'Flight': 0, 'Ship': 1, 'Road': 2},
        'Product_importance': {'low': 0, 'medium': 1, 'high': 2},
        'Gender': {'F': 0, 'M': 1}
    }

    # Preprocessing data
    input_data = pd.DataFrame({
        'Warehouse_block': [encoder['Warehouse_block'][warehouse_block]],
        'Mode_of_Shipment': [encoder['Mode_of_Shipment'][mode_of_shipment]],
        'Customer_care_calls': [customer_care_calls],
        'Customer_rating': [customer_rating],
        'Cost_of_the_Product': [cost_of_the_product],
        'Prior_purchases': [prior_purchases],
        'Product_importance': [encoder['Product_importance'][product_importance]],
        'Gender': [encoder['Gender'][gender]],
        'Discount_offered': [discount_offered],
        'Weight_in_gms': [weight_in_gms],
        'Cost_per_Weight': [cost_of_the_product / weight_in_gms if weight_in_gms != 0 else 0],
        'Discount_Impact': [discount_offered / cost_of_the_product if cost_of_the_product != 0 else 0]
    })

    # Normalisasi (sesuaikan dengan scaler yang digunakan)
    scaler = load_model('scaler.pkl')
    input_scaled = scaler.transform(input_data)

    # Prediksi
    if st.button("Prediksi"):
        prediction = model.predict(input_scaled)
        result = "Tepat Waktu" if prediction[0] == 1 else "Terlambat"
        st.success(f"Hasil Prediksi: {result}")

if __name__ == "__main__":
    main()
