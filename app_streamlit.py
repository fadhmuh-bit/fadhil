import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
	page_title = "klasifikasi buah jeruk",
	page_icon = ":tangerine:"
)

model = joblib.load("ujicoba.joblib")

st.title(":tangerine: Belajar Klasifikasi Jeruk")
st.markdown("Aplikasi machine learning classification untuk memprediksi kualitas jeruk")

ukuran_cm = st.slider("Ukuran Cm", 6.0, 10.0, 8.0)
berat_gram = st.slider("Berat gram", 150.0, 280.0, 200.0)
tingkat_kemanisan = st.slider("Tingkat Kemanisan", 7.0, 14.0, 10.0)
tingkat_keasaman = st.slider("Tingkat Keasaman", 2, 6, 4)
warna_kulit = st.selectbox("Warna Kulit", ["Hijau","Kuning","Oranye"])
tekstur_kulit = st.selectbox("Tekstur Kulit",["Halus","Sedikit Kasar","Kasar"])
musim = st.selectbox("Musim Panen",["Kemarau","Hujan"])

if st.button("Prediksi", type="primary"):
	data_baru = pd.DataFrame([[ukuran_cm,berat_gram,tingkat_kemanisan,tingkat_keasaman,warna_kulit,tekstur_kulit,musim]], columns=["ukuran_cm","berat_gram","tingkat_kemanisan","tingkat_keasaman","warna_kulit","tekstur_kulit","musim"])
	prediksi = model.predict(data_baru)[0]
	presentase = max(model.predict_proba(data_baru)[0])
	st.success(f"model memprediksi **{prediksi}** dengan tingkat keyakinan **{presentase*100:.2f}%**")
	st.snow()

st.divider()
st.caption("Dibuat oleh **Warga Negara Indonesia**")