import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model_klasifikasi_lemon.joblib")

st.set_page_config(
	page_title="Klasifikasi Lemon",
	page_icon=":lemon:"
)

st.title(":lemon: Klasifikasi Lemon")
st.markdown("Aplikasi klasifikasi lemon untuk memprediksi lemon grade A, grade B dan reject.")

diameter = st.slider("Diameter", 45.0, 68.0, 56.0)
berat = st.slider("Berat", 70, 145, 135)
tebal_kulit = st.slider("Tebal Kulit", 3.4, 6.0, 3.8)
kadar_gula = st.slider("Kadar Gula", 6.7, 8.6, 7.5)
asal_daerah = st.pills("Asal Daerah", ["California","Malang","Medan"], default="Medan")
warna = st.pills("Warna", ["Hijau pekat", "Kuning kehijauan", "Kuning cerah"], default="Kuning kehijauan")
musim_panen = st.pills("Musim Panen", ["Awal", "Puncak", "Akhir"], default="Akhir")

if st.button("Prediksi", type="primary") :
	data = pd.DataFrame([[diameter,berat,tebal_kulit,kadar_gula,asal_daerah,warna,musim_panen]], 
                    columns=["diameter","berat","tebal_kulit","kadar_gula","asal_daerah","warna","musim_panen"])
	prediksi = model.predict(data)[0]
	presentase = max(model.predict_proba(data)[0])
	st.success(f"Prediksi **{prediksi}** dengan tingkat keyakinan **{presentase*100:.2f}%**")
	st.balloons()

st.divider()
st.caption("Dibuat dengan :lemon: oleh Nadya")