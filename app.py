import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('models/best_mlp_model.pkl')

# Konfigurasi halaman
st.set_page_config(page_title="Nursery Classifier", page_icon="🌸", layout="centered")

# Judul Aplikasi
st.title("🌸 Nursery Admission Classifier")
st.markdown("Klasifikasi penerimaan anak TK berdasarkan kondisi sosial ekonomi keluarga dan anak.")

st.markdown("---")

# Sidebar Input Data
st.sidebar.header("📝 Input Data Anak")

parents = st.sidebar.selectbox("👨‍👩‍👧 Parents", ['usual', 'pretentious', 'great_pret'])
has_nurs = st.sidebar.selectbox("🏫 Has Nursery", ['proper', 'less_proper', 'improper', 'critical', 'very_crit'])
form = st.sidebar.selectbox("📄 Form", ['complete', 'completed', 'incomplete', 'foster'])
children = st.sidebar.selectbox("👶 Children", ['1', '2', '3', 'more'])
housing = st.sidebar.selectbox("🏠 Housing", ['convenient', 'less_conv', 'critical'])
finance = st.sidebar.selectbox("💰 Finance", ['convenient', 'inconv'])
social = st.sidebar.selectbox("📊 Social", ['nonprob', 'slightly_prob', 'problematic'])
health = st.sidebar.selectbox("🩺 Health", ['recommended', 'priority', 'not_recom'])

# Mapping manual ke angka
parents_mapping = {'usual':0, 'pretentious':1, 'great_pret':2}
has_nurs_mapping = {'proper':0, 'less_proper':1, 'improper':2, 'critical':3, 'very_crit':4}
form_mapping = {'complete':0, 'completed':1, 'incomplete':2, 'foster':3}
children_mapping = {'1':0, '2':1, '3':2, 'more':3}
housing_mapping = {'convenient':0, 'less_conv':1, 'critical':2}
finance_mapping = {'convenient':0, 'inconv':1}
social_mapping = {'nonprob':0, 'slightly_prob':1, 'problematic':2}
health_mapping = {'recommended':0, 'priority':1, 'not_recom':2}
label_mapping = {0:'🚫 Not Recommended', 1:'✅ Very Recommended', 2:'🔶 Priority', 3:'⭐ Special Priority'}

# Tombol Prediksi
if st.sidebar.button("🎯 Predict Now"):
    data = [[
        parents_mapping[parents],
        has_nurs_mapping[has_nurs],
        form_mapping[form],
        children_mapping[children],
        housing_mapping[housing],
        finance_mapping[finance],
        social_mapping[social],
        health_mapping[health],
    ]]

    prediction = model.predict(data)[0]

    st.markdown("## 📊 Hasil Prediksi:")
    st.success(f"Hasil klasifikasi: **{label_mapping[prediction]}**")

    if prediction == 0:
        st.info("❌ Anak ini tidak direkomendasikan untuk diterima.")
    elif prediction == 1:
        st.success("✅ Sangat direkomendasikan untuk diterima!")
    elif prediction == 2:
        st.warning("⚠️ Memiliki prioritas untuk diterima.")
    elif prediction == 3:
        st.success("🌟 Mendapatkan prioritas khusus!")

st.markdown("---")
