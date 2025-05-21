# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from scipy.stats.mstats import winsorize
# LabelEncoder will be needed for categorical inputs if not handled by a saved preprocessor pipeline component
from sklearn.preprocessing import LabelEncoder

# --- Load Saved Artifacts ---
# Use st.cache_resource to load these only once
@st.cache_resource
def load_artifacts():
    try:
        MODEL_LOAD_DIR = "models/" # For Kaggle output, it's "/kaggle/working/models/"
                                   # Adjust if running locally, e.g., "." or "./models/"

        preprocessor_path = os.path.join(MODEL_LOAD_DIR, "preprocessor.joblib")
        best_model_filename = "SVC_best_model.joblib" # From notebook output
        best_model_path = os.path.join(MODEL_LOAD_DIR, best_model_filename)

        preprocessor = joblib.load(preprocessor_path)
        model = joblib.load(best_model_path)
        st.success("Preprocessor and Model loaded successfully.")

        # Unique classes for target (1-5 for stress level)
        unique_classes = [1, 2, 3, 4, 5]

        # Imputation values from notebook's EDA (median for numeric, mode for categorical)
        imputation_values = {
            'Rata_Jam_Kerja_Per_Hari': 9.1,  # From notebook's df.describe() or imputer
            'Wilayah_Kerja': 'Karnataka'     # From notebook's imputer or mode analysis
        }
        # Winsorizing limits from notebook
        winsorize_limits = [0.05, 0.05]

        # Mappings for LabelEncoders (these should ideally be saved from notebook)
        # For now, hardcoding based on typical LabelEncoder behavior (alphabetical)
        # or what was observed/set in the notebook.
        # Notebook used LabelEncoder.fit_transform without saving the encoders.
        # The preprocessor (ColumnTransformer) expects these to be numerically encoded.
        label_encoder_mappings = {
            'Lokasi_Bekerja': {'Hybrid': 0, 'Kantor': 1, 'Rumah': 2},
            'Work_Life_Balance': {'Iya': 0, 'Tidak': 1}, # Notebook's LE for 'Iya', 'Tidak'
            'Tinggal_Bersama_Keluarga': {'Iya': 0, 'Tidak': 1},
            # From notebook: le.classes_ for Wilayah_Kerja likely ['Chennai', 'Delhi', 'Hyderabad', 'Karnataka', 'Pune']
            'Wilayah_Kerja': {'Chennai': 0, 'Delhi': 1, 'Hyderabad': 2, 'Karnataka': 3, 'Pune': 4}
        }
        # Map for 'Keseimbangan_Kerja_Hidup_Numeric'
        map_ya_tidak_fe = {'Iya': 1, 'Tidak': 0}


        return preprocessor, model, unique_classes, imputation_values, winsorize_limits, label_encoder_mappings, map_ya_tidak_fe

    except FileNotFoundError as e:
        st.error(f"Error: Model or preprocessor file not found. {e}")
        return None, None, None, None, None, None, None
    except Exception as e:
        st.error(f"An error occurred during loading: {e}")
        return None, None, None, None, None, None, None


preprocessor, model, unique_classes, imputation_values, winsorize_limits, label_encoder_mappings, map_ya_tidak_fe = load_artifacts()

if model is not None and preprocessor is not None:
    st.title("Prediksi Tingkat Stres Karyawan")
    st.write("Inputkan data karyawan untuk memprediksi tingkat stres (1=Sangat Rendah, 5=Sangat Tinggi).")

    with st.form("prediction_form"):
        st.header("Data Karyawan")

        # Original features as per notebook
        # Rata_Jam_Kerja_Per_Hari: float
        # Lokasi_Bekerja: object (Hybrid, Kantor, Rumah)
        # Tekanan_Pekerjaan: int (1-5)
        # Dukungan_Atasan: int (1-5)
        # Kebiasaan_Tidur: int (1-5)
        # Kebiasaan_Olahraga: int (1-5)
        # Kepuasan_Kerja: int (1-5)
        # Work_Life_Balance: object (Iya, Tidak)
        # Kepribadian_Sosial: int (1-5)
        # Tinggal_Bersama_Keluarga: object (Iya, Tidak)
        # Wilayah_Kerja: object (Karnataka, Pune, Delhi, Hyderabad, Chennai)

        rata_jam_kerja = st.number_input("Rata-rata Jam Kerja per Hari", min_value=4.0, max_value=16.0, value=9.1, step=0.1, format="%.1f")
        lokasi_bekerja_options = list(label_encoder_mappings['Lokasi_Bekerja'].keys())
        lokasi_bekerja = st.selectbox("Lokasi Bekerja", options=lokasi_bekerja_options, index=lokasi_bekerja_options.index('Rumah'))

        tekanan_pekerjaan = st.selectbox("Tekanan Pekerjaan (1-5)", options=[1, 2, 3, 4, 5], index=2)
        dukungan_atasan = st.selectbox("Dukungan Atasan (1-5)", options=[1, 2, 3, 4, 5], index=2)
        kebiasaan_tidur = st.selectbox("Kebiasaan Tidur (1-5)", options=[1, 2, 3, 4, 5], index=2)
        kebiasaan_olahraga = st.selectbox("Kebiasaan Olahraga (1-5)", options=[1, 2, 3, 4, 5], index=2)
        kepuasan_kerja = st.selectbox("Kepuasan Kerja (1-5)", options=[1, 2, 3, 4, 5], index=2)

        work_life_balance_options = list(label_encoder_mappings['Work_Life_Balance'].keys())
        work_life_balance = st.selectbox("Work Life Balance", options=work_life_balance_options, index=work_life_balance_options.index('Tidak'))

        kepribadian_sosial = st.selectbox("Kepribadian Sosial (1-5)", options=[1, 2, 3, 4, 5], index=2)

        tinggal_keluarga_options = list(label_encoder_mappings['Tinggal_Bersama_Keluarga'].keys())
        tinggal_bersama_keluarga = st.selectbox("Tinggal Bersama Keluarga", options=tinggal_keluarga_options, index=tinggal_keluarga_options.index('Iya'))

        wilayah_kerja_options = list(label_encoder_mappings['Wilayah_Kerja'].keys())
        wilayah_kerja = st.selectbox("Wilayah Kerja", options=wilayah_kerja_options, index=wilayah_kerja_options.index('Karnataka'))

        submitted = st.form_submit_button("Prediksi Tingkat Stres")

    if submitted:
        input_data_raw = {
            'Rata_Jam_Kerja_Per_Hari': [rata_jam_kerja],
            'Lokasi_Bekerja': [lokasi_bekerja],
            'Tekanan_Pekerjaan': [tekanan_pekerjaan],
            'Dukungan_Atasan': [dukungan_atasan],
            'Kebiasaan_Tidur': [kebiasaan_tidur],
            'Kebiasaan_Olahraga': [kebiasaan_olahraga],
            'Kepuasan_Kerja': [kepuasan_kerja],
            'Work_Life_Balance': [work_life_balance],
            'Kepribadian_Sosial': [kepribadian_sosial],
            'Tinggal_Bersama_Keluarga': [tinggal_bersama_keluarga],
            'Wilayah_Kerja': [wilayah_kerja]
        }
        input_df = pd.DataFrame(input_data_raw)

        # 1. Imputation (using defaults from form if not provided, or learned values)
        # For single prediction, imputing NaNs usually not an issue with st.inputs
        # If a field could be None and needs imputation:
        if input_df['Rata_Jam_Kerja_Per_Hari'].isnull().any():
            input_df['Rata_Jam_Kerja_Per_Hari'].fillna(imputation_values['Rata_Jam_Kerja_Per_Hari'], inplace=True)
        if input_df['Wilayah_Kerja'].isnull().any(): # Should not happen with selectbox
            input_df['Wilayah_Kerja'].fillna(imputation_values['Wilayah_Kerja'], inplace=True)

        # 2. Winsorizing
        input_df['Avg_Work_Hours_Winsorized'] = winsorize(input_df['Rata_Jam_Kerja_Per_Hari'].astype(float), limits=winsorize_limits)
        input_df.drop('Rata_Jam_Kerja_Per_Hari', axis=1, inplace=True)

        # 3. Feature Engineering: Keseimbangan_Kerja_Hidup_Numeric
        input_df['Keseimbangan_Kerja_Hidup_Numeric'] = input_df['Work_Life_Balance'].map(map_ya_tidak_fe)
        # Work_Life_Balance itself will be LabelEncoded later as per notebook flow

        # 4. Label Encoding for remaining categorical columns
        # The ColumnTransformer (preprocessor) expects these to be numeric
        # These are the columns that were 'object' type before LE in the notebook
        for col, mapping in label_encoder_mappings.items():
            input_df[col] = input_df[col].map(mapping)

        # Ensure column order matches X_train when preprocessor was fitted
        # Order from notebook:
        # 'Lokasi_Bekerja', 'Tekanan_Pekerjaan', 'Dukungan_Atasan', 'Kebiasaan_Tidur',
        # 'Kebiasaan_Olahraga', 'Kepuasan_Kerja', 'Work_Life_Balance', 'Kepribadian_Sosial',
        # 'Tinggal_Bersama_Keluarga', 'Wilayah_Kerja', 'Avg_Work_Hours_Winsorized',
        # 'Keseimbangan_Kerja_Hidup_Numeric'
        column_order = [
            'Lokasi_Bekerja', 'Tekanan_Pekerjaan', 'Dukungan_Atasan', 'Kebiasaan_Tidur',
            'Kebiasaan_Olahraga', 'Kepuasan_Kerja', 'Work_Life_Balance', 'Kepribadian_Sosial',
            'Tinggal_Bersama_Keluarga', 'Wilayah_Kerja', 'Avg_Work_Hours_Winsorized',
            'Keseimbangan_Kerja_Hidup_Numeric'
        ]
        input_df_ordered = input_df[column_order]


        # 5. Scaling (using the loaded ColumnTransformer)
        processed_input = preprocessor.transform(input_df_ordered)


        # Make prediction (model predicts 0-4)
        prediction_0based = model.predict(processed_input)[0]
        prediction_1based = int(prediction_0based + 1) # Convert to 1-5

        st.subheader("Hasil Prediksi")
        if prediction_1based in unique_classes:
            stress_level_description = {
                1: "Sangat Rendah",
                2: "Rendah",
                3: "Sedang",
                4: "Tinggi",
                5: "Sangat Tinggi"
            }
            st.success(f"Tingkat Stres Karyawan: **{prediction_1based}** ({stress_level_description.get(prediction_1based, 'Tidak Diketahui')})")
        else:
            st.warning(f"Prediksi menghasilkan nilai di luar rentang (1-5): {prediction_1based}")
else:
    st.error("Model atau preprocessor gagal dimuat. Prediksi tidak dapat dilakukan.")