# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn # Import torch.nn if using PyTorch MLP
# Import winsorize if you replicated winsorizing manually
from scipy.stats.mstats import winsorize # Needed if winsorizing manually
import category_encoders as ce # Needed if TargetEncoder is loaded separately (though notebook structure suggests it's inside CT)

# --- Define the PyTorch Model Architecture (if your best model is PyTorch) ---
# You NEED to copy the SAME class definition used during training here
# if your best model is the PyTorch MLP (.pth file).
# Example (copy your actual class definitions):
class ResidualBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.fc = nn.Linear(size, size)
        self.bn = nn.BatchNorm1d(size)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn(self.fc(x)))
        out += residual # Skip connection
        return self.relu(out)

class EnhancedMLP(nn.Module):
    def __init__(self, input_size, num_classes, hidden_sizes=[256, 128, 64], dropout_rate=0.4, use_residual=True):
        super().__init__()
        layers = []
        current_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            if use_residual and current_size == hidden_size and i > 0:
                layers.append(ResidualBlock(hidden_size))
            current_size = hidden_size

        layers.append(nn.Linear(current_size, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
# --- End of PyTorch Model Definition ---


# --- Load Saved Artifacts ---
# Use st.cache_resource to load these only once across user interactions
@st.cache_resource
def load_artifacts():
    try:
        # Define the directory where you saved the models
        # Adjust this path based on where you put the saved files
        # If running locally, this might be a relative path or an absolute path
        # If running on Kaggle, it's likely in /kaggle/working/models/
        MODEL_LOAD_DIR = "models/" # <<< ADJUST THIS PATH if running locally, e.g., "." or "./models/"
                                   # <<< For Kaggle output, it's "/kaggle/working/models/"

        preprocessor_path = os.path.join(MODEL_LOAD_DIR, "preprocessor_rfe.joblib")
        selector_path = os.path.join(MODEL_LOAD_DIR, "selector_rfe.joblib")

        # --- IMPORTANT: Define the path to your BEST model file ---
        # Replace 'StackingClassifier_Automated_best_model.joblib' with the actual filename
        # determined in your notebook's output (e.g., 'MLPClassifier_SK_best_model.joblib', 'LGBMClassifier_GPU_best_model.pth')
        best_model_filename = "StackingClassifier_Automated_best_model.joblib" # <<< CHANGE THIS FILENAME
        best_model_path = os.path.join(MODEL_LOAD_DIR, best_model_filename)

        # Load preprocessor and selector
        preprocessor = joblib.load(preprocessor_path)
        selector = joblib.load(selector_path)
        st.success("Preprocessor and RFE Selector loaded successfully.")

        # Load the best model
        if best_model_filename.endswith('.pth'): # Load PyTorch model state_dict
             # Need to instantiate the model architecture first
             # You need the input_size and num_classes used during training
             # Assuming input_size_dl and num_classes_dl are available or hardcoded
             # You also need the exact hidden_sizes, dropout_rate, use_residual from the best config
             # This requires knowing the best PyTorch params from your notebook output
             # Example values based on the notebook's best PyTorch config output:
             pytorch_input_size = 22 # <<< CHANGE if your final RFE output size is different
             pytorch_num_classes = 5 # <<< CHANGE if your number of classes is different (should be 5)
             # <<< CHANGE these based on your best PyTorch config from the notebook output
             pytorch_best_hidden_sizes = [256, 128, 64]
             pytorch_best_dropout_rate = 0.4
             pytorch_best_use_residual = True

             model = EnhancedMLP(pytorch_input_size, pytorch_num_classes,
                                 hidden_sizes=pytorch_best_hidden_sizes,
                                 dropout_rate=pytorch_best_dropout_rate,
                                 use_residual=pytorch_best_use_residual)
             # Load state_dict, map to CPU for broader compatibility
             model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
             model.eval() # Set model to evaluation mode
             st.success(f"PyTorch model state_dict loaded from: {best_model_path}")
        else: # Load scikit-learn model (.joblib)
            model = joblib.load(best_model_path)
            st.success(f"Scikit-learn model loaded from: {best_model_path}")

        # You might also need the unique classes (1-5) for display
        unique_classes = [1, 2, 3, 4, 5] # <<< CONFIRM THESE CLASSES

        # --- Hardcoded/Example Imputation and FE values from Training Data ---
        # You should get the ACTUAL calculated values from your notebook's training run
        imputation_values = {
            'Rata_Jam_Kerja_Per_Hari': 9.10, # <<< REPLACE with actual median from training
            'Wilayah_Kerja': 'Karnataka'     # <<< REPLACE with actual modus from training
        }
        # Need winsorizing limits (from notebook: [0.05, 0.05])
        winsorize_limits = [0.05, 0.05]
        # Need Q3 for Long_Work_Hours_Indicator (from notebook, Q3 of Avg_Work_Hours_Winsorized)
        # Example value (replace with actual Q3 from training data):
        work_hours_q3_winsorized_train = 10.3 # <<< REPLACE with actual Q3 from training

        # Need the mean target encoded value for Wilayah_Kerja (from notebook output)
        # Example value (replace with actual mean):
        mean_target_encoded_region_train = 3.0 # <<< REPLACE with actual mean target encoded value from training

        return preprocessor, selector, model, unique_classes, imputation_values, winsorize_limits, work_hours_q3_winsorized_train, mean_target_encoded_region_train

    except FileNotFoundError as e:
        st.error(f"Error: Model or preprocessor file not found. Please ensure the files are in the correct directory. {e}")
        return None, None, None, None, None, None, None, None # Return None if loading fails
    except Exception as e:
        st.error(f"An error occurred during model/preprocessor loading: {e}")
        st.error(f"Details: {e}", icon="🚨")
        return None, None, None, None, None, None, None, None # Return None if loading fails


preprocessor, selector, model, unique_classes, imputation_values, winsorize_limits, work_hours_q3_winsorized_train, mean_target_encoded_region_train = load_artifacts()

if model is not None and preprocessor is not None and selector is not None:
    # --- Streamlit App Interface ---
    st.title("Prediksi Tingkat Stres Karyawan")
    st.write("Inputkan data karyawan di bawah ini untuk memprediksi tingkat stres mereka (1=Sangat Rendah, 5=Sangat Tinggi).")

    # --- Input Form ---
    # Collect inputs for the ORIGINAL features (before feature engineering and RFE)
    # Refer to the original column names in your initial df.head() and df.info()
    with st.form("prediction_form"):
        st.header("Data Karyawan")

        # Example inputs - adjust based on your actual features and their ranges
        # Rata_Jam_Kerja_Per_Hari was float, others were int 1-5, categoricals had specific levels
        rata_jam_kerja = st.number_input("Rata-rata Jam Kerja per Hari", min_value=4.0, max_value=16.0, value=9.0, step=0.1, format="%.1f") # Adjust range/default/format based on data
        lokasi_bekerja = st.selectbox("Lokasi Bekerja", options=['Rumah', 'Kantor', 'Hybrid'])
        tekanan_pekerjaan = st.selectbox("Tekanan Pekerjaan (1-5)", options=[1, 2, 3, 4, 5], index=2) # Default to 3
        dukungan_atasan = st.selectbox("Dukungan Atasan (1-5)", options=[1, 2, 3, 4, 5], index=2) # Default to 3
        kebiasaan_tidur = st.selectbox("Kebiasaan Tidur (1-5)", options=[1, 2, 3, 4, 5], index=2) # Default to 3
        kebiasaan_olahraga = st.selectbox("Kebiasaan Olahraga (1-5)", options=[1, 2, 3, 4, 5], index=2) # Default to 3
        kepuasan_kerja = st.selectbox("Kepuasan Kerja (1-5)", options=[1, 2, 3, 4, 5], index=2) # Default to 3
        work_life_balance = st.selectbox("Work Life Balance", options=['Iya', 'Tidak'])
        kepribadian_sosial = st.selectbox("Kepribadian Sosial (1-5)", options=[1, 2, 3, 4, 5], index=2) # Default to 3
        tinggal_bersama_keluarga = st.selectbox("Tinggal Bersama Keluarga", options=['Iya', 'Tidak'])
        # You need the actual unique values for Wilayah_Kerja from your training data
        # Get these from df['Wilayah_Kerja'].unique() after imputing if needed
        wilayah_kerja_options = ['Karnataka', 'Chennai', 'Hyderabad', 'Pune', 'Delhi'] # <<< CHANGE/CONFIRM THESE OPTIONS
        wilayah_kerja = st.selectbox("Wilayah Kerja", options=wilayah_kerja_options)


        submitted = st.form_submit_button("Prediksi Tingkat Stres")

    # --- Prediction Logic ---
    if submitted:
        # Create a DataFrame from input values
        # Column names MUST match the original DataFrame columns BEFORE preprocessing
        input_data = {
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
        input_df = pd.DataFrame(input_data)

        # --- Replicate Initial Imputation (Modus/Median) ---
        # Apply imputation to the input DataFrame using loaded/hardcoded values
        for col, value in imputation_values.items():
             if col in input_df.columns and input_df[col].isnull().any():
                 input_df[col].fillna(value, inplace=True) # Fill NaNs if any (unlikely with selectbox/number_input)
        # Note: With Streamlit form inputs like number_input and selectbox, NaNs are usually not possible unless min_value is None.

        # --- Replicate Feature Engineering ---
        # Replicate all feature engineering steps done *before* the ColumnTransformer fitting in the notebook
        # WLB_Numeric is intermediate
        map_yes_no_input = {'Iya': 1, 'Tidak': 0}
        input_df['WLB_Numeric'] = input_df['Work_Life_Balance'].map(map_yes_no_input)

        # Replicate engineered features
        input_df['Pressure_vs_Support_Ratio'] = input_df['Tekanan_Pekerjaan'] / (input_df['Dukungan_Atasan'] + 0.01)
        input_df['Sleep_Exercise_Combo'] = input_df['Kebiasaan_Tidur'] + input_df['Kebiasaan_Olahraga']

        # Replicate Winsorizing to get Avg_Work_Hours_Winsorized
        # Requires winsorize function and the limits used during training
        # Note: winsorize might expect a Series, ensure input_df column is Series
        input_df['Avg_Work_Hours_Winsorized'] = winsorize(input_df['Rata_Jam_Kerja_Per_Hari'], limits=winsorize_limits)

        # Replicate features that depend on Avg_Work_Hours_Winsorized
        # Requires the Q3 value from training data
        input_df['Long_Work_Hours_Indicator'] = (input_df['Avg_Work_Hours_Winsorized'] > work_hours_q3_winsorized_train).astype(int)
        input_df['WorkHours_x_Pressure'] = input_df['Avg_Work_Hours_Winsorized'] * input_df['Tekanan_Pekerjaan']

        # Replicate Negative_Factors_Count
        input_df['Negative_Factors_Count'] = (
            (input_df['Tekanan_Pekerjaan'] >= 4).astype(int) +
            (input_df['Dukungan_Atasan'] <= 2).astype(int) +
            (input_df['Kebiasaan_Tidur'] <= 2).astype(int) +
            (input_df['Kebiasaan_Olahraga'] <= 2).astype(int)
        )

        # Replicate Satisfaction_x_WLB
        input_df['Satisfaction_x_WLB'] = input_df['Kepuasan_Kerja'] * input_df['WLB_Numeric']

        # Replicate Target Encoding for Wilayah_Kerja
        # The preprocessor expects a TargetEncoded column.
        # Since we don't load the TargetEncoder object itself, we use the mean value as a simple fill.
        # This is an approximation based on the notebook's apparent flow where the CT uses the *result* of TE.
        input_df['Wilayah_Kerja_TargetEncoded'] = input_df['Wilayah_Kerja'].apply(lambda x: mean_target_encoded_region_train) # Apply the mean for simplicity


        # --- Apply the loaded preprocessor (ColumnTransformer) ---
        # This expects the DataFrame with all the original + engineered features
        processed_input = preprocessor.transform(input_df)

        # --- Apply the loaded RFE selector ---
        # This selects the final set of features
        final_input = selector.transform(processed_input)

        # Make the prediction
        # Handle PyTorch prediction separately
        if isinstance(model, nn.Module):
             # PyTorch model expects torch tensor input
             final_input_tensor = torch.tensor(final_input.astype(np.float32))
             # PyTorch model needs to be on the correct device for prediction
             device_pred = torch.device("cuda" if torch.cuda.is_available() else "cpu")
             model.to(device_pred)
             model.eval() # Ensure eval mode
             with torch.no_grad():
                 outputs = model(final_input_tensor.to(device_pred))
                 _, predicted_tensor = torch.max(outputs.data, 1) # predicted_tensor is 0-indexed
                 prediction = (predicted_tensor.item() + 1) # Convert back to 1-5 index and get scalar
        else: # Scikit-learn model
            prediction = model.predict(final_input)[0] # Get the scalar prediction

        # --- Output Prediction ---
        st.subheader("Hasil Prediksi")
        if prediction in unique_classes:
             stress_level_description = {
                 1: "Sangat Rendah",
                 2: "Rendah",
                 3: "Sedang",
                 4: "Tinggi",
                 5: "Sangat Tinggi"
             }
             st.success(f"Tingkat Stres Karyawan: **{prediction}** ({stress_level_description.get(prediction, 'Tidak Diketahui')})")
        else:
             st.warning(f"Prediksi menghasilkan nilai di luar rentang yang diharapkan (1-5): {prediction}")

else:
    st.error("Model atau preprocessor gagal dimuat. Prediksi tidak dapat dilakukan. Mohon periksa file yang disimpan.")

# --- How to Run ---
# 1. Save this code as a Python file (e.g., app.py)
# 2. Make sure the .joblib / .pth files are in the specified MODEL_LOAD_DIR (or create a 'models' subdirectory and put them there)
# 3. Save the requirements.txt file (see below) in the same directory as app.py
# 4. Open your terminal or command prompt
# 5. Navigate to the directory where you saved the files
# 6. Install dependencies: pip install -r requirements.txt
# 7. Run the command: streamlit run app.py