
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import os
import pennylane as qml
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# CONFIG (Ensure these match the training config)
# =========================
RESULTS = "results"
N_LAYERS = 4 # Needs to match the N_LAYERS used during training

# =========================
# Streamlit Page Configuration
# =========================
st.set_page_config(layout="wide", page_title="IDS ML/QML Dashboard", page_icon="🛡️")

# =========================
# Load Models and Preprocessing Objects
# =========================
@st.cache_resource(show_spinner="Loading ML/QML Models and Preprocessing Components...")
def load_artifacts():
    try:
        # st.write("Loading artifacts...") # This will show in the Streamlit UI
        print("Loading artifacts...") # These will print to console (localtunnel output)

        print("Loading RF model...")
        rf_model = joblib.load(os.path.join(RESULTS, "rf_model.pkl"))
        print("RF model loaded.")

        print("Loading scaler...")
        scaler = joblib.load(os.path.join(RESULTS, "scaler.pkl"))
        print("Scaler loaded.")

        print("Loading PCA...")
        pca = joblib.load(os.path.join(RESULTS, "pca.pkl"))
        print("PCA loaded.")

        print("Loading selector...")
        selector = joblib.load(os.path.join(RESULTS, "selector.pkl"))
        print("Selector loaded.")

        print("Loading min/max for quantum scaling...")
        train_min = np.load(os.path.join(RESULTS, "train_min.npy"))
        train_max = np.load(os.path.join(RESULTS, "train_max.npy"))
        print("Min/max loaded.")

        print("Loading feature columns...")
        feature_columns = joblib.load(os.path.join(RESULTS, "feature_columns.pkl"))
        print("Feature columns loaded.")

        # Load PyTorch CNN model
        cnn_input_dim = pca.n_components_
        if cnn_input_dim == 0:
            st.error("CNN input dimension is zero after PCA. Check your training data and PCA configuration.")
            raise ValueError("CNN input dimension cannot be zero.")
        print(f"Initializing CNN with input_dim: {cnn_input_dim}...")
        cnn_model = CNN(cnn_input_dim)
        cnn_model.load_state_dict(torch.load(os.path.join(RESULTS, "cnn_model_state_dict.pth")))
        cnn_model.eval()
        print("CNN model loaded.")

        # Load PyTorch QCNN model
        qcnn_n_qubits = max(1, pca.n_components_) # Ensure at least 1 qubit for qml.device
        print(f"Initializing QCNN with n_qubits: {qcnn_n_qubits}...")
        qcnn_model = QCNN(qcnn_n_qubits)
        qcnn_model.load_state_dict(torch.load(os.path.join(RESULTS, "qcnn_model_state_dict.pth")))
        qcnn_model.eval()
        print("QCNN model loaded.")

        print("All artifacts loaded successfully.")
        return rf_model, cnn_model, qcnn_model, scaler, pca, selector, train_min, train_max, qcnn_n_qubits, feature_columns

    except Exception as e:
        st.error(f"Error loading models or preprocessing components: {e}")
        st.stop() # Stop the app if critical components can't be loaded

# =========================
# CNN MODEL Class Definition (copied from training script)
# =========================
class CNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

# =========================
# QCNN MODEL Class Definition (copied from training script)
# =========================
class QCNN(nn.Module):
    def __init__(self, n_qubits_effective):
        super().__init__()
        self.n_qubits = n_qubits_effective
        self.weights = nn.Parameter(torch.randn(N_LAYERS, self.n_qubits, 2))
        self.fc = nn.Linear(self.n_qubits, 2)
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def _qcnn_circuit(inputs, weights_param):
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)

            for l in range(N_LAYERS):
                for i in range(self.n_qubits):
                    qml.RY(weights_param[l, i, 0], wires=i)
                    qml.RZ(weights_param[l, i, 1], wires=i)

                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i+1])

            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        self.qcnn_circuit = _qcnn_circuit

    def forward(self, x):
        out = []
        for i in range(x.shape[0]):
            out.append(torch.tensor(self.qcnn_circuit(x[i][:self.n_qubits], self.weights), dtype=torch.float32))

        out = torch.stack(out)
        return self.fc(out)

# =========================
# Feature Engineering (copied from training script)
# =========================
def feature_engineering(X_raw):
    X = X_raw.copy()
    if "duration" in X.columns and X["duration"].dtype != object:
        X["duration"] = X["duration"].replace(0, 1)

    if "packets" in X.columns and "duration" in X.columns and X["packets"].dtype != object and X["duration"].dtype != object:
        X["packet_rate"] = X["packets"] / X["duration"]

    if "bytes" in X.columns and "duration" in X.columns and X["bytes"].dtype != object and X["duration"].dtype != object:
        X["byte_rate"] = X["bytes"] / X["duration"]

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    return X

# =========================
# Preprocessing Function
# =========================
def preprocess_input(input_df, selector, scaler, pca, train_min, train_max, qcnn_n_qubits, feature_columns):
    try:
        # Apply feature engineering
        df_fe = feature_engineering(input_df)

        # Align columns with training data before feature selection
        missing_cols = set(feature_columns) - set(df_fe.columns)
        for c in missing_cols:
            df_fe[c] = 0
        df_fe = df_fe[feature_columns] # Ensure the order of columns is the same

        df_fs = selector.transform(df_fe)

        # Scaling
        df_scaled = scaler.transform(df_fs)

        # PCA
        df_pca = pca.transform(df_scaled)

        # Quantum Scaling
        df_q_scaled = np.pi * (df_pca - train_min) / (train_max - train_min + 1e-8)
        df_q_scaled = np.clip(df_q_scaled, 0, np.pi)

        # Ensure the input dimensions match N_QUBITS for QCNN, and general input for CNN
        if df_q_scaled.shape[1] < qcnn_n_qubits:
            st.warning(f"Input features after PCA ({df_q_scaled.shape[1]}) are less than QCNN qubits ({qcnn_n_qubits}). Padding with zeros.")
            padded_data = np.zeros((df_q_scaled.shape[0], qcnn_n_qubits))
            padded_data[:, :df_q_scaled.shape[1]] = df_q_scaled
            df_q_scaled = padded_data
        elif df_q_scaled.shape[1] > qcnn_n_qubits:
            df_q_scaled = df_q_scaled[:, :qcnn_n_qubits]

        return torch.tensor(df_q_scaled, dtype=torch.float32), torch.tensor(df_pca, dtype=torch.float32)
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        st.stop()

# =========================
# Streamlit UI
# =========================
st.title("🛡️ Intrusion Detection System (IDS) - ML/QML Models")
st.markdown("Enter the file path to your CSV or Parquet file containing network traffic data for anomaly detection using classical and quantum machine learning models.")
st.markdown("**Note:** For large files (>2GB), it's recommended to place the file directly in the Colab environment and provide its path (e.g., `/content/your_data.parquet`).")

# Sidebar
st.sidebar.header("Configuration")
model_choice = st.sidebar.selectbox(
    "Choose a Model for Prediction:",
    ("Random Forest", "CNN", "QCNN")
)

# Load artifacts (cached)
rf_model, cnn_model, qcnn_model, scaler, pca, selector, train_min, train_max, qcnn_n_qubits, feature_columns = load_artifacts()

file_path = st.text_input("Enter file path (e.g., /content/data.parquet)", "/root/.cache/kagglehub/datasets/dhoogla/csecicids2018/versions/3/DDoS1-Tuesday-20-02-2018_TrafficForML_CICFlowMeter.parquet")

input_df = None
if file_path:
    if not os.path.exists(file_path):
        st.error(f"File not found at path: {file_path}")
    else:
        file_extension = os.path.splitext(file_path)[1].lower()

        try:
            if file_extension == ".csv":
                input_df = pd.read_csv(file_path)
            elif file_extension == ".parquet":
                input_df = pd.read_parquet(file_path)
            else:
                st.error("Unsupported file type. Please provide a path to a CSV or Parquet file.")
        except Exception as e:
            st.error(f"Error reading file: {e}")


if input_df is not None:
    st.subheader("Loaded Data Sample:")
    st.dataframe(input_df.head())

    if 'Label' in input_df.columns:
        st.warning("Removing 'Label' column from input data as it's the target for prediction.")
        input_df = input_df.drop(columns=['Label'])

    input_df_numeric = input_df.select_dtypes(include=[np.number])

    st.subheader("Preprocessed Data Sample for Prediction (after scaling and PCA):")

    q_input_tensor, cnn_input_tensor = preprocess_input(input_df_numeric, selector, scaler, pca, train_min, train_max, qcnn_n_qubits, feature_columns)

    st.dataframe(pd.DataFrame(q_input_tensor.numpy()).head())

    st.subheader("Prediction Results:")

    if st.button(f"Run {model_choice} Prediction"):
        with st.spinner(f'Making predictions with {model_choice}...'):
            results_df = input_df.copy()
            prediction_column_name = f'{model_choice}_Prediction'

            if model_choice == "Random Forest":
                df_fe = feature_engineering(input_df_numeric)
                missing_cols = set(feature_columns) - set(df_fe.columns)
                for c in missing_cols:
                    df_fe[c] = 0
                df_fe = df_fe[feature_columns]
                df_fs = selector.transform(df_fe)
                df_scaled = scaler.transform(df_fs)
                rf_preds = rf_model.predict(df_scaled)
                results_df[prediction_column_name] = rf_preds

            elif model_choice == "CNN":
                with torch.no_grad():
                    cnn_outputs = cnn_model(cnn_input_tensor)
                    cnn_preds = torch.argmax(cnn_outputs, dim=1).numpy()
                results_df[prediction_column_name] = cnn_preds

            elif model_choice == "QCNN":
                with torch.no_grad():
                    qcnn_outputs = qcnn_model(q_input_tensor)
                    qcnn_preds = torch.argmax(qcnn_outputs, dim=1).numpy()
                results_df[prediction_column_name] = qcnn_preds

            results_df['Prediction_Label'] = results_df[prediction_column_name].apply(lambda x: 'Benign' if x == 0 else 'Attack')

            st.success(f"{model_choice} Predictions Complete!")

            # === Prediction Summary ===
            st.write(f"### {model_choice} Prediction Summary")
            prediction_counts = results_df['Prediction_Label'].value_counts()
            total_predictions = len(results_df)

            if 'Benign' in prediction_counts:
                benign_count = prediction_counts['Benign']
                benign_percentage = (benign_count / total_predictions) * 100
                st.info(f"- **Benign Samples:** {benign_count} ({benign_percentage:.2f}%) 🟢")
            else:
                benign_count = 0
                st.info(f"- **Benign Samples:** 0 (0.00%) 🟢")

            if 'Attack' in prediction_counts:
                attack_count = prediction_counts['Attack']
                attack_percentage = (attack_count / total_predictions) * 100
                st.warning(f"- **Attack Samples:** {attack_count} ({attack_percentage:.2f}%) 🔴")
            else:
                attack_count = 0
                st.warning(f"- **Attack Samples:** 0 (0.00%) 🔴")

            st.write(f"- **Total Samples Predicted:** {total_predictions}")

            # === Visualizations ===
            st.write(f"## {model_choice} Prediction Distribution")

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x=prediction_counts.index, y=prediction_counts.values, ax=ax, palette=['green', 'red'])
            ax.set_title(f'Prediction Counts for {model_choice}')
            ax.set_xlabel('Prediction Label')
            ax.set_ylabel('Count')
            st.pyplot(fig)

            st.write("### Sample of Predictions:")
            st.dataframe(results_df.head())

            st.write("### All Predictions:")
            # Optionally display the full dataframe or allow download
            # st.dataframe(results_df)
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download all predictions as CSV",
                data=csv,
                file_name=f'{model_choice}_predictions.csv',
                mime='text/csv',
            )
