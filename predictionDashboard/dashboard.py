import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Konfigurasi Halaman
st.set_page_config(page_title="IDS Prediction Dashboard", layout="wide")

st.title("🛡️ Cyber Attack Prediction Dashboard")
st.markdown("Unggah log Snort atau Suricata untuk mendeteksi teknik serangan berdasarkan MITRE ATT&CK.")

# Sidebar untuk Pilihan IDS
st.sidebar.header("Konfigurasi")
ids_type = st.sidebar.selectbox("Pilih Jenis IDS:", ["Snort", "Suricata"])

# Fungsi untuk Load Model dan Encoder
@st.cache_resource
def load_resources(ids):
    if ids == "Snort":
        model = joblib.load('snort.pkl')
        encoder = joblib.load('proto_snort.pkl')
    else:
        model = joblib.load('suricata.pkl')
        encoder = joblib.load('proto_suricata.pkl')
    return model, encoder

model, le_proto = load_resources(ids_type)

# File Uploader
uploaded_file = st.file_uploader(f"Unggah file CSV {ids_type}", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"File {uploaded_file.name} berhasil diunggah!")
    
    # --- PROSES PEMPROSESAN DATA (Feature Engineering) ---
    with st.spinner('Sedang memproses data...'):
        try:
            df_new = df.copy()

            if ids_type == "Snort":
                # Logika dari testing_snort.py
                df_new['timestamp'] = '2025-' + df_new['timestamp'].str.replace('/', '-')
                df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], format='%Y-%m-%d-%H:%M:%S.%f', errors='coerce')
                df_new = df_new.sort_values(by='timestamp')

                df_new['src_port'] = pd.to_numeric(df_new['src_port'], errors='coerce').fillna(0)
                df_new['dst_port'] = pd.to_numeric(df_new['dst_port'], errors='coerce').fillna(0)

                # Feature Engineering
                df_new['time_diff'] = df_new.groupby('src_ip')['timestamp'].diff().dt.total_seconds().fillna(0)
                df_new['pkt_rate_1s'] = df_new.groupby(['src_ip', df_new['timestamp'].dt.floor('1s')])['src_port'].transform('count')
                df_new['same_dst_port'] = df_new.groupby('src_ip')['dst_port'].diff().fillna(1).eq(0).astype(int)
                df_new['diff_src_port'] = df_new.groupby('src_ip')['src_port'].diff().fillna(1).ne(0).astype(int)
                
                # Encode & Drop
                df_new['protocol'] = le_proto.transform(df_new['protocol'].astype(str))
                features_to_drop = ['timestamp', 'src_ip', 'dst_ip', 'mitre', 'pkt_num', 'action', 'msg', 'label']

            else:
                # Logika dari testing_suricata.py
                df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], errors='coerce')
                df_new = df_new.sort_values(by='timestamp')

                df_new['src_port'] = pd.to_numeric(df_new['src_port'], errors='coerce')
                df_new['dst_port'] = pd.to_numeric(df_new['dst_port'], errors='coerce')

                # Feature Engineering
                df_new['time_diff'] = df_new.groupby('src_ip')['timestamp'].diff().dt.total_seconds().fillna(0)
                df_new['pkt_rate_1s'] = df_new.groupby(['src_ip', df_new['timestamp'].dt.floor('1s')])['src_port'].transform('count')
                df_new['same_dst_port'] = (df_new.groupby('src_ip')['dst_port'].diff() == 0).astype(int)
                df_new['diff_src_port'] = (df_new.groupby('src_ip')['src_port'].diff() != 0).astype(int)

                # Encode & Drop
                df_new['protocol'] = le_proto.transform(df_new['protocol'].astype(str))
                features_to_drop = ['timestamp', 'src_ip', 'dst_ip', 'gid', 'sid', 'rev', 'message', 'classification', 'priority', 'mitre']

            # Penyiapan fitur untuk Prediksi
            X_new = df_new.drop(columns=features_to_drop, errors='ignore')
            X_new = X_new.fillna(0)

            # Prediksi
            df_new['predicted_mitre'] = model.predict(X_new)

            # --- TAMPILAN DASHBOARD ---
            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("Ringkasan Prediksi")
                prediction_counts = df_new['predicted_mitre'].value_counts()
                st.write(prediction_counts)
                
                # Download Button
                csv_result = df_new.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Hasil Prediksi (CSV)",
                    data=csv_result,
                    file_name=f"hasil_prediksi_{ids_type.lower()}.csv",
                    mime='text/csv',
                )

            with col2:
                st.subheader("Visualisasi")
                st.bar_chart(prediction_counts)

            st.divider()
            st.subheader("Data Hasil Prediksi (Preview)")
            st.dataframe(df_new, use_container_width=True)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses data: {e}")
            st.info("Pastikan kolom pada file CSV sesuai dengan standar log IDS yang dipilih.")
