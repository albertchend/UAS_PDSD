# streamlit_app.py
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px

# Mengatur konfigurasi halaman SEBELUM segala hal lain
st.set_page_config(page_title='Penyewaan Sepeda', layout='wide')

# Menampilkan judul dan deskripsi aplikasi
st.markdown("""
# Analisis dan Visualisasi Clustering Penyewaan Sepeda

Aplikasi ini dibuat untuk mengidentifikasi pola dalam data penyewaan sepeda dan mengelompokkan data berdasarkan karakteristik penyewaan menggunakan metode clustering.

## Fitur Aplikasi:

- **Load Data**: Memuat data penyewaan sepeda dari file CSV.
- **Preprocessing Data**: Standarisasi fitur numerik untuk persiapan analisis.
- **Clustering dengan KMeans**: Mengelompokkan data menjadi beberapa cluster berdasarkan fitur yang dipilih.
- **Visualisasi Clustering**: Menampilkan hasil clustering menggunakan scatter plot.
- **Analisis Komponen Utama (PCA)**: Reduksi dimensi dan visualisasi data multidimensi.
- **Analisis Korelasi**: Menghitung dan menampilkan heatmap korelasi fitur dengan jumlah penyewaan.
- **Rata-Rata Penyewaan Per Jam**: Visualisasi rata-rata penyewaan sepeda per jam.

Scroll ke bawah untuk melihat analisis dan visualisasi data!
""", unsafe_allow_html=True)

# Sidebar: Pengaturan Analisis
st.sidebar.header('Pengaturan')
n_clusters = st.sidebar.slider('Jumlah Cluster', min_value=2, max_value=10, value=3)

# Load dan Tampilkan Data
df = pd.read_csv('hour.csv')
df['dteday'] = pd.to_datetime(df['dteday'])

# Tampilkan snapshot data
if st.sidebar.checkbox('Tampilkan Snapshot Data', False):
    st.write(df.head())

# Preprocessing Data
features = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered']
X = df[features]
X_scaled = StandardScaler().fit_transform(X)

# Clustering dengan KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Visualisasi Clustering dengan Plotly
fig = px.scatter(df, x='temp', y='casual', color='cluster', title='Clustering Hasil Penyewaan Sepeda', template='plotly_white')
st.plotly_chart(fig)

# PCA Visualisasi
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)
fig_pca = px.scatter(components, x=0, y=1, color=df['cluster'], title='PCA: Visualisasi Data Multidimensi', template='plotly_white')
fig_pca.update_layout(xaxis_title='PC1', yaxis_title='PC2')
st.plotly_chart(fig_pca)

# Korelasi Fitur
if st.sidebar.checkbox('Tampilkan Heatmap Korelasi', False):
    corr_matrix = df[features + ['cnt']].corr()
    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", title='Heatmap Korelasi Fitur dengan Jumlah Penyewaan')
    st.plotly_chart(fig_corr)

# Rata-Rata Penyewaan Per Jam
if st.sidebar.checkbox('Tampilkan Rata-Rata Penyewaan Per Jam', False):
    fig_hourly = px.line(df.groupby('hr').mean().reset_index(), x='hr', y='cnt', title='Rata-Rata Penyewaan Sepeda Per Jam', template='plotly_white')
    fig_hourly.update_xaxes(title_text='Jam')
    fig_hourly.update_yaxes(title_text='Rata-Rata Penyewaan')
    st.plotly_chart(fig_hourly)
