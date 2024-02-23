import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Mengatur konfigurasi halaman SEBELUM segala hal lain
st.set_page_config(page_title='Penyewaan Sepeda', layout='wide')

# Menampilkan judul dan deskripsi aplikasi
st.markdown("""
# Nama Anggota Kelompok :
- 1.10122386 - Raka Naufal Fauzan
- 2. 10122471 - Dhimas Muhammad Fattah Arrumy
- 3. 10122379 - Andhika Fachrul Mubarrok
- 4. 10122382 - Albert Cendra Hernawan
- 5. 10122371 - Yosia marpaung
- 6. 10122355 - Achmad Karis wibowo
# Informasi pekerjaan dari anggota :
- 1. Albert Cendra Hernawan, Melakukan Analisis dataset,membuat python, menghububgkan file py ke Streamlit, mengupload ke github agar dapat dilihat public.
- 2. Dhimas Muhammad Fattah Arrumy, membantu Albert Cendra Hernawan mengurus masalah pada saat mengupload ke github.
- 3. Raka Naufal Fauzan dan Andhika Fachrul Mubarrok, membantu Albert Cendra Hernawan menganalisa dataset serta mengupload ke Streamlit
- 4. Yosia marpaung dan Achmad Karis wibowo, membuat Code Jupyter setelah di analisis.

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

# Visualisasi Clustering dengan Matplotlib
fig, ax = plt.subplots()
for cluster in df['cluster'].unique():
    ax.scatter(df[df['cluster'] == cluster]['temp'], df[df['cluster'] == cluster]['casual'], label=f'Cluster {cluster}')
ax.set_title('Clustering Hasil Penyewaan Sepeda')
ax.set_xlabel('Temp')
ax.set_ylabel('Casual')
st.pyplot(fig)

# PCA Visualisasi
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)
fig, ax = plt.subplots()
scatter = ax.scatter(components[:, 0], components[:, 1], c=df['cluster'])
legend = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend)
ax.set_title('PCA: Visualisasi Data Multidimensi')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
st.pyplot(fig)

# Korelasi Fitur
if st.sidebar.checkbox('Tampilkan Heatmap Korelasi', False):
    corr_matrix = df[features + ['cnt']].corr()
    fig, ax = plt.subplots()
    cax = ax.matshow(corr_matrix, cmap='coolwarm')
    fig.colorbar(cax)
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    ax.set_title('Heatmap Korelasi Fitur dengan Jumlah Penyewaan')
    st.pyplot(fig)

# Rata-Rata Penyewaan Per Jam
if st.sidebar.checkbox('Tampilkan Rata-Rata Penyewaan Per Jam', False):
    fig, ax = plt.subplots()
    df.groupby('hr').mean().reset_index().plot(x='hr', y='cnt', ax=ax, legend=None)
    ax.set_title('Rata-Rata Penyewaan Sepeda Per Jam')
    ax.set_xlabel('Jam')
    ax.set_ylabel('Rata-Rata Penyewaan')
    st.pyplot(fig)
