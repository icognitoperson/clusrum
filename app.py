from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import logging
import os

app = Flask(__name__)

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model dan scaler
try:
    # Sesuaikan path model sesuai dengan lokasi penyimpanan
    model_path = 'model_fcm.pkl'  # Ganti dengan nama file model Anda
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info("Model berhasil dimuat")

    # Inisialisasi scaler
    scaler = StandardScaler()
    logger.info("Scaler berhasil diinisialisasi")
except Exception as e:
    logger.error(f"Error saat memuat model: {str(e)}")


def predict_cluster(features):
    try:
        # Ubah input menjadi array numpy
        features_array = np.array(features).reshape(1, -1)

        # Lakukan scaling pada fitur
        features_scaled = scaler.fit_transform(features_array)

        # Prediksi menggunakan model Fuzzy C-Means
        # Transpose karena FCM membutuhkan format yang berbeda
        features_scaled_T = features_scaled.T

        # Hitung membership menggunakan centroids dari model
        # Ambil model dengan 3 cluster (sesuaikan dengan kebutuhan)
        cntr = model[3]['cntr']  # centroids untuk 3 cluster

        # Hitung jarak ke setiap centroid
        d = np.zeros((len(cntr), 1))
        for i in range(len(cntr)):
            d[i] = np.linalg.norm(features_scaled_T -
                                  cntr[i].reshape(-1, 1), axis=0)

        # Tentukan cluster berdasarkan jarak terdekat
        cluster = np.argmin(d)

        # Mapping cluster ke label
        cluster_labels = {
            0: "Rumah Kecil",
            1: "Rumah Menengah",
            2: "Rumah Mewah"
        }

        return cluster_labels[cluster]
    except Exception as e:
        logger.error(f"Error dalam prediksi: {str(e)}")
        return None


@app.route('/')
def index():
    return render_template('index.html', prediction=None, error=None)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            # Ambil data dari form
            features = [
                float(request.form['luas_bangunan']),
                float(request.form['luas_tanah']),
                float(request.form['kamar_tidur']),
                float(request.form['kamar_mandi']),
                float(request.form['garasi']),
                float(request.form['harga'])
            ]

            # Lakukan prediksi
            prediction = predict_cluster(features)

            if prediction:
                return render_template('index.html', prediction=prediction, error=None)
            else:
                return render_template('index.html', prediction=None, error='Terjadi kesalahan dalam melakukan prediksi')
    except ValueError:
        return render_template('index.html', prediction=None, error='Mohon masukkan angka yang valid')
    except Exception as e:
        logger.error(f"Error dalam route predict: {str(e)}")
        return render_template('index.html', prediction=None, error='Terjadi kesalahan sistem')


@app.route('/contact', methods=['POST'])
def contact():
    try:
        if request.method == 'POST':
            # Ambil data dari form kontak
            name = request.form.get('name')
            email = request.form.get('email')
            message = request.form.get('message')

            # Di sini Anda bisa menambahkan logika untuk menyimpan pesan kontak
            # Misalnya menyimpan ke database atau mengirim email

            return render_template('index.html', prediction=None, error=None, message='Pesan Anda telah terkirim!')
    except Exception as e:
        logger.error(f"Error dalam route contact: {str(e)}")
        return render_template('index.html', prediction=None, error='Terjadi kesalahan saat mengirim pesan')


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
