# File: sentimen/views.py

from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse
# Mengimpor 'otak' sistem dari file controllers.py yang sudah dibuat sebelumnya
from .controllers import DatasetManager, KlasifikasiController, DashboardController, VisualisasiController

# ==============================================================================
# INISIALISASI GLOBAL (Server Startup)
# ==============================================================================
# Baris ini sangat krusial! Memuat semua dataset dan model ML (file .joblib & .pkl) 
# HANYA SATU KALI saat server Django pertama kali di-run. 
# Ini mencegah server membuang waktu memuat file berulang kali setiap kali ada user yang mengakses web.
DatasetManager.load_all_assets()
# ==============================================================================
# VIEWS: HALAMAN UTAMA & DASHBOARD
# ==============================================================================
def dashboard(request):
    """
    View untuk halaman utama (Dashboard).
    Tugas komputasi angka diserahkan penuh ke DashboardController.
    """
    context = DashboardController.get_dashboard_context()
    return render(request, 'sentimen/dashboard.html', context)

def evaluasi_model_view(request):
    """
    View untuk halaman detail Evaluasi Model (Confusion Matrix & Classification Report).
    """
    context = DashboardController.get_evaluasi_context()
    return render(request, 'sentimen/evaluasi_model.html', context)

def visualisasi_lanjutan_view(request):
    """
    View untuk halaman Grafik dan Wordcloud.
    """
    context = VisualisasiController.get_visualisasi_context()
    # Pengaman: Jika file JSON evaluasi tidak ditemukan/kosong, tampilkan pesan peringatan di web
    if not context['data_aman_untuk_visualisasi']:
        messages.warning(request, "Data visualisasi tidak ditemukan.")
    return render(request, 'sentimen/visualisasi_lanjutan.html', context)
    """
    View untuk memuat antarmuka (UI) halaman pengujian teks secara real-time.
    Tidak ada konteks yang dikirim karena data akan dimuat via AJAX (Javascript).
    """
def analisis_realtime_view(request):
    return render(request, 'sentimen/analisis_realtime.html')
# ==============================================================================
# VIEWS: API ENDPOINT (Pemrosesan Data)
# ==============================================================================
def klasifikasi_sentimen(request):
    """""
    API Endpoint (AJAX). View ini menerima ketikan user dari halaman 'analisis_realtime',
    menyerahkannya ke KlasifikasiController untuk ditebak sentimennya, lalu 
    mengembalikan hasilnya dalam format JSON agar bisa ditangkap oleh Javascript.
    """""
    # 1. Validasi Metode HTTP (Hanya menerima POST untuk keamanan data)
    if request.method == 'POST':
        text_mentah = request.POST.get('teks_ulasan', '')
        # Pengaman: Tolak jika user mengirim teks kosong atau hanya spasi
        if not text_mentah.strip():
            return JsonResponse({'error': 'Teks tidak boleh kosong'}, status=400)
        
        try: # 2. Validasi Kesiapan Model
            if DatasetManager.model is None or DatasetManager.vectorizer is None:
                 return JsonResponse({'error': 'Model belum dimuat. Hubungi admin.'}, status=500)

            # 3. Eksekusi Prediksi (Memanggil Controller)
            # Seluruh logika pembersihan teks (NLP) dan prediksi diselesaikan di satu baris ini
            hasil = KlasifikasiController.predict(text_mentah)
            # 4. Menyusun Respon JSON
            # Mengemas hasil prediksi beserta tahapan NLP-nya untuk dikirim kembali ke browser user
            return JsonResponse({
                'original_text': text_mentah,
                'processed_steps': {
                    'normalisasi': hasil['normalisasi'],
                    'stopwords': hasil['stopwords'],
                    'stemming': hasil['stemming']
                },
                'sentiment': hasil['sentiment'],
                'confidence': hasil['confidence'],
                'trigger_words': hasil['trigger_words'],
                'similar_reviews': hasil['similar_reviews']
            })

        except Exception as e: # Menangkap error tak terduga (misal server down atau tipe data salah)
            print(f"Error di klasifikasi_sentimen: {e}")
            return JsonResponse({'error': f'Terjadi kesalahan: {str(e)}'}, status=500)
            # Jika user mencoba mengakses URL ini lewat browser secara langsung (metode GET), tolak aksesnya
    return JsonResponse({'error': 'Metode tidak diizinkan'}, status=405)

def tentang_proyek_view(request):
    return render(request, 'sentimen/tentang_proyek.html')

def train_model(request):
    messages.info(request, 'Fitur training via web dinonaktifkan untuk keamanan.')
    return redirect('dashboard')
"""
    Tombol/URL keamanan. Mencegah user biasa untuk men-trigger pelatihan ulang model 
    (yang bisa memakan RAM server sangat besar) melalui web interface.
    """