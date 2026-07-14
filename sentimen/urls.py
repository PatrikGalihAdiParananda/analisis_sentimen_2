from django.urls import path
from django.http import JsonResponse  # <-- Tambahkan import ini
from . import views

urlpatterns = [
    # Halaman Dashboard (Halaman Utama)
    path('', views.dashboard, name='dashboard'),
    
    # Halaman Evaluasi & Visualisasi
    path('evaluasi-model/', views.evaluasi_model_view, name='evaluasi_model'),
    path('visualisasi-lanjutan/', views.visualisasi_lanjutan_view, name='visualisasi_lanjutan'),
    
    # Halaman Analisis Realtime
    path('analisis-realtime/', views.analisis_realtime_view, name='analisis_realtime'),
    
    # API Backend untuk melakukan prediksi (Digunakan oleh AJAX/Javascript)
    path('klasifikasi/', views.klasifikasi_sentimen, name='klasifikasi_sentimen'),
    
    # Halaman Lainnya
    path('tentang-proyek/', views.tentang_proyek_view, name='tentang_proyek'),
    
    # Placeholder untuk training (Mencegah error jika tombol diklik)
    path('train-model/', views.train_model, name='train_model'),

    # Membungkam peringatan 404 dari Google Chrome DevTools
    path('.well-known/appspecific/com.chrome.devtools.json', lambda r: JsonResponse({})),  # <-- Tambahkan baris ini
]