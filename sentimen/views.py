# File: sentimen/views.py

from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse
from .controllers import DatasetManager, KlasifikasiController, DashboardController, VisualisasiController

# Muat semua dataset dan model ML secara global hanya sekali saat server Django berjalan
DatasetManager.load_all_assets()

def dashboard(request):
    """View ini meminta data ke DashboardController."""
    context = DashboardController.get_dashboard_context()
    return render(request, 'sentimen/dashboard.html', context)

def evaluasi_model_view(request):
    """View ini meminta data komputasi ke DashboardController."""
    context = DashboardController.get_evaluasi_context()
    return render(request, 'sentimen/evaluasi_model.html', context)

def visualisasi_lanjutan_view(request):
    """View ini meminta konfigurasi grafik ke VisualisasiController."""
    context = VisualisasiController.get_visualisasi_context()
    if not context['data_aman_untuk_visualisasi']:
        messages.warning(request, "Data visualisasi tidak ditemukan.")
    return render(request, 'sentimen/visualisasi_lanjutan.html', context)
    
def analisis_realtime_view(request):
    return render(request, 'sentimen/analisis_realtime.html')

def klasifikasi_sentimen(request):
    """View ini menyerahkan input teks user ke KlasifikasiController untuk diprediksi."""
    if request.method == 'POST':
        text_mentah = request.POST.get('teks_ulasan', '')
        if not text_mentah.strip():
            return JsonResponse({'error': 'Teks tidak boleh kosong'}, status=400)
        
        try:
            if DatasetManager.model is None or DatasetManager.vectorizer is None:
                 return JsonResponse({'error': 'Model belum dimuat. Hubungi admin.'}, status=500)

            # --- OOP IMPLEMENTATION ---
            # Pemanggilan Method dari Class KlasifikasiController sesuai Sequence Diagram
            hasil = KlasifikasiController.predict(text_mentah)

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

        except Exception as e:
            print(f"Error di klasifikasi_sentimen: {e}")
            return JsonResponse({'error': f'Terjadi kesalahan: {str(e)}'}, status=500)
            
    return JsonResponse({'error': 'Metode tidak diizinkan'}, status=405)

def tentang_proyek_view(request):
    return render(request, 'sentimen/tentang_proyek.html')

def train_model(request):
    messages.info(request, 'Fitur training via web dinonaktifkan untuk keamanan.')
    return redirect('dashboard')