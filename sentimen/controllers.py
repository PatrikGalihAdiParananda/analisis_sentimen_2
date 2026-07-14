# File: sentimen/controllers.py

import os
import re
import json
import copy
import pandas as pd
import joblib
from django.conf import settings
from sklearn.metrics.pairwise import cosine_similarity

# Inisialisasi Sastrawi dan NLTK
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from nltk.corpus import stopwords
    import nltk
    try:
        stopwords_indonesia = stopwords.words('indonesian')
    except LookupError:
        nltk.download('stopwords')
        stopwords_indonesia = stopwords.words('indonesian')

    # Filter stopwords agar tidak menghapus kata sentimen penting
    kata_penting = {
        'tidak', 'kurang', 'belum', 'bukan', 'jangan', 
        'sangat', 'paling', 'agak', 'lebih', 'banget', 
        'selalu', 'amat', 'sungguh', 'terus', 'makin', 
        'banyak', 'beberapa', 'kali', 'segera', 'benar','dulu','yang','mau',
        'mesti','harus','semua','semuanya','semua nya','semua nya','semua nya','semua nya','bisa',
        
        'biasa', 'saja', 'lumayan', 'cukup', 'tapi', 
        'padahal', 'namun', 'walaupun', 'meskipun', 'meski',
        'cuma', 'hanya', 'sayang', 'sayangnya', 'malah',
        'karena', 'kalau', 'kenapa', 'bagaimana', 'sebenarnya',
        'bukannya', 'tetapi', 'kontradiksi',
        
        'bagus', 'baik', 'mantap', 'mudah', 'gampang', 'cepat', 
        'keren', 'oke', 'puas', 'terbaik', 'membantu', 
        'tertolong', 'lancar', 'aman', 'nyaman', 'jelas', 
        'lengkap', 'responsif', 'berguna', 'bermanfaat', 'solusi',
        'terima kasih', 'salam',
        
        'jelek', 'buruk', 'parah', 'malas', 'kecewa', 'hancur', 
        'payah', 'kacau', 'ribet', 'repot', 'bingung', 
        'membingungkan', 'kesal', 'marah', 'emosi', 'nyesel', 
        'menyesal', 'bodoh', 'goblok', 'sampah', 'tolol', 
        'aneh', 'bohong', 'penipu', 'busuk', 'bobrok', 'rusak',
        'muak', 'bosan', 'pusing', 'biadab', 'kotoran', 'sialan',
        'kentut', 'mengeluh', 'miskin', 'korupsi', 'becus', 'ambyar','gajelas',
        
        'error', 'bug', 'lambat', 'lemot', 'lelet', 'sulit', 
        'susah', 'rumit', 'memperumit', 'lama', 'gagal', 'macet', 
        'muter', 'loading', 'mental', 'blank', 'gangguan', 
        'sibuk', 'nyangkut', 'putus', 'antre', 'antri', 
        'penuh', 'ditolak', 'habis', 'crash', 'lag', 'offline', 'terkoneksi',
        
        'faskes', 'fasilitas kesehatan', 'rujukan', 'kis', 'bpjs', 
        'jkn', 'kesehatan', 'pasien', 'otp', 'login', 'masuk', 
        'keluar', 'daftar', 'pendaftaran', 'sandi', 'password', 
        'verifikasi', 'captcha', 'pindah', 'kelas', 'iuran', 'denda', 
        'tagihan', 'nunggak', 'premi', 'klaim', 'rs', 'puskesmas', 
        'klinik', 'dokter', 'gigi', 'kacamata', 'operasi', 'sesar', 
        'melahirkan', 'ktp', 'nik', 'kk', 'kartu', 'kartu digital', 
        'kartu fisik', 'cetak', 'antrean', 'jadwal', 'poli', 'riwayat',
        'nomor ponsel', 'nomor peserta', 'nomor', 'autodebit', 'notifikasi',
        'komentar', 'customer service', 'admin', 'operator', 'aplikasi', 
        'playstore', 'google', 'pembaruan','pengiriman',
        
        'perbaiki', 'tingkatkan', 'benahi', 'urus', 'bantu', 
        'tolong', 'minta', 'mohon', 'kembalikan', 'hapus', 'ganti', 
        'copot', 'uninstal', 'uninstall', 'update', 'perbarui',
        'buat', 'menghabiskan', 'memasukkan', 'membuat', 'coba', 
        'silakan', 'buka', 'diklik', 'muncul', 'menunggu', 
        'bolak-balik', 'respons', 'lihat', 'didaftarkan', 'dipakai',
        'sesuai', 'harapan', 'dijanjikan', 'perbaikan',
        
        
        'mending', 'mendingan', 'percuma', 'ketimbang', 'daripada', 
        'sia', 'sia-sia', 'katanya', 'masa', 'giliran', 
        'tertawa', 'sedih', 'astaga', 'naudzubillah', 'umpatan', 'ekspresi'
    }
    stopwords_indonesia = [word for word in stopwords_indonesia if word not in kata_penting]

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
except ImportError:
    stemmer = None; stopwords_indonesia = []

REPLACEMENTS_DICT = {
    'yg': 'yang', 'ga': 'tidak', 'gak': 'tidak', 'gk': 'tidak', 'enggak': 'tidak', 'nggak': 'tidak', 'ndak': 'tidak', 'ndk': 'tidak', 'gaada': 'tidak ada',
    'tdk': 'tidak', 'bkn': 'bukan', 'utk': 'untuk', 'dgn': 'dengan', 'krn': 'karena', 'karna': 'karena',
    'bgt': 'banget', 'jg': 'juga', 'aja': 'saja', 'jd': 'jadi', 'jgn': 'jangan', 'aza': 'saja',
    'klo': 'kalau', 'kalo': 'kalau', 'kl': 'kalau', 'knp': 'kenapa', 'gmn': 'bagaimana', 'gimana': 'bagaimana', 'bgmn': 'bagaimana', 'ginana': 'bagaimana',
    'sdh': 'sudah', 'udh': 'sudah', 'blm': 'belum', 'balum': 'belum', 'sblm': 'sebelum', 'blom': 'belum',
    'skrg': 'sekarang', 'skrng': 'sekarang', 'bbrp': 'beberapa', 'brp': 'berapa', 'byk': 'banyak', 'bnyk': 'banyak',
    'tp': 'tapi', 'tpi': 'tapi', 'pdhl': 'padahal','bz':'bisa',
    'trus': 'terus', 'trs': 'terus', 'bngt': 'banget', 'dpt': 'dapat', 'dr': 'dari',
    'kpd': 'kepada', 'kyk': 'seperti', 'ky': 'seperti', 'nih': 'ini', 'sih': 'sih', 'tuh': 'itu', 'ni': 'ini',
    'dong': 'dong', 'kok': 'kok', 'pake': 'pakai', 'dipake': 'dipakai', 'dipke': 'dipakai', 'dipakeitx': 'dipakai',
    'sm': 'sama', 'ama': 'sama', 'cmn': 'cuma',
    'kpn': 'kapan', 'tlg': 'tolong', 'tlong': 'tolong', 'mnt': 'minta', 'gpp': 'tidak apa-apa', 'donk': 'dong',
    'emg': 'memang', 'sbnrnya': 'sebenarnya', 'tq': 'terima kasih', 'yth': 'yang terhormat', 'sgt': 'sangat', 'stlh': 'setelah',
    'lbih': 'lebih', 'bner': 'benar', 'sgera': 'segera', 'slm': 'salam',
    'itx': 'itu', 'bknnya': 'bukannya', 'tmbh': 'tambah', 'mlh': 'malah',
    'pas': 'saat', 'kmrin': 'kemarin', 'i': 'ini', 'liat': 'lihat', 'libat': 'lihat',
    'masiah': 'masih', 'hancur': 'hancur', 'ancur': 'hancur', 'boro boro': 'boro-boro', 'boro2': 'boro-boro', 'gajelas': 'tidak jelas',
    'notif': 'notifikasi', 'didaftarin': 'didaftarkan', 'bwat': 'buat', 'komen': 'komentar', 'ngabisin': 'menghabiskan',
    'mala': 'malah', 'bredel': 'rusak', 'disaat': 'disaat', 'genting': 'genting', 'masukin': 'memasukkan', 'kocak': '[tertawa]',
    'bikin': 'membuat', 'muak': 'muak', 'sia': 'sia-sia', 'bosen': 'bosan',
    'aku': 'saya', 'aq': 'saya', 'gw': 'saya', 'gua': 'saya', 'gue': 'saya', 'w': 'saya', 'sy': 'saya', 'sya': 'saya',
    'kamu': 'kamu', 'lo': 'kamu', 'loe': 'kamu', 'lu': 'kamu', 'org': 'orang', 'kau': 'kamu',
    'x': 'kali', 'pe x': 'berkali kali', 'y': 'ya', 'maww': 'mau', 'mw': 'mau', 'af': 'maaf', 'maap': 'maaf', 'maunya': 'maunya',
    'wkwk': '[tertawa]', 'wkwkwk': '[tertawa]', 'hahaha': '[tertawa]', 'hehe': '[tertawa]', 'xixi': '[tertawa]',
    'hiks': '[sedih]', 'huhu': '[sedih]',
    'astaga': 'astaga', 'astaghfirullah': 'astaga', 'naudzubillah': 'naudzubillah',
    'anjim': '[umpatan]', 'anjir': '[umpatan]', 'anjay': 'umpatan', 'njir': '[umpatan]', 'jir': '[umpatan]','k0nt0l': '[umpatan]',
    '4nj1ng': '[umpatan]','4njg': '[umpatan]','b4b1': '[umpatan]','t0l0l': '[umpatan]','gblg': '[umpatan]','mmk': '[umpatan]','ngntd': '[umpatan]','asw': '[umpatan]',
    't41': '[umpatan]','itil': '[umpatan]','puki': '[umpatan]','pukimak': '[umpatan]','pukymak': '[umpatan]','pepeq': '[umpatan]',
    'peler': '[umpatan]','pler': '[umpatan]','lonte': '[umpatan]','perek': '[umpatan]','ndasmu': '[umpatan]',
    'raimu': '[umpatan]','gateli': '[umpatan]','modar': '[umpatan]','sempak': '[umpatan]','bagong': '[umpatan]','bujanginam': '[umpatan]','bujangam': '[umpatan]','pantek': '[umpatan]','panteq': '[umpatan]','pepe': '[umpatan]',
    'opo': 'apa', 'iki': 'ini', 'iku': 'itu', 'kui': 'itu', 'piye': 'bagaimana', 'kepiye': 'bagaimana',
    'angel': 'sulit', 'abot': 'berat', 'rudet': 'rumit', 'suwe': 'lama', 'suwi': 'lama', 'seabad': 'lama sekali', 'lila': 'lama',
    'meneh': 'lagi', 'wae': 'saja', 'ae': 'saja', 'jan': 'sungguh', 'tenan': 'benar', 'tenan': 'sungguh',
    'ora': 'tidak', 'ra': 'tidak', 'raiso': 'tidak bisa', 'oraiso': 'tidak bisa', 'g bisa': 'tidak bisa', 'gak bisa': 'tidak bisa', 'gabisa': 'tidak bisa',
    'kabeh': 'semua', 'mbuh': 'tidak tahu', 'mumet': 'pusing', 'nang': 'di', 'neng': 'di',
    'jancok': '[umpatan]', 'matamu': '[umpatan]', 'cok': '[umpatan]', 'cuk': '[umpatan]', 'nanaon': 'apa-apaan', 'hese': 'sulit', 'kie': 'begini', 'ge': 'juga',
    'opo o': 'apa apa', 'wes': 'sudah', 'isoh': 'bisa', 'iso': 'bisa', 'bsa': 'bisa', 'bari jeung': 'tetapi', 'teu': 'tidak', 'cobaen': 'coba',
    'monggo': 'silakan', 'monggo': 'silakan', 'matur nuwun': 'terima kasih', 'suwun': 'terima kasih', 'asline': 'aslinya', 'nggwe': 'buat',
    'disek': 'dulu', 'sakdurunge': 'sebelumnya', 'bukaen': 'buka', 'koyok': 'seperti',
    'dongo': 'bodoh', 'dungu': 'bodoh', 'bego': 'bodoh', 'goblok': 'bodoh', 'gblk': 'bodoh', 'tolol': 'bodoh', 'tlol': 'tolol', 'dongok': 'bodoh',
    'anjing': '[umpatan]', 'anjiing': '[umpatan]', 'asu': '[umpatan]', 'hancok': '[umpatan]', 'anjeng': '[umpatan]',
    'bangsat': '[umpatan]', 'bgst': '[umpatan]', 'biadab': 'biadab','dlu': 'dulu', 'dlu': 'dulu', 'dlu': 'dulu',
    'kontol': '[umpatan]', 'memek': '[umpatan]', 'jembut': '[umpatan]', 'kntl': '[umpatan]',
    'tai': 'kotoran', 'taik': 'kotoran', 'taek': 'kotoran', 'kampret': '[umpatan]', 'sial': 'sialan', 'najis': '[umpatan]','njing': '[umpatan]','4njing': '[umpatan]','anyink':'[umpatan]','anjrit': '[umpatan]','anjrot': '[umpatan]','bangst': '[umpatan]',
    'bgsat': '[umpatan]','bgsd': '[umpatan]','keparat': '[umpatan]','brengsek': '[umpatan]','goblog': '[umpatan]',
    'babi': '[umpatan]', 'binatang': '[umpatan]', 'bajingan': '[umpatan]', 'bangke': '[umpatan]','sampah': '[sampah]', 'bobrok': 'bobrok', 'kotoran': '[kotoran]', 'buwosok': 'busuk', 'bosok': 'busuk',
    'ngehek': '[umpatan]', 'entut': 'kentut', 'mungak': 'muak',
    'apk': 'aplikasi', 'app': 'aplikasi', 'apps': 'aplikasi', 'apliksi': 'aplikasi', 'uplikasinya': 'aplikasinya', 'aplikasinya': 'aplikasinya',
    'eror': 'error', 'erorr': 'error', 'erorrr': 'error', 'ngebug': 'bug',
    'lemot': 'lambat', 'lelet': 'lambat', 'loding': 'loading',
    'update': 'pembaruan', 'updet': 'pembaruan', 'diupdate': 'diperbarui', 'pembaruan': 'pembaruan', 'diperbaharui': 'diperbarui', 'abdet': 'pembaruan',
    'login': 'masuk', 'log in': 'masuk', 'logout': 'keluar', 'log out': 'keluar', 'msuk': 'masuk',
    'register': 'daftar', 'registrasi': 'pendaftaran', 'daptar': 'daftar', 'pndftran': 'pendaftaran', 'didaptar': 'didaftar', 'daftarin': 'daftarkan',
    'verif': 'verifikasi', 'ferivikasi': 'verifikasi', 'perivikasi': 'verifikasi', 'verifikasi': 'verifikasi', 'verivikasi': 'verifikasi', 'blel': 'gagal', 'captcha': 'captcha', 'captca': 'captcha',
    'faskes': 'fasilitas kesehatan', 'faskesnya': 'fasilitas kesehatannya',
    'bpjs': 'bpjs', 'jkn': 'jkn', 'kesehatan': 'kesehatan', 'pasien': 'pasien',
    'cs': 'customer service', 'admin': 'admin', 'operator': 'operator',
    'kartu digital': 'kartu digital', 'kartu fisik': 'kartu fisik', 'kk': 'kartu keluarga', 'ktp': 'kartu tanda penduduk', 'kluarga': 'keluarga',
    'tagihan': 'tagihan', 'iuran': 'iuran', 'premi': 'premi',
    'no hp': 'nomor ponsel', 'nomor hp': 'nomor ponsel', 'no peserta': 'nomor peserta', 'nomer': 'nomor',
    'password': 'password', 'pasword': 'password', 'pssword': 'password', 'asword': 'password',
    'cesar': 'sesar', 'operasi': 'operasi', 'autodebit': 'autodebit', 'nik': 'nik',
    'diklik': 'diklik', 'diklik': 'diklik', 'muncul': 'muncul', 'nungguan': 'menunggu', 'oflline': 'offline', 'offline': 'offline',
    'bulak balek': 'bolak-balik', 'bulak bali': 'bolak-balik',
    'bagusss': 'bagus', 'baguss': 'bagus', 'bgs': 'bagus', 'good': 'bagus', 'gud': 'bagus',
    'jelek': 'jelek', 'jlek': 'jelek', 'jellek': 'jelek', 'jelex': 'jelek', 'bapuk': 'jelek',
    'susah': 'sulit', 'ribet': 'rumit', 'gampang': 'mudah',
    'mantap': 'mantap', 'mntp': 'mantap', 'mantul': 'mantap betul',
    'ok': 'oke', 'okeey': 'oke',
    'makasih': 'terima kasih', 'mksh': 'terima kasih', 'thx': 'terima kasih', 'terimakasih': 'terima kasih',
    'tolongg': 'tolong', 'plis': 'tolong', 'pls': 'tolong',
    'respon': 'respons', 'responnya': 'responsnya',
    'bangettt': 'banget', 'bangett': 'banget', 'bgtt': 'banget',
    'parah': 'parah', 'males': 'malas', 'mager': 'malas gerak',
    'kecewa': 'kecewa', 'kcewa': 'kecewa',
    'bintang': 'bintang', 'bntg': 'bintang', 'nol': 'nol',
    'fungsi': 'fungsi', 'fungsik': 'fungsi', 'guna': 'guna',
    'nyampe': 'sampai', 'sampe': 'sampai', 'dikirim': 'dikirim', 'balesan': 'balasan',
    'jt': 'juta', 'rb': 'ribu', 'th': 'tahun', 'bln': 'bulan', 'wktu': 'waktu', 'ttap': 'tetap',
    'ketrangannya': 'keterangannya', 'pnjelasanya': 'penjelasannya', 'mngeluh': 'mengeluh', 'tambahkn': 'tambahkan', 'angak': 'angka',
    'kedepannya': 'kedepannya', 'dihapus': 'dihapus', 'playstore': 'playstore', 'google': 'google', 'macam': 'seperti',
    'terpakse': 'terpaksa', 'selanjutnya': 'selanjutnya', 'giliran': 'giliran', 'aktif': 'aktif', 'sayang': 'sayang', 'nya': 'nya', 'mikir': 'pikir',
    'miskin': 'miskin', 'rakyat': 'rakyat', 'gaji': 'gaji', 'naik': 'naik', 'boom': '[ekspresi]',
    'korup': 'korupsi', 'mulu': 'terus', 'idih': '[ekspresi]', 'sms': 'sms', 'pulsa': 'pulsa', 'min': 'admin',
    'becus': 'becus', 'klen': 'kalian', 'adeuh': '[ekspresi]', 'ambyar': 'hancur'
}
NORMALIZATION_REGEX = re.compile(r'\b(' + '|'.join(re.escape(key) for key in REPLACEMENTS_DICT.keys()) + r')\b')


class DatasetManager:
    """
    Class terpusat untuk memanajemen aset data, model Random Forest, dan hasil evaluasi.
    Menggunakan desain 'Class Variables' agar aset bertindak sebagai Cache di memori server.
    """
    df_preview = pd.DataFrame()
    model = None
    vectorizer = None
    EVALUATION_RESULTS = {}
    
    @classmethod
    def load_all_assets(cls):
        try:# 1. Menyiapkan Path (Lokasi File)
            # os.path.join memastikan lokasi file valid baik dijalankan di Windows
            APP_DIR = os.path.join(settings.BASE_DIR, 'sentimen')
            DATA_PATH = os.path.join(APP_DIR, 'hasil_final_preprocessed.csv')
            MODEL_PATH = os.path.join(APP_DIR, 'model_random_forest_final.joblib')
            VEC_PATH = os.path.join(APP_DIR, 'tfidf_vectorizer_final.joblib')
            EVAL_PATH = os.path.join(APP_DIR, 'hasil_evaluasi.json')

            # 2. Memuat Dataset Ulasan (CSV)
            if os.path.exists(DATA_PATH):
                cls.df_preview = pd.read_csv(DATA_PATH)
                rename_map = {}
                
                for col in ['Label', 'label', 'sentimen', 'sentiment', 'class']:
                    if col in cls.df_preview.columns: rename_map[col] = 'sentiment_label'
                if rename_map: cls.df_preview.rename(columns=rename_map, inplace=True)
                else: cls.df_preview['sentiment_label'] = 'N/A'
                if 'sentiment_label' in cls.df_preview.columns:
                    cls.df_preview['sentiment_label'] = cls.df_preview['sentiment_label'].astype(str).str.capitalize()

            # 3. Memuat Model Machine Learning & Vectorizer
            if os.path.exists(MODEL_PATH):
                with open(MODEL_PATH, 'rb') as f: cls.model = joblib.load(f)
            if os.path.exists(VEC_PATH):
                with open(VEC_PATH, 'rb') as f: cls.vectorizer = joblib.load(f)
            if os.path.exists(EVAL_PATH):
                with open(EVAL_PATH, 'r', encoding='utf-8') as f: cls.EVALUATION_RESULTS = json.load(f)
            
            print("✅ DatasetManager: Aset berhasil dimuat.")
        except Exception as e:
            print(f"❌ PENTING: Error fatal saat DatasetManager inisialisasi: {e}")
        
    @classmethod
    def reload_evaluation_data(cls):
        EVAL_PATH = os.path.join(settings.BASE_DIR, 'sentimen', 'hasil_evaluasi.json')
        try:
            if os.path.exists(EVAL_PATH):
                with open(EVAL_PATH, 'r', encoding='utf-8') as f:
                    cls.EVALUATION_RESULTS = json.load(f)
                    return True
        except Exception as e: print(f"Gagal reload JSON: {e}")
        return False
      

class KlasifikasiController:
    """Class untuk memproses logika NLP dan prediksi algoritma."""
    
    @staticmethod
    def cleaning_text(text_mentah):
        # 1. Normalisasi Awal
        text = str(text_mentah).lower()
        tahap1_normalisasi = NORMALIZATION_REGEX.sub(lambda match: REPLACEMENTS_DICT[match.group(0)], text)
        tahap1_clean = re.sub(r'[^a-zA-Z\s]', ' ', tahap1_normalisasi).strip()
        tokens = tahap1_clean.split()

        # 2. Filter Penghapusan Stopwords
        tahap2_stopwords = [] 
        for word in tokens: 
            if word not in stopwords_indonesia:
                tahap2_stopwords.append(word) 

        # 3. Proses Pemotongan Kata Dasar (Stemming)
        if stemmer:
            tahap3_stemming_tokens = [] 
            for word in tahap2_stopwords:
                tahap3_stemming_tokens.append(stemmer.stem(word))
        else:
            tahap3_stemming_tokens = tahap2_stopwords 

        # 4. Penggabungan Teks Final & Pencarian Kata Pemicu
        text_final_untuk_model = ' '.join(tahap3_stemming_tokens)
        kata_pemicu = []
        words_raw = re.split(r'\s+', str(text_mentah).lower())
        
        for word in words_raw:
            kata_bersih = re.sub(r'[^a-zA-Z]', '', word)
            if kata_bersih:
                kata_normalized = NORMALIZATION_REGEX.sub(lambda m: REPLACEMENTS_DICT[m.group(0)], kata_bersih)#Node14
                if stemmer: 
                    
                    kata_stemmed = stemmer.stem(kata_normalized)
                else:
                    
                    kata_stemmed = kata_normalized 
                    
                if kata_stemmed in tahap3_stemming_tokens:
                    kata_pemicu.append(kata_bersih) 
        return tahap1_clean, ' '.join(tahap2_stopwords), text_final_untuk_model, list(set(kata_pemicu))#Node20

    @staticmethod
    def get_cosine_similarity(text_vector, prediction):
        # 1. Inisialisasi Variabel Global
        df_preview = DatasetManager.df_preview
        vectorizer = DatasetManager.vectorizer   
        similar_reviews = [] 
        if df_preview.empty or vectorizer is None: 
            return similar_reviews 

        
        available_cols = set(df_preview.columns)
        
        
        col_proc = next(iter(available_cols.intersection(['content_cleaned', 'teks_bersih', 'processed_text', 'text_final'])), None)
        col_orig = next(iter(available_cols.intersection(['content', 'teks_asli', 'text', 'ulasan'])), None) #Node4
        if not col_proc and col_orig:
            col_proc = col_orig 

        if col_proc and col_orig: 
            try: 
                pred_lower = str(prediction).lower()
                df_subset = df_preview[df_preview['sentiment_label'].astype(str).str.lower() == pred_lower].copy()#Node9
                if df_subset.empty:
                    df_subset = df_preview.copy() 
                    
                if len(df_subset) > 500:
                    df_subset = df_subset.sample(500, random_state=42) 
                subset_vectors = vectorizer.transform(df_subset[col_proc].fillna('').astype(str)) 
                similarities = cosine_similarity(text_vector, subset_vectors).flatten() 
                top_indices = similarities.argsort()[-3:][::-1] 
                for idx in top_indices: 
                    if similarities[idx] >= 0.0: 
                        similar_reviews.append({
                            'text': df_subset.iloc[idx][col_orig],
                            'similarity': f"{similarities[idx]:.0%}"
                        })
            except Exception as e:
                print(f"⚠️ Gagal mencari similarity: {e}") 
                similar_reviews = []
        return similar_reviews 


    @staticmethod
    def predict(text_mentah):
        normalisasi, stopwords_txt, text_final, trigger_words = KlasifikasiController.cleaning_text(text_mentah) 
        
        
        if any(word in text_final.split() for word in ('umpat', 'umpatan')):
            return {
                'normalisasi': normalisasi,
                'stopwords': stopwords_txt,
                'stemming': text_final,
                'sentiment': 'Negatif',
                'confidence': '99.00%',
                'trigger_words': trigger_words,
                'similar_reviews': []
            }
        
        text_vector = DatasetManager.vectorizer.transform([text_final])
        
        if text_vector.nnz == 0:
            raise ValueError("Kata tidak dikenali oleh sistem atau merupakan ketikan acak. Mohon masukkan ulasan yang bermakna.")
        
        prediction = DatasetManager.model.predict(text_vector)[0]
        confidence = DatasetManager.model.predict_proba(text_vector).max() 
        similar_reviews = KlasifikasiController.get_cosine_similarity(text_vector, prediction) 
        
        return {
            'normalisasi': normalisasi,
            'stopwords': stopwords_txt,
            'stemming': text_final,
            'sentiment': prediction,
            'confidence': f"{confidence:.2%}",
            'trigger_words': trigger_words,
            'similar_reviews': similar_reviews
        }
        
class DashboardController:
    """Class untuk menyiapkan data yang dirender ke halaman Dashboard & Evaluasi."""
    @staticmethod 
    def get_dashboard_context():
        if not DatasetManager.EVALUATION_RESULTS: DatasetManager.reload_evaluation_data()
        report = DatasetManager.EVALUATION_RESULTS.get('classification_report', {})
        df_preview = DatasetManager.df_preview
        
        if not report and not df_preview.empty and 'sentiment_label' in df_preview.columns:
            counts = df_preview['sentiment_label'].value_counts()
            positif = counts.get('Positif', 0); negatif = counts.get('Negatif', 0); netral = counts.get('Netral', 0)
        else:
            positif = int(report.get('Positif', {}).get('support', 0))
            negatif = int(report.get('Negatif', {}).get('support', 0))
            netral = int(report.get('Netral', {}).get('support', 0))
        
        return {
            'total_ulasan': len(df_preview) if not df_preview.empty else 0,
            'positif': positif, 'negatif': negatif, 'netral': netral,
            'akurasi': DatasetManager.EVALUATION_RESULTS.get('akurasi', "N/A"),
            'f1_score': DatasetManager.EVALUATION_RESULTS.get('f1_score_weighted', "N/A"),
            'data_preview': df_preview.fillna('').head(5).to_dict('records') if not df_preview.empty else []
        }

    @staticmethod
    def get_evaluasi_context(): 
        if not DatasetManager.EVALUATION_RESULTS: DatasetManager.reload_evaluation_data()
        
        report = copy.deepcopy(DatasetManager.EVALUATION_RESULTS.get('classification_report', {}))
        for key, metrics in report.items():
            if isinstance(metrics, dict) and 'f1-score' in metrics: metrics['f1_score'] = metrics.pop('f1-score')

        weighted_avg = report.get('weighted avg', {})
        classification_rows = []
        labels_in_order = DatasetManager.EVALUATION_RESULTS.get('confusion_matrix_labels', [])
       
        for class_name in labels_in_order:
            if class_name in report:
                classification_rows.append({
                    'class_name': class_name, 'precision': report[class_name].get('precision', 0),
                    'recall': report[class_name].get('recall', 0), 'f1_score': report[class_name].get('f1_score', 0),
                    'support': report[class_name].get('support', 0)
                })

        cm_data = []
        matrix = DatasetManager.EVALUATION_RESULTS.get('confusion_matrix', [])
        if labels_in_order and matrix and len(labels_in_order) == len(matrix): 
            for i, actual_label in enumerate(labels_in_order):                 
                row_data = {'actual_label': actual_label, 'predictions': []}
                for j, pred_label in enumerate(labels_in_order):
                    row_data['predictions'].append({ 'value': matrix[i][j], 'is_correct': (i == j) })
                cm_data.append(row_data)
                    
        return { 
            'akurasi': DatasetManager.EVALUATION_RESULTS.get('akurasi', 'N/A'),
            'presisi_weighted': weighted_avg.get('precision', 0),
            'recall_weighted': weighted_avg.get('recall', 0),
            'f1_score_weighted': DatasetManager.EVALUATION_RESULTS.get('f1_score_weighted', 'N/A'),
            'classification_rows': classification_rows,
            'weighted_avg': weighted_avg,
            'cm_data_for_template': cm_data,
            'confusion_matrix_labels': labels_in_order
        }

class VisualisasiController:
    """Class untuk menyiapkan data JSON yang dirender pada Chart JS / Wordcloud."""
    @staticmethod
    def get_visualisasi_context():
        if not DatasetManager.EVALUATION_RESULTS: DatasetManager.reload_evaluation_data()
        EVAL = DatasetManager.EVALUATION_RESULTS
        return {
            'data_aman_untuk_visualisasi': bool(EVAL.get('tren_labels')),
            'tren_labels': json.dumps(EVAL.get('tren_labels', [])),
            'tren_positif': json.dumps(EVAL.get('tren_positif', [])),
            'tren_negatif': json.dumps(EVAL.get('tren_negatif', [])),
            'tren_netral': json.dumps(EVAL.get('tren_netral', [])),
            'wordcloud_positif': json.dumps(EVAL.get('wordcloud_positif', [])),
            'wordcloud_negatif': json.dumps(EVAL.get('wordcloud_negatif', [])),
            'wordcloud_netral': json.dumps(EVAL.get('wordcloud_netral', [])),
        } 