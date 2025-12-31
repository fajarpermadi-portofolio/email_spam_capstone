import streamlit as st
import joblib
import hashlib

# =========================
# 🔹 KONFIGURASI HALAMAN
# =========================
st.set_page_config(page_title="Email Security Classifier", page_icon="📧")
st.title(" Email Security Classifier (Taksonomi Bloom + Fuzzy Logic)")

# =========================
# 🔹 LOAD MODEL DAN VEKTOR
# =========================
@st.cache_resource
def load_components():
    model = joblib.load("best_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_components()

# =========================
# 🔹 HASHING FUNKSI
# =========================
def generate_hash(pred):
    return hashlib.sha256(pred.encode()).hexdigest()

# =========================
# 🔹 PENJELASAN DINAMIS
# =========================
RISK_TERMS = {
    "password": "mengandung kata 'password' yang sering digunakan untuk mencuri kredensial.",
    "login": "memuat kata 'login' yang umum digunakan dalam serangan phishing akun.",
    "klik": "mengandung kata 'klik' yang biasanya mengarah ke tautan berbahaya.",
    "verifikasi": "meminta pengguna untuk 'verifikasi' yang sering jadi taktik phishing.",
    "akun": "menyebut 'akun', indikasi umum pencurian data pribadi.",
    "bonus": "menawarkan 'bonus' atau 'hadiah' yang sering menjadi umpan spam.",
    "transfer": "menyebut 'transfer', potensi penipuan finansial.",
    "bank": "berkaitan dengan 'bank', rawan phishing finansial.",
    "update": "meminta 'update' informasi akun, ciri phishing umum."
}

def explain_email_dynamic(cleaned_text):
    ditemukan = [term for term in RISK_TERMS if term in cleaned_text]
    if ditemukan:
        details = " ".join([RISK_TERMS[t] for t in ditemukan])
        return f"Email ini {details} Harap berhati-hati terhadap tautan atau lampiran mencurigakan."
    else:
        return "Email ini tidak mengandung istilah berisiko yang umum. Namun tetap waspada terhadap tautan eksternal."

# =========================
# 🔹 INPUT EMAIL
# =========================
email_input = st.text_area(" Masukkan isi email:", height=200)

# =========================
# 🔹 DETEKSI EMAIL
# =========================
if st.button(" Deteksi Email"):
    from main_pipeline import clean_text, bloom_weight, fuzzy_weight

    if email_input.strip():
        cleaned = clean_text(email_input)
        bloom_score = bloom_weight(cleaned)
        fuzzy_score = fuzzy_weight(bloom_score)
        X_input = vectorizer.transform([cleaned])
        pred = model.predict(X_input)[0]
        hashed = generate_hash(pred)
        penjelasan = explain_email_dynamic(cleaned)

        # =========================
        # 🔹 TAMPILAN HASIL DETEKSI
        # =========================
        st.markdown("---")
        st.subheader(" Hasil Analisis")

        if pred.lower() in ["spam", "phishing"]:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #ffcccc, #ff9999);
                border-left: 6px solid #d32f2f;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 4px 10px rgba(255,0,0,0.25);
                color: #4a0000;
                font-family: 'Segoe UI', sans-serif;">
                <h2 style="margin-top:0;"> <b>Email TERDETEKSI: {pred.upper()}</b></h2>
                <hr style="border: 1px solid #ff5555;">
                <p><b> Skor Bloom:</b> {bloom_score:.2f}</p>
                <p><b> Fuzzy Risk:</b> {fuzzy_score:.2f}</p>
                <p style="margin-top:15px; font-size:15px;">
                     <b>Penjelasan:</b> {penjelasan}
                </p>
                <p style="font-size: 12px; color: #6e0000; margin-top:15px;">
                     Hash Validasi: <code>{hashed}</code>
                </p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #d7ffd9, #b9ffbf);
                border-left: 6px solid #2e7d32;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 4px 10px rgba(0,128,0,0.25);
                color: #003300;
                font-family: 'Segoe UI', sans-serif;">
                <h2 style="margin-top:0;"> <b>Email AMAN: {pred.upper()}</b></h2>
                <hr style="border: 1px solid #55ff55;">
                <p><b> Skor Bloom:</b> {bloom_score:.2f}</p>
                <p><b> Fuzzy Risk:</b> {fuzzy_score:.2f}</p>
                <p style="margin-top:15px; font-size:15px;">
                     <b>Penjelasan:</b> {penjelasan}
                </p>
                <p style="font-size: 12px; color: #004d00; margin-top:15px;">
                     Hash Validasi: <code>{hashed}</code>
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning(" Harap masukkan teks email terlebih dahulu sebelum mendeteksi.")
