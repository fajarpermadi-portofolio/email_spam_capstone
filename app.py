import streamlit as st
import joblib
import hashlib

# =========================
# 🔹 IMPORT PIPELINE (WAJIB DI ATAS)
# =========================
from main_pipeline import clean_text, bloom_weight, fuzzy_weight

# =========================
# 🔹 KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Email Security Classifier",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Email Security Classifier")
st.caption("Taksonomi Bloom + Fuzzy Logic untuk Deteksi Spam & Phishing")

# =========================
# 🔹 LOAD MODEL & VECTORIZER
# =========================
@st.cache_resource
def load_components():
    try:
        model = joblib.load("best_model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        return model, vectorizer
    except Exception as e:
        st.error("❌ Model atau vectorizer tidak ditemukan.")
        st.stop()

model, vectorizer = load_components()

# =========================
# 🔹 HASH VALIDASI
# =========================
def generate_hash(prediction: str) -> str:
    return hashlib.sha256(prediction.encode()).hexdigest()

# =========================
# 🔹 PENJELASAN DINAMIS EMAIL
# =========================
RISK_TERMS = {
    "password": "mengandung kata *password* yang sering digunakan untuk mencuri kredensial.",
    "login": "memuat kata *login* yang umum digunakan dalam phishing akun.",
    "klik": "mengandung kata *klik* yang biasanya mengarah ke tautan berbahaya.",
    "verifikasi": "meminta *verifikasi*, salah satu ciri umum phishing.",
    "akun": "menyebut *akun*, indikasi pencurian data pribadi.",
    "bonus": "menawarkan *bonus* atau *hadiah* yang sering menjadi umpan spam.",
    "transfer": "menyebut *transfer*, potensi penipuan finansial.",
    "bank": "berkaitan dengan *bank*, rawan phishing finansial.",
    "update": "meminta *update* data akun, ciri umum phishing."
}

def explain_email(cleaned_text: str) -> str:
    ditemukan = [t for t in RISK_TERMS if t in cleaned_text]
    if ditemukan:
        detail = " ".join(RISK_TERMS[t] for t in ditemukan)
        return f"Email ini {detail} Harap berhati-hati terhadap tautan atau lampiran mencurigakan."
    return "Email tidak mengandung istilah berisiko umum, namun tetap waspada terhadap tautan eksternal."

# =========================
# 🔹 INPUT EMAIL
# =========================
email_input = st.text_area(
    "✉️ Masukkan isi email:",
    height=200,
    placeholder="Contoh: Selamat! Anda memenangkan hadiah..."
)

# =========================
# 🔹 PROSES DETEKSI
# =========================
if st.button("🔍 Deteksi Email"):
    if not email_input.strip():
        st.warning("⚠️ Silakan masukkan isi email terlebih dahulu.")
        st.stop()

    # Preprocessing & scoring
    cleaned = clean_text(email_input)
    bloom_score = bloom_weight(cleaned)
    fuzzy_score = fuzzy_weight(bloom_score)

    # Prediksi ML
    X_input = vectorizer.transform([cleaned])
    pred = model.predict(X_input)[0]

    # Hash & penjelasan
    hashed = generate_hash(pred)
    penjelasan = explain_email(cleaned)

    # =========================
    # 🔹 TAMPILKAN HASIL
    # =========================
    st.markdown("---")
    st.subheader("📊 Hasil Analisis")

    if pred.lower() in ["spam", "phishing"]:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #ffcccc, #ff9999);
            border-left: 6px solid #d32f2f;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 10px rgba(255,0,0,0.25);
            color: #4a0000;">
            <h3>🚨 Email TERDETEKSI: <b>{pred.upper()}</b></h3>
            <p><b>Skor Bloom:</b> {bloom_score:.2f}</p>
            <p><b>Fuzzy Risk:</b> {fuzzy_score:.2f}</p>
            <p><b>Penjelasan:</b> {penjelasan}</p>
            <p style="font-size:12px;">Hash Validasi: <code>{hashed}</code></p>
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
            color: #003300;">
            <h3>✅ Email AMAN: <b>{pred.upper()}</b></h3>
            <p><b>Skor Bloom:</b> {bloom_score:.2f}</p>
            <p><b>Fuzzy Risk:</b> {fuzzy_score:.2f}</p>
            <p><b>Penjelasan:</b> {penjelasan}</p>
            <p style="font-size:12px;">Hash Validasi: <code>{hashed}</code></p>
        </div>
        """, unsafe_allow_html=True)

# =========================
# 🔹 FOOTER
# =========================
st.markdown("---")
st.caption("Capstone Project • Sistem Informasi • Email Security Classifier")
