import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd

st.set_page_config(
    page_title="Face Detection App",
    page_icon="👤",
    layout="wide",
)

st.markdown("""
<style>
    /* ── Global ── */
    [data-testid="stAppViewContainer"] { background: #f8f9fa; }
    [data-testid="stMain"] > div { padding-top: 2rem; }

    /* ── Fix: pastikan teks utama selalu gelap (tanpa override widget Streamlit) ── */
    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3,
    [data-testid="stMain"] h4,
    [data-testid="stMain"] p,
    [data-testid="stMain"] span,
    [data-testid="stMain"] label { color: #1a1a2e; }

    /* ── File uploader: paksa tampil normal (putih, bukan hitam) ── */
    [data-testid="stFileUploader"] {
        background: white !important;
        border-radius: 12px !important;
    }
    [data-testid="stFileUploader"] section {
        background: white !important;
        border: 2px dashed #d1d5db !important;
        border-radius: 12px !important;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: #667eea !important;
        background: #f5f3ff !important;
    }
    [data-testid="stFileUploader"] section * { color: #555 !important; }
    [data-testid="stFileUploader"] button { color: #667eea !important; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #1a1a2e !important;
        box-shadow: 4px 0 24px rgba(0,0,0,0.18);
    }
    /* Hanya warnai elemen DALAM sidebar */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 { color: white !important; }

    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label { color: rgba(255,255,255,0.7) !important; font-size: 0.85rem; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.1) !important; }

    /* ── Sidebar logo area ── */
    .sidebar-logo {
        display: flex; align-items: center; gap: 12px;
        padding: 1.2rem 0 1rem;
    }
    .sidebar-logo-icon {
        width: 48px; height: 48px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 14px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.5rem; flex-shrink: 0;
    }
    .sidebar-logo-text h2 { margin: 0; font-size: 1rem; font-weight: 700; color: white !important; }
    .sidebar-logo-text p  { margin: 0; font-size: 0.75rem; color: rgba(255,255,255,0.5) !important; }

    /* ── Gradient Cards (metric) ── */
    .g-card {
        border-radius: 20px; padding: 2rem 1.5rem;
        color: white !important; margin-bottom: 0.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.12);
    }
    .g-card * { color: white !important; }
    .g-card .icon  { font-size: 2rem; margin-bottom: 0.5rem; opacity: 0.85; }
    .g-card .label { font-size: 0.82rem; opacity: 0.85; margin-bottom: 0.3rem; }
    .g-card .value { font-size: 3rem; font-weight: 800; line-height: 1; }
    .g-card .sub   { font-size: 0.78rem; opacity: 0.7; margin-top: 0.3rem; }
    .g-card-blue { background: linear-gradient(135deg, #667eea, #764ba2); }
    .g-card-pink { background: linear-gradient(135deg, #f093fb, #f5576c); }
    .g-card-cyan { background: linear-gradient(135deg, #4facfe, #00f2fe); }

    /* ── White Cards ── */
    .w-card {
        background: white; border-radius: 20px;
        padding: 1.8rem; box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }
    .w-card h3 { font-size: 1.2rem; font-weight: 700; margin-bottom: 1rem; color: #1a1a2e !important; }
    .w-card p  { color: #555 !important; }

    /* ── Info box ── */
    .info-box {
        background: #f0f4ff; border-left: 4px solid #667eea;
        border-radius: 10px; padding: 1rem 1.2rem; margin: 0.5rem 0;
        font-size: 0.92rem; color: #1a1a2e !important;
    }
    .info-box * { color: #1a1a2e !important; }

    /* ── Hero section ── */
    .hero-icon {
        width: 96px; height: 96px; margin: 0 auto 1.5rem;
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 28px;
        display: flex; align-items: center; justify-content: center;
        font-size: 3rem;
        box-shadow: 0 12px 40px rgba(102,126,234,0.4);
    }
    .hero-title {
        font-size: 3rem !important; font-weight: 800 !important;
        color: #1a1a2e !important; text-align: center; margin-bottom: 0.5rem;
    }
    .hero-sub {
        font-size: 1.1rem; color: #666 !important; text-align: center;
        max-width: 560px; margin: 0 auto 2rem;
    }

    /* ── Feature cards (home) ── */
    .feat-card {
        border-radius: 20px; padding: 2rem 1.5rem;
        color: white !important; box-shadow: 0 8px 32px rgba(0,0,0,0.14);
        height: 100%;
    }
    .feat-card * { color: white !important; }
    .feat-card .feat-icon { font-size: 2.5rem; margin-bottom: 0.8rem; }
    .feat-card h3 { font-size: 1.2rem; font-weight: 700; margin: 0 0 0.5rem; }
    .feat-card p  { font-size: 0.88rem; opacity: 0.9; margin: 0; }

    /* ── Step items ── */
    .step-row { display: flex; gap: 1rem; align-items: flex-start; margin-bottom: 1rem; }
    .step-num {
        width: 40px; height: 40px; flex-shrink: 0;
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 12px; display: flex; align-items: center; justify-content: center;
        color: white !important; font-weight: 700; font-size: 1rem;
        box-shadow: 0 4px 12px rgba(102,126,234,0.35);
    }
    .step-body h4 { margin: 0 0 0.2rem; font-size: 0.95rem; font-weight: 600; color: #1a1a2e !important; }
    .step-body p  { margin: 0; font-size: 0.85rem; color: #666 !important; }

    /* ── Identity / Tech stack ── */
    .id-row { margin-bottom: 0.8rem; }
    .id-row .id-label { font-size: 0.75rem; color: #999 !important; margin-bottom: 0.1rem; }
    .id-row .id-val   { font-weight: 600; color: #1a1a2e !important; }
    .tech-badge {
        display: flex; align-items: center; gap: 10px;
        padding: 0.5rem 1rem; background: #f8f9fa;
        border-radius: 12px; margin-bottom: 0.5rem;
        font-weight: 500; color: #1a1a2e !important; font-size: 0.9rem;
    }

    /* ── Comparison cards ── */
    .cmp-card {
        background: white; border-radius: 20px; padding: 1.8rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    }
    .cmp-card.left-blue { border-left: 4px solid #667eea; }
    .cmp-card.left-pink { border-left: 4px solid #f5576c; }
    .cmp-card-header { display: flex; align-items: center; gap: 12px; margin-bottom: 1.2rem; }
    .cmp-icon {
        width: 48px; height: 48px; border-radius: 14px;
        display: flex; align-items: center; justify-content: center; font-size: 1.4rem;
    }
    .cmp-icon-blue { background: linear-gradient(135deg, #667eea, #764ba2); }
    .cmp-icon-pink { background: linear-gradient(135deg, #f093fb, #f5576c); }
    .cmp-row { display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid #f0f0f0; }
    .cmp-row:last-child { border-bottom: none; }
    .cmp-row .cmp-key { color: #666 !important; font-size: 0.9rem; }
    .cmp-row .cmp-val { font-weight: 600; color: #1a1a2e !important; font-size: 0.9rem; }

    /* ── Method badge in table ── */
    .badge-haar { background: #ede9fe; color: #6d28d9 !important; padding: 2px 10px; border-radius: 8px; font-size: 0.8rem; font-weight: 500; }
    .badge-dnn  { background: #fce7f3; color: #be185d !important; padding: 2px 10px; border-radius: 8px; font-size: 0.8rem; font-weight: 500; }
    .badge-both { background: #e0f2fe; color: #0369a1 !important; padding: 2px 10px; border-radius: 8px; font-size: 0.8rem; font-weight: 500; }
    .badge-upload  { background: #dbeafe; color: #1d4ed8 !important; padding: 2px 10px; border-radius: 8px; font-size: 0.8rem; font-weight: 500; }
    .badge-webcam  { background: #dcfce7; color: #15803d !important; padding: 2px 10px; border-radius: 8px; font-size: 0.8rem; font-weight: 500; }

    /* ── Metric override ── */
    [data-testid="stMetric"] {
        background: white; border-radius: 14px; padding: 1rem 1.2rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    }
    [data-testid="stMetricLabel"] { color: #666 !important; }
    [data-testid="stMetricValue"] { color: #1a1a2e !important; }

    /* ── Streamlit widget text di area utama ── */
    [data-testid="stMain"] .stRadio label { color: #1a1a2e !important; }
    [data-testid="stMain"] .stSelectbox label { color: #1a1a2e !important; }
    [data-testid="stMain"] .stFileUploader label { color: #1a1a2e !important; }
    [data-testid="stMain"] .stCheckbox label { color: #1a1a2e !important; }
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] p { color: #1a1a2e !important; }

    /* hide streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    header { visibility: hidden; }

    /* ── Fix: tombol toggle sidebar selalu visible ── */
    [data-testid="collapsedControl"] {
        visibility: visible !important;
        display: flex !important;
        opacity: 1 !important;
        z-index: 999999 !important;
    }
    [data-testid="stSidebarCollapseButton"] {
        visibility: visible !important;
        display: flex !important;
    }
    [data-testid="stSidebarCollapseButton"] svg path {
        fill: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────
for k, v in [
    ("total_tested", 0),
    ("total_faces", 0),
    ("infer_times", []),
    ("history", []),
    ("last_file", None),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# Model Loaders
# ─────────────────────────────────────────────
@st.cache_resource
def load_haar():
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(path)

@st.cache_resource
def load_dnn():
    return cv2.dnn.readNetFromCaffe(
        "models/deploy.prototxt",
        "models/res10_300x300.caffemodel"
    )

# ─────────────────────────────────────────────
# Detection Functions
# ─────────────────────────────────────────────
MAX_DIM = 1280

def resize_for_processing(img, max_dim=MAX_DIM):
    """Resize gambar besar agar tidak OOM. Kembalikan (img_kecil, scale_x, scale_y)."""
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img, 1.0, 1.0
    if w >= h:
        new_w = max_dim
        new_h = int(h * max_dim / w)
    else:
        new_h = max_dim
        new_w = int(w * max_dim / h)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, w / new_w, h / new_h

def detect_haar(img, scale, neighbors):
    small, sx, sy = resize_for_processing(img)
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    clf   = load_haar()
    t0    = time.time()
    faces = clf.detectMultiScale(gray, scaleFactor=scale, minNeighbors=neighbors, minSize=(30, 30))
    ms    = (time.time() - t0) * 1000
    out   = img.copy()
    for (x, y, w, h) in faces:
        x1, y1 = int(x * sx), int(y * sy)
        x2, y2 = int((x + w) * sx), int((y + h) * sy)
        cv2.rectangle(out, (x1, y1), (x2, y2), (50, 205, 100), 2)
        cv2.putText(out, "Face", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 205, 100), 2)
    return out, len(faces), ms, []

def detect_dnn(img, conf_thresh):
    small, sx, sy = resize_for_processing(img)
    net  = load_dnn()
    sh, sw = small.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(small, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    t0   = time.time()
    net.setInput(blob)
    dets = net.forward()
    ms   = (time.time() - t0) * 1000
    out  = img.copy()
    face_count = 0
    scores = []
    for i in range(dets.shape[2]):
        conf = float(dets[0, 0, i, 2])
        if conf < conf_thresh:
            continue
        x1 = int(dets[0, 0, i, 3] * sw * sx); y1 = int(dets[0, 0, i, 4] * sh * sy)
        x2 = int(dets[0, 0, i, 5] * sw * sx); y2 = int(dets[0, 0, i, 6] * sh * sy)
        x1, y1 = max(0, x1), max(0, y1)
        cv2.rectangle(out, (x1, y1), (x2, y2), (80, 120, 255), 2)
        cv2.putText(out, f"{conf:.0%}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 120, 255), 2)
        face_count += 1
        scores.append(round(conf, 3))
    return out, face_count, ms, scores

def update_stats(n_faces, ms, source, method):
    st.session_state.total_tested += 1
    st.session_state.total_faces  += n_faces
    st.session_state.infer_times.append(ms)
    st.session_state.history.append({
        "Metode": method, "Wajah": n_faces,
        "Waktu (ms)": round(ms, 2), "Sumber": source
    })

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">👤</div>
        <div class="sidebar-logo-text">
            <h2>Face Detection</h2>
            <p>Computer Vision App</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio("📌 Navigasi", ["🏠 Home", "🔍 Detection", "📊 Dashboard"])
    st.markdown("---")

    st.markdown("##### Metode Deteksi")
    method = st.selectbox("Pilih Metode", ["Haar Cascade", "OpenCV DNN", "Perbandingan Keduanya"], label_visibility="collapsed")

    if method in ["Haar Cascade", "Perbandingan Keduanya"]:
        st.markdown("##### Haar Cascade")
        haar_scale     = st.slider("Scale Factor",  1.05, 1.5, 1.1, 0.05)
        haar_neighbors = st.slider("Min Neighbors", 1, 10, 5, 1)
    else:
        haar_scale, haar_neighbors = 1.1, 5

    if method in ["OpenCV DNN", "Perbandingan Keduanya"]:
        st.markdown("##### OpenCV DNN")
        dnn_conf = st.slider("Min Confidence", 0.1, 1.0, 0.5, 0.05)
    else:
        dnn_conf = 0.5

# ─────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 2rem;">
        <div class="hero-icon">👤</div>
        <div class="hero-title">Face Detection App</div>
        <div class="hero-sub">
            Aplikasi deteksi wajah menggunakan Haar Cascade dan OpenCV DNN
            dengan antarmuka yang modern dan intuitif
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="feat-card" style="background: linear-gradient(135deg,#667eea,#764ba2);">
            <div class="feat-icon">🎯</div>
            <h3>Haar Cascade</h3>
            <p>Metode klasik yang cepat dan efisien untuk deteksi wajah real-time</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="feat-card" style="background: linear-gradient(135deg,#f093fb,#f5576c);">
            <div class="feat-icon">🤖</div>
            <h3>OpenCV DNN</h3>
            <p>Deep learning model ResNet SSD untuk akurasi deteksi yang lebih tinggi</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="feat-card" style="background: linear-gradient(135deg,#4facfe,#00f2fe);">
            <div class="feat-icon">📊</div>
            <h3>Dashboard Statistik</h3>
            <p>Visualisasi performa dan riwayat deteksi untuk analisis mendalam</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="w-card" style="border-left: 4px solid #667eea;">
        <h3>📋 Tentang Proyek</h3>
        <p style="color:#555; line-height:1.7;">
        Aplikasi ini dikembangkan untuk membandingkan dua metode deteksi wajah yang populer:
        <b>Haar Cascade Classifier</b> dan <b>OpenCV DNN (ResNet-10 SSD)</b>.
        Pengguna dapat menguji kedua metode pada gambar statis maupun video,
        serta melihat perbandingan performa keduanya dalam dashboard statistik yang komprehensif.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="w-card">
        <h3>🚀 Cara Penggunaan</h3>
        <div class="step-row">
            <div class="step-num">1</div>
            <div class="step-body">
                <h4>Pilih Metode Deteksi</h4>
                <p>Gunakan sidebar untuk memilih metode (Haar Cascade, OpenCV DNN, atau Perbandingan)</p>
            </div>
        </div>
        <div class="step-row">
            <div class="step-num">2</div>
            <div class="step-body">
                <h4>Atur Parameter</h4>
                <p>Sesuaikan Scale Factor, Min Neighbors, atau Min Confidence di sidebar</p>
            </div>
        </div>
        <div class="step-row">
            <div class="step-num">3</div>
            <div class="step-body">
                <h4>Upload Gambar atau Video</h4>
                <p>Pada halaman Detection, pilih upload gambar atau video</p>
            </div>
        </div>
        <div class="step-row">
            <div class="step-num">4</div>
            <div class="step-body">
                <h4>Lihat Hasil dan Statistik</h4>
                <p>Analisis hasil deteksi dan lihat performa di halaman Dashboard</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="w-card">
            <h3>👨‍💻 Identitas</h3>
            <div class="id-row"><div class="id-label">Nama</div><div class="id-val">Muhammad Hafiz</div></div>
            <div class="id-row"><div class="id-label">NIM</div><div class="id-val">2311532007</div></div>
            <div class="id-row"><div class="id-label">Program Studi</div><div class="id-val">Informatika</div></div>
            <div class="id-row"><div class="id-label">Universitas</div><div class="id-val">Universitas Andalas</div></div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="w-card">
            <h3>🛠️ Technology Stack</h3>
            <div class="tech-badge">🐍 Python 3</div>
            <div class="tech-badge">⚡ Streamlit</div>
            <div class="tech-badge">📷 OpenCV 4</div>
            <div class="tech-badge">📊 NumPy &amp; Pandas</div>
            <div class="tech-badge">🧠 ResNet-10 SSD DNN</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DETECTION
# ─────────────────────────────────────────────
elif page == "🔍 Detection":
    st.markdown("## 🔍 Face Detection")
    st.markdown("<p style='color:#666;margin-top:-0.5rem;'>Upload gambar atau video untuk deteksi wajah</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    input_mode = st.radio("Input Mode", ["📁 Upload Gambar", "🎥 Upload Video"], horizontal=True, label_visibility="collapsed")
    st.markdown("---")

    if input_mode == "📁 Upload Gambar":
        uploaded = st.file_uploader("Upload gambar (JPG / PNG)", type=["jpg", "jpeg", "png"])

        if uploaded:
            arr = np.frombuffer(uploaded.read(), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            col_orig, col_res = st.columns(2)
            with col_orig:
                st.markdown('<div class="w-card"><h3>🖼️ Gambar Asli</h3>', unsafe_allow_html=True)
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            if method != "Perbandingan Keduanya":
                if method == "Haar Cascade":
                    out, n, ms, scores = detect_haar(img, haar_scale, haar_neighbors)
                else:
                    out, n, ms, scores = detect_dnn(img, dnn_conf)

                with col_res:
                    st.markdown(f'<div class="w-card"><h3>✅ Hasil – {method}</h3>', unsafe_allow_html=True)
                    st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f"""
                    <div class="g-card g-card-blue">
                        <div class="icon">👤</div>
                        <div class="label">Wajah Terdeteksi</div>
                        <div class="value">{n}</div>
                    </div>""", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div class="g-card g-card-pink">
                        <div class="icon">⏱️</div>
                        <div class="label">Waktu Inferensi</div>
                        <div class="value">{ms:.0f}<span style="font-size:1.2rem;margin-left:4px">ms</span></div>
                    </div>""", unsafe_allow_html=True)
                with m3:
                    conf_val = f"{np.mean(scores):.0%}" if scores else "N/A"
                    st.markdown(f"""
                    <div class="g-card g-card-cyan">
                        <div class="icon">🎯</div>
                        <div class="label">Avg Confidence</div>
                        <div class="value">{conf_val}</div>
                    </div>""", unsafe_allow_html=True)

                # Catat statistik hanya jika file baru
                if st.session_state.last_file != uploaded.name:
                    update_stats(n, ms, uploaded.name, method)
                    st.session_state.last_file = uploaded.name

            else:
                out_h, n_h, ms_h, _    = detect_haar(img, haar_scale, haar_neighbors)
                out_d, n_d, ms_d, sc_d = detect_dnn(img, dnn_conf)

                with col_res:
                    st.markdown('<div class="w-card"><h3>✅ Haar Cascade</h3>', unsafe_allow_html=True)
                    st.image(cv2.cvtColor(out_h, cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="w-card"><h3>🤖 OpenCV DNN</h3>', unsafe_allow_html=True)
                st.image(cv2.cvtColor(out_d, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("<br>**📊 Perbandingan**")
                c = st.columns(5)
                c[0].metric("Haar – Wajah",  n_h)
                c[1].metric("Haar – Waktu",  f"{ms_h:.1f} ms")
                c[2].metric("DNN – Wajah",   n_d)
                c[3].metric("DNN – Waktu",   f"{ms_d:.1f} ms")
                if sc_d:
                    c[4].metric("DNN Conf",  f"{np.mean(sc_d):.1%}")

                # Catat statistik hanya jika file baru
                if st.session_state.last_file != uploaded.name:
                    update_stats(max(n_h, n_d), (ms_h + ms_d) / 2, uploaded.name, "Both")
                    st.session_state.last_file = uploaded.name

        else:
            st.markdown("""
            <div class="w-card" style="text-align:center; padding: 3rem;">
                <div style="font-size:3rem; margin-bottom:1rem; opacity:0.3;">📁</div>
                <p style="color:#999;">Klik atau drag gambar untuk upload<br>
                <span style="font-size:0.82rem;">JPG, PNG (maks. 5MB)</span></p>
            </div>
            """, unsafe_allow_html=True)

    else:
        uploaded_video = st.file_uploader("Upload video (MP4 / AVI)", type=["mp4", "avi", "mov"])

        if uploaded_video:
            import tempfile, os
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            tfile.flush()
            tfile.close()

            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30

            st.markdown(f"**{total_frames} frame** | **{fps:.0f} FPS** | Proses setiap 5 frame")
            progress  = st.progress(0)
            frame_slot = st.empty()
            info_slot  = st.empty()

            fc = 0
            last_n, last_ms = 0, 0.0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                fc += 1
                progress.progress(min(fc / max(total_frames, 1), 1.0))

                if fc % 5 != 0:
                    continue

                if method == "Haar Cascade":
                    out, n, ms, _ = detect_haar(frame, haar_scale, haar_neighbors)
                    info_slot.markdown(f"👤 **{n} wajah** | ⏱️ **{ms:.1f} ms** | Frame {fc}/{total_frames}")
                elif method == "OpenCV DNN":
                    out, n, ms, sc = detect_dnn(frame, dnn_conf)
                    conf_str = f" | 🎯 {np.mean(sc):.1%}" if sc else ""
                    info_slot.markdown(f"👤 **{n} wajah** | ⏱️ **{ms:.1f} ms** | Frame {fc}/{total_frames}{conf_str}")
                else:
                    out_h, n_h, ms_h, _ = detect_haar(frame, haar_scale, haar_neighbors)
                    out_d, n_d, ms_d, _ = detect_dnn(frame, dnn_conf)
                    out = np.hstack([out_h, out_d])
                    n, ms = max(n_h, n_d), (ms_h + ms_d) / 2
                    info_slot.markdown(f"Haar: **{n_h}** wajah / {ms_h:.0f}ms | DNN: **{n_d}** wajah / {ms_d:.0f}ms")

                last_n, last_ms = n, ms
                frame_slot.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_container_width=True)

            cap.release()
            os.unlink(tfile.name)
            progress.progress(1.0)
            st.success("✅ Video selesai diproses!")

            # Catat statistik satu kali setelah video selesai
            if st.session_state.last_file != uploaded_video.name:
                update_stats(last_n, last_ms, uploaded_video.name, method)
                st.session_state.last_file = uploaded_video.name

        else:
            st.markdown("""
            <div class="w-card" style="text-align:center; padding: 3rem;">
                <div style="font-size:3rem; margin-bottom:1rem; opacity:0.3;">🎥</div>
                <p style="color:#999;">Upload video untuk deteksi wajah per-frame<br>
                <span style="font-size:0.82rem;">MP4, AVI, MOV (maks. 200MB)</span></p>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────
elif page == "📊 Dashboard":
    st.markdown("## 📊 Dashboard Statistik")
    st.markdown("<p style='color:#666; margin-top:-0.5rem;'>Analisis performa dan riwayat deteksi wajah</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    total = st.session_state.total_tested
    faces = st.session_state.total_faces
    times = st.session_state.infer_times
    avg_t = np.mean(times) if times else 0
    avg_per = round(faces / total, 1) if total > 0 else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="g-card g-card-blue">
            <div class="icon">🖼️</div>
            <div class="label">Total Gambar Diuji</div>
            <div class="value">{total}</div>
            <div class="sub">Sesi deteksi</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="g-card g-card-pink">
            <div class="icon">👥</div>
            <div class="label">Total Wajah Terdeteksi</div>
            <div class="value">{faces}</div>
            <div class="sub">Rata-rata {avg_per} per gambar</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="g-card g-card-cyan">
            <div class="icon">⏱️</div>
            <div class="label">Rata-rata Waktu Inferensi</div>
            <div class="value">{avg_t:.0f}<span style="font-size:1.2rem;margin-left:4px">ms</span></div>
            <div class="sub">Seluruh sesi</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if times:
        st.markdown('<div class="w-card">', unsafe_allow_html=True)
        st.markdown("### ⏱️ Waktu Inferensi per Sesi")
        chart_df = pd.DataFrame({"Waktu (ms)": times}, index=range(1, len(times)+1))
        st.line_chart(chart_df, color="#667eea")
        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.history:
            st.markdown('<div class="w-card">', unsafe_allow_html=True)
            st.markdown("### 📋 Riwayat Deteksi")

            rows_html = ""
            for i, row in enumerate(st.session_state.history, 1):
                m = row["Metode"]
                s = row["Sumber"]
                badge_m = (
                    f'<span class="badge-haar">{m}</span>' if m == "Haar Cascade"
                    else f'<span class="badge-dnn">{m}</span>' if m == "OpenCV DNN"
                    else f'<span class="badge-both">{m}</span>'
                )
                badge_s = f'<span class="badge-webcam">{s}</span>' if s == "webcam" else f'<span class="badge-upload">{s}</span>'
                rows_html += f"""
                <tr style="border-bottom:1px solid #f0f0f0;">
                    <td style="padding:12px 8px; color:#666; font-weight:500;">#{i}</td>
                    <td style="padding:12px 8px;">{badge_m}</td>
                    <td style="padding:12px 8px; font-weight:600; color:#1a1a2e;">{row['Wajah']}</td>
                    <td style="padding:12px 8px; color:#666;">{row['Waktu (ms)']} ms</td>
                    <td style="padding:12px 8px;">{badge_s}</td>
                </tr>"""

            st.markdown(f"""
            <table style="width:100%; border-collapse:collapse;">
                <thead>
                    <tr style="border-bottom:2px solid #e9ecef;">
                        <th style="text-align:left;padding:12px 8px;color:#999;font-size:0.82rem;font-weight:600;">ID</th>
                        <th style="text-align:left;padding:12px 8px;color:#999;font-size:0.82rem;font-weight:600;">METODE</th>
                        <th style="text-align:left;padding:12px 8px;color:#999;font-size:0.82rem;font-weight:600;">WAJAH</th>
                        <th style="text-align:left;padding:12px 8px;color:#999;font-size:0.82rem;font-weight:600;">WAKTU</th>
                        <th style="text-align:left;padding:12px 8px;color:#999;font-size:0.82rem;font-weight:600;">SUMBER</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="w-card" style="text-align:center; padding:3rem;">
            <div style="font-size:3rem; opacity:0.2; margin-bottom:1rem;">📊</div>
            <p style="color:#999;">Belum ada data. Jalankan deteksi terlebih dahulu di halaman <b>Detection</b>.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="cmp-card left-blue">
            <div class="cmp-card-header">
                <div class="cmp-icon cmp-icon-blue">🎯</div>
                <h3 style="margin:0;font-size:1.1rem;color:#1a1a2e;">Haar Cascade (Viola-Jones)</h3>
            </div>
            <div class="cmp-row"><span class="cmp-key">Input</span><span class="cmp-val">Grayscale image</span></div>
            <div class="cmp-row"><span class="cmp-key">Fitur</span><span class="cmp-val">Haar-like features</span></div>
            <div class="cmp-row"><span class="cmp-key">Output</span><span class="cmp-val">Bounding box (x,y,w,h)</span></div>
            <div class="cmp-row"><span class="cmp-key">Kecepatan</span><span class="cmp-val">⚡ Sangat Cepat</span></div>
            <div class="cmp-row"><span class="cmp-key">Best For</span><span class="cmp-val">Real-time detection</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="cmp-card left-pink">
            <div class="cmp-card-header">
                <div class="cmp-icon cmp-icon-pink">🤖</div>
                <h3 style="margin:0;font-size:1.1rem;color:#1a1a2e;">OpenCV DNN (ResNet-10 SSD)</h3>
            </div>
            <div class="cmp-row"><span class="cmp-key">Input</span><span class="cmp-val">RGB image 300×300</span></div>
            <div class="cmp-row"><span class="cmp-key">Arsitektur</span><span class="cmp-val">ResNet-10 + SSD</span></div>
            <div class="cmp-row"><span class="cmp-key">Output</span><span class="cmp-val">Bounding box + confidence</span></div>
            <div class="cmp-row"><span class="cmp-key">Kecepatan</span><span class="cmp-val">🔄 Moderate</span></div>
            <div class="cmp-row"><span class="cmp-key">Best For</span><span class="cmp-val">High accuracy</span></div>
        </div>
        """, unsafe_allow_html=True)