# Face Detection App

Aplikasi deteksi wajah interaktif berbasis Streamlit.

## Cara Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Struktur Folder

```
face_detection_app/
├── app.py              # Aplikasi utama
├── requirements.txt    # Dependencies
├── README.md
└── models/
    ├── deploy.prototxt
    └── res10_300x300.caffemodel
```

## Deployment (Hugging Face Spaces)

Rename `app.py` → `app.py`, upload semua file termasuk folder `models/`.
Set SDK: Streamlit.
