from collections import defaultdict
from datetime import datetime
from html import escape
from pathlib import Path, PurePosixPath
import re
from textwrap import dedent

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input

from custom_layers import CBAMBlock


# KONFIGURASI HALAMAN
st.set_page_config(
    page_title="MRI ASD Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# GAYA TAMPILAN
st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(
                    circle at top right,
                    rgba(59, 130, 246, 0.08),
                    transparent 35%
                ),
                radial-gradient(
                    circle at top left,
                    rgba(139, 92, 246, 0.06),
                    transparent 30%
                ),
                #f8fafc;
            color: #1f2937;
        }

        .main-title {
            font-size: 2.7rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
            background: linear-gradient(
                90deg,
                #60a5fa,
                #a78bfa,
                #f472b6
            );
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .subtitle {
            color: #475569;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }

        .hero-card {
            border: 1px solid rgba(148, 163, 184, 0.30);
            background: rgba(255, 255, 255, 0.94);
            color: #1f2937;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
            border-radius: 18px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 1.4rem;
        }

        .result-card {
            border: 1px solid rgba(148, 163, 184, 0.30);
            background: #ffffff;
            color: #1f2937;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
            border-radius: 18px;
            padding: 1.3rem;
            min-height: 230px;
        }

        .model-label {
            color: #64748b;
            font-size: 0.9rem;
            margin-bottom: 0.3rem;
        }

        .model-name {
            color: #0f172a;
            font-size: 1.35rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }

        .prediction-asd {
            color: #be123c;
            font-size: 2.1rem;
            font-weight: 800;
        }

        .prediction-tc {
            color: #15803d;
            font-size: 2.1rem;
            font-weight: 800;
        }

        .probability-label {
            color: #64748b;
            margin-top: 0.8rem;
        }

        .probability-value {
            color: #0f172a;
            font-size: 1.75rem;
            font-weight: 700;
        }

        .comparison-box {
            border-left: 4px solid #8b5cf6;
            background: rgba(139, 92, 246, 0.08);
            color: #334155;
            border-radius: 10px;
            padding: 1rem 1.2rem;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }

        .folder-summary {
            border: 1px solid rgba(96, 165, 250, 0.28);
            background: rgba(59, 130, 246, 0.08);
            color: #1e3a8a;
            border-radius: 14px;
            padding: 0.9rem 1rem;
            margin: 0.5rem 0 1rem 0;
        }

        .footer-text {
            text-align: center;
            color: #64748b;
            font-size: 0.85rem;
            padding-top: 1.5rem;
        }

        div.stButton > button {
            min-height: 3rem;
            border-radius: 12px;
            font-weight: 700;
        }

        div[data-testid="stFileUploader"] {
            border-radius: 14px;
        }

        /* File dari seluruh folder subjek tetap dibaca agar struktur bidang
           dapat dikenali, tetapi daftar mentahnya disembunyikan. Aplikasi
           menampilkan sendiri hanya lima citra yang benar-benar dipakai. */
        div[data-testid="stFileUploaderFile"] {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# KONFIGURASI MODEL
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

IMAGE_SIZE = (224, 224)
NUMBER_OF_SLICES = 5
THRESHOLD = 0.5
MINIMUM_STREAMLIT_VERSION = (1, 50)

PLANE_ORDER = ["Axial", "Coronal", "Sagittal"]
PLANE_LABELS = {
    "axial": "Axial",
    "coronal": "Coronal",
    "sagittal": "Sagittal",
}

MODEL_PATHS = {
    "Axial": {
        "Baseline": (
            MODEL_DIR
            / "best_vgg19_baseline_anat_axial_finetune_seed42_da516cee.keras"
        ),
        "CBAM": (
            MODEL_DIR
            / "best_vgg19_cbam_anat_axial_finetune_seed42_da516cee.keras"
        ),
    },
    "Coronal": {
        "Baseline": (
            MODEL_DIR
            / "best_vgg19_baseline_anat_coronal_frozen_seed42_a4a6a17d.keras"
        ),
        "CBAM": (
            MODEL_DIR
            / "best_vgg19_cbam_anat_coronal_finetune_seed42_a4a6a17d.keras"
        ),
    },
    "Sagittal": {
        "Baseline": (
            MODEL_DIR
            / "best_vgg19_baseline_anat_sagittal_frozen_seed42_fa0aa5ea.keras"
        ),
        "CBAM": (
            MODEL_DIR
            / "best_vgg19_cbam_anat_sagittal_finetune_seed42_fa0aa5ea.keras"
        ),
    },
}

CUSTOM_OBJECTS = {
    "CBAMBlock": CBAMBlock,
    "ANAT_ASD>CBAMBlock": CBAMBlock,
}


# FUNGSI BANTU
def get_major_minor_version(version_string):
    """Ambil pasangan angka mayor-minor tanpa menambah dependensi baru."""
    numbers = []

    for section in version_string.split(".")[:2]:
        digits = "".join(character for character in section if character.isdigit())
        numbers.append(int(digits) if digits else 0)

    while len(numbers) < 2:
        numbers.append(0)

    return tuple(numbers)


def natural_file_key(uploaded_file):
    """Urutkan nama file secara alami tanpa mewajibkan pola nama tertentu."""
    file_name = PurePosixPath(
        uploaded_file.name.replace("\\", "/")
    ).name

    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", file_name)
        if part
    )


def inspect_subject_folder(uploaded_files):
    """
    Membaca struktur folder unggahan berbentuk:

        ID_SUBJEK/
            axial/*.png
            coronal/*.png
            sagittal/*.png

    Nama ID diambil otomatis dari folder tepat sebelum nama bidang. Nama file
    citra sepenuhnya bebas; aplikasi hanya memeriksa ekstensi PNG, nama folder
    bidang, dan jumlah citra.
    """
    files_by_plane = defaultdict(list)
    subject_ids = set()
    ignored_files = []
    missing_subject_parent = []

    for uploaded_file in uploaded_files or []:
        relative_path = PurePosixPath(uploaded_file.name.replace("\\", "/"))
        lowered_parts = [part.lower() for part in relative_path.parts]

        plane_index = next(
            (
                index
                for index, part in enumerate(lowered_parts)
                if part in PLANE_LABELS
            ),
            None,
        )

        if plane_index is None:
            ignored_files.append(uploaded_file.name)
            continue

        if plane_index == 0:
            missing_subject_parent.append(uploaded_file.name)
            continue

        selected_plane = PLANE_LABELS[lowered_parts[plane_index]]
        subject_ids.add(relative_path.parts[plane_index - 1])
        files_by_plane[selected_plane].append(uploaded_file)

    for plane_files in files_by_plane.values():
        plane_files.sort(key=natural_file_key)

    errors = []
    warnings = []

    if missing_subject_parent:
        errors.append(
            "Kamu memilih folder bidang secara langsung. Pilih folder subjek "
            "yang berada satu tingkat di atas folder axial, coronal, dan "
            "sagittal agar ID subjek dapat dibaca otomatis."
        )
    elif not subject_ids:
        errors.append(
            "ID subjek tidak dapat dibaca. Pastikan struktur foldernya "
            "ID_SUBJEK/axial, ID_SUBJEK/coronal, dan ID_SUBJEK/sagittal."
        )
    elif len(subject_ids) > 1:
        joined_ids = ", ".join(sorted(subject_ids))
        errors.append(
            "Terdeteksi lebih dari satu subjek: "
            f"{joined_ids}. Pilih tepat satu folder subjek."
        )

    if ignored_files:
        warnings.append(
            f"{len(ignored_files)} file PNG di luar folder bidang diabaikan."
        )

    subject_id = next(iter(subject_ids)) if len(subject_ids) == 1 else None

    return {
        "subject_id": subject_id,
        "files_by_plane": dict(files_by_plane),
        "errors": errors,
        "warnings": warnings,
    }


# FUNGSI MODEL DAN CITRA
@st.cache_resource(show_spinner=False)
def load_model(model_path_string):
    return tf.keras.models.load_model(
        model_path_string,
        custom_objects=CUSTOM_OBJECTS,
        compile=False,
    )


def preprocess_image(uploaded_file):
    image_bytes = uploaded_file.getvalue()

    image = tf.io.decode_png(
        image_bytes,
        channels=3,
    )

    image = tf.cast(
        image,
        tf.float32,
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
        method="bilinear",
        antialias=True,
    )

    display_image = tf.cast(
        tf.clip_by_value(image, 0, 255),
        tf.uint8,
    ).numpy()

    model_input = preprocess_input(image)
    model_input = tf.ensure_shape(
        model_input,
        (224, 224, 3),
    )

    return display_image, model_input


def run_prediction(model, image_batch):
    slice_probabilities = (
        model.predict(
            image_batch,
            verbose=0,
        )
        .reshape(-1)
        .astype(float)
    )

    subject_probability = float(np.mean(slice_probabilities))
    predicted_class = "ASD" if subject_probability >= THRESHOLD else "TC"

    return {
        "slice_probabilities": slice_probabilities,
        "subject_probability": subject_probability,
        "predicted_class": predicted_class,
    }


def show_result_card(model_name, result):
    prediction = result["predicted_class"]
    probability_asd = result["subject_probability"]
    probability_tc = 1.0 - probability_asd

    prediction_class = (
        "prediction-asd" if prediction == "ASD" else "prediction-tc"
    )
    prediction_text = (
        "Autism Spectrum Disorder" if prediction == "ASD" else "Typical Control"
    )

    result_html = dedent(
        f"""
        <div class="result-card">
            <div class="model-label">MODEL</div>
            <div class="model-name">{escape(model_name)}</div>
            <div class="{prediction_class}">{prediction}</div>
            <div style="color:#475569; margin-bottom:0.7rem;">
                {prediction_text}
            </div>
            <div class="probability-label">Probabilitas keluaran ASD</div>
            <div class="probability-value">{probability_asd * 100:.2f}%</div>
            <div class="probability-label">
                Probabilitas keluaran TC: {probability_tc * 100:.2f}%
            </div>
        </div>
        """
    ).strip()

    st.markdown(
        result_html,
        unsafe_allow_html=True,
    )


# VALIDASI VERSI STREAMLIT
streamlit_version = get_major_minor_version(st.__version__)

if streamlit_version < MINIMUM_STREAMLIT_VERSION:
    st.error(
        "Aplikasi memerlukan Streamlit 1.50 atau lebih baru agar dapat "
        "memilih satu folder secara langsung."
    )
    st.code('pip install --upgrade "streamlit>=1.50,<2"', language="powershell")
    st.stop()


# RESET FOLDER UPLOADER
if "subject_uploader_version" not in st.session_state:
    st.session_state.subject_uploader_version = 0


def clear_uploaded_subject_folder():
    """Kosongkan seluruh unggahan dengan membuat ulang widget uploader."""
    st.session_state.subject_uploader_version += 1


uploader_version = st.session_state.subject_uploader_version
uploader_key = f"subject_folder_uploader_{uploader_version}"
plane_selector_key = f"plane_selector_{uploader_version}"
clear_button_key = f"clear_subject_folder_{uploader_version}"

# Hapus state widget versi lama agar file yang sudah dibuang tidak tertahan.
for state_key in list(st.session_state.keys()):
    is_old_uploader = (
        state_key.startswith("subject_folder_uploader_")
        and state_key != uploader_key
    )
    is_old_plane_selector = (
        state_key.startswith("plane_selector_")
        and state_key != plane_selector_key
    )
    is_old_clear_button = (
        state_key.startswith("clear_subject_folder_")
        and state_key != clear_button_key
    )

    if is_old_uploader or is_old_plane_selector or is_old_clear_button:
        del st.session_state[state_key]


# SIDEBAR DAN PEMILIHAN FOLDER
with st.sidebar:
    st.header("⚙️ Konfigurasi")

    folder_files = st.file_uploader(
        "Pilih folder subjek",
        type=["png"],
        accept_multiple_files="directory",
        help=(
            "Pilih folder seperti BNI_29006 yang berisi subfolder axial, "
            "coronal, dan sagittal. Nama file PNG di dalamnya boleh "
            "berbeda-beda."
        ),
        key=uploader_key,
    )

    if folder_files:
        st.button(
            "Hapus Semua File",
            key=clear_button_key,
            on_click=clear_uploaded_subject_folder,
            type="secondary",
            use_container_width=True,
            help="Menghapus seluruh file yang sudah diunggah sekaligus.",
        )

    folder_info = inspect_subject_folder(folder_files)
    subject_id = folder_info["subject_id"]
    files_by_plane = folder_info["files_by_plane"]

    if not folder_files:
        st.caption("Belum ada folder subjek yang dipilih.")
    else:
        for message in folder_info["errors"]:
            st.error(message)

        for message in folder_info["warnings"]:
            st.warning(message)

        if subject_id:
            st.success(f"ID subjek: {subject_id}")

    available_planes = [
        plane for plane in PLANE_ORDER if plane in files_by_plane
    ]

    selected_plane = None

    if available_planes and not folder_info["errors"]:
        selected_plane = st.selectbox(
            "Bidang irisan",
            options=available_planes,
            help="Daftar bidang dibaca otomatis dari folder subjek.",
            key=plane_selector_key,
        )

        st.markdown("**Isi folder terdeteksi**")

        for plane in PLANE_ORDER:
            file_count = len(files_by_plane.get(plane, []))
            status_icon = "✅" if file_count == NUMBER_OF_SLICES else "⚠️"
            st.caption(f"{status_icon} {plane}: {file_count} citra PNG")

    st.divider()
    st.subheader("Parameter")
    st.write("Jumlah irisan: **5**")
    st.write("Ukuran input: **224 × 224**")
    st.write("Ambang klasifikasi: **0,5**")
    st.write("Agregasi: **Rata-rata probabilitas**")

    st.divider()
    st.info(
        "Kedua model akan dibandingkan menggunakan lima citra yang sama."
    )


# HEADER
st.markdown(
    """
    <div class="main-title">🧠 MRI ASD Classification</div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Perbandingan VGG19 Baseline dan VGG19-CBAM untuk
        klasifikasi Autism Spectrum Disorder pada citra sMRI.
    </div>
    """,
    unsafe_allow_html=True,
)

subject_display = escape(subject_id) if subject_id else "Belum dipilih"
plane_display = selected_plane if selected_plane else "Belum dipilih"

st.markdown(
    f"""
    <div class="hero-card">
        <b>ID subjek:</b> {subject_display}<br>
        <b>Bidang terpilih:</b> {plane_display}<br>
        <b>Mode evaluasi:</b> Perbandingan Baseline dan CBAM<br>
        <b>Unit prediksi:</b> Tingkat subjek dari lima irisan ANAT
    </div>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "Aplikasi ini digunakan untuk keperluan penelitian dan "
    "tidak ditujukan sebagai alat diagnosis klinis."
)


# PEMBACAAN CITRA DARI FOLDER
st.subheader("1. Pilih Folder Subjek dan Bidang Irisan")

if not folder_files:
    st.info(
        "Klik 'Pilih folder subjek' pada sidebar, lalu pilih folder seperti "
        "BNI_29006 yang berisi subfolder axial, coronal, dan sagittal."
    )

elif folder_info["errors"]:
    st.info("Perbaiki pilihan folder pada sidebar untuk melanjutkan.")

elif not selected_plane:
    st.error(
        "Tidak ada folder bidang axial, coronal, atau sagittal yang terdeteksi."
    )

else:
    selected_files = files_by_plane.get(selected_plane, [])
    number_selected = len(selected_files)

    if number_selected != NUMBER_OF_SLICES:
        st.error(
            f"Folder {selected_plane.lower()} berisi {number_selected} citra PNG. "
            "Jumlah yang diperlukan tepat lima."
        )

    else:
        st.success(
            f"Subjek {subject_id} · bidang {selected_plane} · "
            "input model: 5 dari 5 citra."
        )

        preview_columns = st.columns(NUMBER_OF_SLICES)
        processed_images = []
        preprocessing_failed = False

        for index, uploaded_file in enumerate(selected_files):
            try:
                display_image, model_input = preprocess_image(uploaded_file)
                processed_images.append(model_input)

                with preview_columns[index]:
                    st.image(
                        display_image,
                        caption=f"Irisan {index + 1}",
                        use_container_width=True,
                    )
                    st.caption(PurePosixPath(uploaded_file.name).name)

            except Exception as error:
                preprocessing_failed = True
                st.error(f"File {uploaded_file.name} gagal diproses.")
                st.exception(error)

        if not preprocessing_failed:
            run_button = st.button(
                "🔍 Bandingkan Baseline dan CBAM",
                type="primary",
                use_container_width=True,
            )

            if run_button:
                baseline_path = MODEL_PATHS[selected_plane]["Baseline"]
                cbam_path = MODEL_PATHS[selected_plane]["CBAM"]

                missing_models = [
                    str(path)
                    for path in [baseline_path, cbam_path]
                    if not path.is_file()
                ]

                if missing_models:
                    st.error("Terdapat file model yang tidak ditemukan:")

                    for missing_model in missing_models:
                        st.code(missing_model)

                else:
                    image_batch = tf.stack(processed_images, axis=0)

                    try:
                        with st.spinner(
                            "Memuat kedua model dan menjalankan prediksi..."
                        ):
                            baseline_model = load_model(str(baseline_path))
                            cbam_model = load_model(str(cbam_path))

                            baseline_result = run_prediction(
                                baseline_model,
                                image_batch,
                            )
                            cbam_result = run_prediction(
                                cbam_model,
                                image_batch,
                            )

                        st.divider()
                        st.subheader("2. Hasil Perbandingan")

                        info_column_1, info_column_2 = st.columns(2)

                        with info_column_1:
                            st.metric("ID Subjek", subject_id)

                        with info_column_2:
                            st.metric("Bidang Irisan", selected_plane)

                        baseline_column, cbam_column = st.columns(2)

                        with baseline_column:
                            show_result_card(
                                "VGG19 Baseline",
                                baseline_result,
                            )

                        with cbam_column:
                            show_result_card(
                                "VGG19-CBAM",
                                cbam_result,
                            )

                        baseline_probability = baseline_result[
                            "subject_probability"
                        ]
                        cbam_probability = cbam_result["subject_probability"]
                        probability_difference = (
                            cbam_probability - baseline_probability
                        )

                        same_prediction = (
                            baseline_result["predicted_class"]
                            == cbam_result["predicted_class"]
                        )

                        comparison_message = (
                            "Kedua model menghasilkan kelas prediksi yang sama."
                            if same_prediction
                            else "Kedua model menghasilkan kelas prediksi yang berbeda."
                        )

                        st.markdown(
                            f"""
                            <div class="comparison-box">
                                <b>Ringkasan perbandingan</b><br><br>
                                {comparison_message}<br>
                                Selisih probabilitas ASD (CBAM − Baseline):
                                <b>{probability_difference * 100:+.2f}
                                poin persentase</b>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.subheader("3. Probabilitas Setiap Irisan")

                        result_table = pd.DataFrame(
                            {
                                "Irisan": [
                                    f"Irisan {index + 1}"
                                    for index in range(NUMBER_OF_SLICES)
                                ],
                                "Nama File": [
                                    PurePosixPath(uploaded_file.name).name
                                    for uploaded_file in selected_files
                                ],
                                "Baseline": baseline_result[
                                    "slice_probabilities"
                                ],
                                "CBAM": cbam_result["slice_probabilities"],
                            }
                        )

                        display_table = result_table.copy()
                        display_table["Baseline"] = display_table[
                            "Baseline"
                        ].map(lambda value: f"{value:.4f}")
                        display_table["CBAM"] = display_table["CBAM"].map(
                            lambda value: f"{value:.4f}"
                        )

                        st.dataframe(
                            display_table,
                            use_container_width=True,
                            hide_index=True,
                        )

                        st.subheader("4. Unduh Hasil")

                        export_table = result_table.copy()
                        export_table.insert(0, "ID Subjek", subject_id)
                        export_table.insert(1, "Bidang", selected_plane)
                        export_table["Prediksi Baseline"] = baseline_result[
                            "predicted_class"
                        ]
                        export_table[
                            "Probabilitas Subjek Baseline"
                        ] = baseline_probability
                        export_table["Prediksi CBAM"] = cbam_result[
                            "predicted_class"
                        ]
                        export_table[
                            "Probabilitas Subjek CBAM"
                        ] = cbam_probability
                        export_table["Waktu Prediksi"] = datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )

                        csv_data = export_table.to_csv(index=False).encode(
                            "utf-8-sig"
                        )

                        safe_subject_id = (
                            subject_id.replace(" ", "_")
                            .replace("/", "_")
                            .replace("\\", "_")
                        )

                        st.download_button(
                            label="Unduh Hasil CSV",
                            data=csv_data,
                            file_name=(
                                f"hasil_{safe_subject_id}_"
                                f"{selected_plane.lower()}.csv"
                            ),
                            mime="text/csv",
                            use_container_width=True,
                        )

                    except Exception as error:
                        st.error("Prediksi gagal dijalankan.")
                        st.exception(error)


# FOOTER
st.divider()

st.markdown(
    """
    <div class="footer-text">
        VGG19 Baseline dan VGG19-CBAM ·
        Citra ANAT axial, coronal, dan sagittal ·
        Prediksi tingkat subjek dari lima irisan tengah
    </div>
    """,
    unsafe_allow_html=True,
)
