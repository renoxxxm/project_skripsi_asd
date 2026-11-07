from __future__ import annotations

import argparse
import gc
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input

try:
    from custom_layers import CBAMBlock
except ImportError as error:
    raise SystemExit(
        "custom_layers.py tidak ditemukan. Letakkan script ini di folder "
        "yang sama dengan app.py dan custom_layers.py."
    ) from error


BASE_DIR = Path(__file__).resolve().parent

DEFAULT_DATASET_ROOT = Path(
    r"D:\RISET2025\ANAT_3plane_5slice\slicesANAT_canonical_3plane5"
)
DEFAULT_MODELS_DIR = BASE_DIR / "models"
DEFAULT_OUTPUT_DIR = BASE_DIR / "scan_results"

IMAGE_SIZE = (224, 224)
NUMBER_OF_SLICES = 5
THRESHOLD = 0.5
PLANE_ORDER = ("axial", "coronal", "sagittal")
MODEL_ORDER = ("baseline", "cbam")

MODEL_FILENAMES = {
    "axial": {
        "baseline": (
            "best_vgg19_baseline_anat_axial_"
            "finetune_seed42_da516cee.keras"
        ),
        "cbam": (
            "best_vgg19_cbam_anat_axial_"
            "finetune_seed42_da516cee.keras"
        ),
    },
    "coronal": {
        "baseline": (
            "best_vgg19_baseline_anat_coronal_"
            "frozen_seed42_a4a6a17d.keras"
        ),
        "cbam": (
            "best_vgg19_cbam_anat_coronal_"
            "finetune_seed42_a4a6a17d.keras"
        ),
    },
    "sagittal": {
        "baseline": (
            "best_vgg19_baseline_anat_sagittal_"
            "frozen_seed42_fa0aa5ea.keras"
        ),
        "cbam": (
            "best_vgg19_cbam_anat_sagittal_"
            "finetune_seed42_fa0aa5ea.keras"
        ),
    },
}

CUSTOM_OBJECTS = {
    "CBAMBlock": CBAMBlock,
    "ANAT_ASD>CBAMBlock": CBAMBlock,
}


@dataclass(frozen=True)
class SubjectRecord:
    subject_id: str
    sub_id: str
    true_class: str
    subject_folder: Path
    paths_by_plane: dict[str, tuple[Path, ...]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cari subjek ASD dan TC yang diprediksi benar oleh Baseline dan "
            "CBAM pada axial, coronal, dan sagittal."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Folder yang berisi folder-folder subjek, misalnya BNI_29006.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help=(
            "CSV metadata. Jika tidak diisi, script mencari file "
            "metadata_ANAT_axial_5slice*.csv di folder script."
        ),
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Folder enam file model Keras.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder keluaran CSV.",
    )
    parser.add_argument(
        "--subject-list",
        type=Path,
        default=None,
        help=(
            "Opsional: CSV/TXT daftar subjek uji. CSV dapat memakai kolom "
            "unique_key, subject_id, atau SUB_ID."
        ),
    )
    parser.add_argument(
        "--subjects-per-batch",
        type=int,
        default=8,
        help="Jumlah subjek yang diproses sekaligus. Default: 8.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Jumlah kandidat teratas per kelas yang ditampilkan. Default: 10.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Opsional untuk uji cepat, misalnya --limit 10.",
    )
    return parser.parse_args()


def natural_sort_key(path: Path) -> tuple[tuple[int, object], ...]:
    """Urutkan nama file berdasarkan angka tanpa pola nama wajib."""
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", path.name)
        if part
    )


def normalize_identifier(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text


def locate_metadata(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path

    candidates = sorted(BASE_DIR.glob("metadata_ANAT_axial_5slice*.csv"))
    if not candidates:
        raise FileNotFoundError(
            "CSV metadata tidak ditemukan. Letakkan "
            "metadata_ANAT_axial_5slice*.csv di folder yang sama dengan "
            "script, atau gunakan parameter --metadata."
        )

    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        print(f"Ditemukan beberapa metadata: {names}")
        print(f"Digunakan: {candidates[0].name}")

    return candidates[0]


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata tidak ditemukan: {metadata_path}")

    metadata = pd.read_csv(metadata_path, low_memory=False)
    required_columns = {"unique_key", "SUB_ID", "class_name"}
    missing_columns = required_columns - set(metadata.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Kolom metadata tidak lengkap: {missing_text}")

    prepared = metadata.loc[:, ["unique_key", "SUB_ID", "class_name"]].copy()
    prepared["unique_key"] = prepared["unique_key"].map(normalize_identifier)
    prepared["SUB_ID"] = prepared["SUB_ID"].map(normalize_identifier)
    prepared["class_name"] = (
        prepared["class_name"].astype(str).str.strip().str.upper()
    )

    prepared = prepared[
        prepared["unique_key"].ne("")
        & prepared["class_name"].isin(["ASD", "TC"])
    ]

    conflicts = prepared.groupby("unique_key")["class_name"].nunique()
    conflicting_ids = conflicts[conflicts > 1].index.tolist()
    if conflicting_ids:
        preview = ", ".join(conflicting_ids[:10])
        raise ValueError(f"Label bertentangan ditemukan pada: {preview}")

    return (
        prepared.drop_duplicates(subset=["unique_key"])
        .rename(
            columns={
                "unique_key": "subject_id",
                "SUB_ID": "sub_id",
                "class_name": "true_class",
            }
        )
        .sort_values("subject_id")
        .reset_index(drop=True)
    )


def load_subject_scope(subject_list_path: Path | None) -> set[str] | None:
    if subject_list_path is None:
        return None

    if not subject_list_path.is_file():
        raise FileNotFoundError(
            f"Daftar subjek tidak ditemukan: {subject_list_path}"
        )

    if subject_list_path.suffix.lower() == ".csv":
        subject_list = pd.read_csv(subject_list_path, low_memory=False)
        preferred_columns = ("unique_key", "subject_id", "SUB_ID")
        selected_column = next(
            (column for column in preferred_columns if column in subject_list),
            None,
        )
        if selected_column is None:
            selected_column = subject_list.columns[0]
        values = subject_list[selected_column].tolist()
    else:
        values = subject_list.read_text(encoding="utf-8").splitlines()

    normalized = {normalize_identifier(value) for value in values}
    normalized.discard("")
    if not normalized:
        raise ValueError("Daftar subjek kosong.")
    return normalized


def apply_subject_scope(
    labels: pd.DataFrame,
    allowed_ids: set[str] | None,
) -> pd.DataFrame:
    if allowed_ids is None:
        return labels

    mask = labels["subject_id"].isin(allowed_ids) | labels["sub_id"].isin(
        allowed_ids
    )
    filtered = labels[mask].copy()
    if filtered.empty:
        raise ValueError(
            "Tidak ada ID pada daftar subjek yang cocok dengan metadata."
        )
    return filtered.reset_index(drop=True)


def find_plane_folder(subject_folder: Path, plane: str) -> Path | None:
    try:
        child_directories = {
            child.name.casefold(): child
            for child in subject_folder.iterdir()
            if child.is_dir()
        }
    except OSError:
        return None
    return child_directories.get(plane.casefold())


def inspect_dataset(
    labels: pd.DataFrame,
    dataset_root: Path,
) -> tuple[list[SubjectRecord], pd.DataFrame]:
    complete_subjects: list[SubjectRecord] = []
    skipped_rows: list[dict[str, str]] = []

    for row in labels.itertuples(index=False):
        subject_folder = dataset_root / row.subject_id
        if not subject_folder.is_dir():
            skipped_rows.append(
                {
                    "subject_id": row.subject_id,
                    "true_class": row.true_class,
                    "reason": "Folder subjek tidak ditemukan",
                }
            )
            continue

        paths_by_plane: dict[str, tuple[Path, ...]] = {}
        problems: list[str] = []

        for plane in PLANE_ORDER:
            plane_folder = find_plane_folder(subject_folder, plane)
            if plane_folder is None:
                problems.append(f"folder {plane} tidak ditemukan")
                continue

            image_paths = sorted(
                (
                    path
                    for path in plane_folder.iterdir()
                    if path.is_file() and path.suffix.casefold() == ".png"
                ),
                key=natural_sort_key,
            )

            if len(image_paths) != NUMBER_OF_SLICES:
                problems.append(
                    f"{plane} berisi {len(image_paths)} PNG, bukan 5"
                )
                continue

            paths_by_plane[plane] = tuple(image_paths)

        if problems:
            skipped_rows.append(
                {
                    "subject_id": row.subject_id,
                    "true_class": row.true_class,
                    "reason": "; ".join(problems),
                }
            )
            continue

        complete_subjects.append(
            SubjectRecord(
                subject_id=row.subject_id,
                sub_id=row.sub_id,
                true_class=row.true_class,
                subject_folder=subject_folder,
                paths_by_plane=paths_by_plane,
            )
        )

    return complete_subjects, pd.DataFrame(skipped_rows)


def validate_model_paths(models_dir: Path) -> dict[str, dict[str, Path]]:
    paths = {
        plane: {
            model_name: models_dir / MODEL_FILENAMES[plane][model_name]
            for model_name in MODEL_ORDER
        }
        for plane in PLANE_ORDER
    }

    missing = [
        path
        for plane_paths in paths.values()
        for path in plane_paths.values()
        if not path.is_file()
    ]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"File model tidak ditemukan:\n{formatted}")
    return paths


def configure_tensorflow() -> None:
    tf.get_logger().setLevel("ERROR")
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    device_text = ", ".join(device.name for device in gpus) or "CPU"
    print(f"Perangkat TensorFlow: {device_text}")


def preprocess_path(image_path: Path) -> tf.Tensor:
    image_bytes = tf.io.read_file(str(image_path))
    image = tf.io.decode_png(image_bytes, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(
        image,
        IMAGE_SIZE,
        method="bilinear",
        antialias=True,
    )
    image = preprocess_input(image)
    return tf.ensure_shape(image, (224, 224, 3))


def make_image_batch(
    subjects: list[SubjectRecord],
    plane: str,
) -> tf.Tensor:
    images = [
        preprocess_path(image_path)
        for subject in subjects
        for image_path in subject.paths_by_plane[plane]
    ]
    return tf.stack(images, axis=0)


def load_model(model_path: Path) -> tf.keras.Model:
    return tf.keras.models.load_model(
        str(model_path),
        custom_objects=CUSTOM_OBJECTS,
        compile=False,
    )


def scan_subjects(
    subjects: list[SubjectRecord],
    model_paths: dict[str, dict[str, Path]],
    subjects_per_batch: int,
) -> pd.DataFrame:
    if subjects_per_batch < 1:
        raise ValueError("--subjects-per-batch minimal 1.")

    result_rows = {
        subject.subject_id: {
            "subject_id": subject.subject_id,
            "sub_id": subject.sub_id,
            "true_class": subject.true_class,
            "subject_folder": str(subject.subject_folder),
        }
        for subject in subjects
    }

    total_subjects = len(subjects)

    for plane in PLANE_ORDER:
        print(f"\nMemuat model {plane}...")
        baseline_model = load_model(model_paths[plane]["baseline"])
        cbam_model = load_model(model_paths[plane]["cbam"])
        models = {
            "baseline": baseline_model,
            "cbam": cbam_model,
        }

        for start in range(0, total_subjects, subjects_per_batch):
            batch_subjects = subjects[start : start + subjects_per_batch]
            image_batch = make_image_batch(batch_subjects, plane)

            for model_name, model in models.items():
                slice_probabilities = np.asarray(
                    model.predict(image_batch, batch_size=32, verbose=0)
                ).reshape(-1)

                expected_count = len(batch_subjects) * NUMBER_OF_SLICES
                if len(slice_probabilities) != expected_count:
                    raise RuntimeError(
                        f"Output {plane} {model_name} berjumlah "
                        f"{len(slice_probabilities)}, seharusnya "
                        f"{expected_count}."
                    )

                subject_probabilities = slice_probabilities.reshape(
                    len(batch_subjects), NUMBER_OF_SLICES
                ).mean(axis=1)

                for subject, probability in zip(
                    batch_subjects,
                    subject_probabilities,
                ):
                    probability = float(probability)
                    prediction = "ASD" if probability >= THRESHOLD else "TC"
                    prefix = f"{plane}_{model_name}"
                    result_rows[subject.subject_id][
                        f"{prefix}_prob_asd"
                    ] = probability
                    result_rows[subject.subject_id][
                        f"{prefix}_prediction"
                    ] = prediction

            completed = min(start + len(batch_subjects), total_subjects)
            print(
                f"{plane.capitalize()}: {completed}/{total_subjects} subjek",
                end="\r" if completed < total_subjects else "\n",
                flush=True,
            )

        del baseline_model, cbam_model, models
        tf.keras.backend.clear_session()
        gc.collect()

    results = pd.DataFrame(result_rows.values())
    prediction_columns = [
        f"{plane}_{model_name}_prediction"
        for plane in PLANE_ORDER
        for model_name in MODEL_ORDER
    ]
    probability_columns = [
        f"{plane}_{model_name}_prob_asd"
        for plane in PLANE_ORDER
        for model_name in MODEL_ORDER
    ]

    results["all_six_agree"] = results[prediction_columns].nunique(axis=1).eq(1)
    results["all_six_correct"] = results[prediction_columns].apply(
        lambda row: bool((row == results.loc[row.name, "true_class"]).all()),
        axis=1,
    )

    probabilities = results[probability_columns].to_numpy(dtype=float)
    is_asd = results["true_class"].eq("ASD").to_numpy()[:, None]
    true_class_confidences = np.where(is_asd, probabilities, 1.0 - probabilities)
    results["minimum_true_class_confidence"] = true_class_confidences.min(axis=1)
    results["mean_true_class_confidence"] = true_class_confidences.mean(axis=1)

    ordered_columns = [
        "subject_id",
        "sub_id",
        "true_class",
        *[
            column
            for plane in PLANE_ORDER
            for model_name in MODEL_ORDER
            for column in (
                f"{plane}_{model_name}_prediction",
                f"{plane}_{model_name}_prob_asd",
            )
        ],
        "all_six_agree",
        "all_six_correct",
        "minimum_true_class_confidence",
        "mean_true_class_confidence",
        "subject_folder",
    ]
    return results[ordered_columns]


def save_results(
    results: pd.DataFrame,
    skipped: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)

    ranked_results = results.sort_values(
        [
            "all_six_correct",
            "minimum_true_class_confidence",
            "mean_true_class_confidence",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    candidates = (
        ranked_results[ranked_results["all_six_correct"]]
        .sort_values(
            [
                "true_class",
                "minimum_true_class_confidence",
                "mean_true_class_confidence",
            ],
            ascending=[True, False, False],
        )
        .reset_index(drop=True)
    )
    if not candidates.empty:
        candidates.insert(
            3,
            "rank_within_class",
            candidates.groupby("true_class").cumcount() + 1,
        )

    recommendations = (
        candidates[candidates["rank_within_class"].eq(1)].copy()
        if not candidates.empty
        else candidates.copy()
    )

    ranked_results.to_csv(
        output_dir / "all_subject_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    candidates.to_csv(
        output_dir / "demo_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    recommendations.to_csv(
        output_dir / "recommended_demo_subjects.csv",
        index=False,
        encoding="utf-8-sig",
    )
    skipped.to_csv(
        output_dir / "skipped_subjects.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return candidates, recommendations


def print_summary(
    results: pd.DataFrame,
    candidates: pd.DataFrame,
    recommendations: pd.DataFrame,
    skipped: pd.DataFrame,
    output_dir: Path,
    top_k: int,
) -> None:
    print("\nRingkasan")
    print(f"Subjek berhasil dipindai : {len(results)}")
    print(f"Subjek dilewati          : {len(skipped)}")
    print(f"Benar pada keenam model  : {len(candidates)}")

    for class_name in ("ASD", "TC"):
        class_candidates = candidates[candidates["true_class"].eq(class_name)]
        print(f"\nKandidat {class_name}: {len(class_candidates)}")

        if class_candidates.empty:
            print("Tidak ditemukan kandidat yang memenuhi seluruh syarat.")
            continue

        display_columns = [
            "rank_within_class",
            "subject_id",
            "minimum_true_class_confidence",
            "mean_true_class_confidence",
        ]
        print(
            class_candidates[display_columns]
            .head(top_k)
            .to_string(index=False)
        )

    print("\nRekomendasi utama")
    if recommendations.empty:
        print("Belum ada subjek ASD/TC yang benar pada keenam prediksi.")
    else:
        print(
            recommendations[
                [
                    "subject_id",
                    "true_class",
                    "minimum_true_class_confidence",
                    "mean_true_class_confidence",
                ]
            ].to_string(index=False)
        )

    print(f"\nSemua hasil disimpan di: {output_dir}")
    print("Buka file recommended_demo_subjects.csv terlebih dahulu.")


def main() -> int:
    args = parse_args()

    try:
        metadata_path = locate_metadata(args.metadata)
        dataset_root = args.dataset_root
        models_dir = args.models_dir
        output_dir = args.output_dir

        if not dataset_root.is_dir():
            raise FileNotFoundError(
                f"Folder dataset tidak ditemukan: {dataset_root}"
            )

        print(f"Dataset  : {dataset_root}")
        print(f"Metadata : {metadata_path}")
        print(f"Model    : {models_dir}")

        labels = load_metadata(metadata_path)
        allowed_ids = load_subject_scope(args.subject_list)
        labels = apply_subject_scope(labels, allowed_ids)

        if args.subject_list is None:
            print(
                "PERINGATAN: --subject-list tidak diberikan, sehingga semua "
                "subjek metadata akan dipindai. Untuk pemilihan yang dapat "
                "dipertanggungjawabkan sebagai data uji, gunakan daftar ID "
                "test dengan parameter --subject-list."
            )

        complete_subjects, skipped = inspect_dataset(labels, dataset_root)
        if args.limit is not None:
            if args.limit < 1:
                raise ValueError("--limit minimal 1.")
            complete_subjects = complete_subjects[: args.limit]

        if not complete_subjects:
            raise RuntimeError(
                "Tidak ada subjek lengkap dengan 5 PNG pada ketiga plane."
            )

        print(f"Subjek lengkap: {len(complete_subjects)}")
        print(f"Subjek dilewati: {len(skipped)}")

        model_paths = validate_model_paths(models_dir)
        configure_tensorflow()
        results = scan_subjects(
            complete_subjects,
            model_paths,
            args.subjects_per_batch,
        )
        candidates, recommendations = save_results(
            results,
            skipped,
            output_dir,
        )
        print_summary(
            results,
            candidates,
            recommendations,
            skipped,
            output_dir,
            args.top_k,
        )
        return 0

    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
