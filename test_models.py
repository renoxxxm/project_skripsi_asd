from pathlib import Path
import gc

import tensorflow as tf

from custom_layers import CBAMBlock


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_FILES = [
    "best_vgg19_baseline_anat_axial_finetune_seed42_da516cee.keras",
    "best_vgg19_cbam_anat_axial_finetune_seed42_da516cee.keras",
    "best_vgg19_baseline_anat_coronal_frozen_seed42_a4a6a17d.keras",
    "best_vgg19_cbam_anat_coronal_finetune_seed42_a4a6a17d.keras",
    "best_vgg19_baseline_anat_sagittal_frozen_seed42_fa0aa5ea.keras",
    "best_vgg19_cbam_anat_sagittal_finetune_seed42_fa0aa5ea.keras",
]

CUSTOM_OBJECTS = {
    "CBAMBlock": CBAMBlock,
    "ANAT_ASD>CBAMBlock": CBAMBlock,
}


print("=" * 70)
print("PENGUJIAN FILE MODEL")
print("=" * 70)

for index, filename in enumerate(MODEL_FILES, start=1):
    model_path = MODEL_DIR / filename

    print(f"\n[{index}/6] {filename}")

    if not model_path.is_file():
        print("STATUS: FILE TIDAK DITEMUKAN")
        continue

    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=CUSTOM_OBJECTS,
            compile=False,
        )

        print("STATUS      : BERHASIL")
        print("INPUT SHAPE :", model.input_shape)
        print("OUTPUT SHAPE:", model.output_shape)

        del model
        tf.keras.backend.clear_session()
        gc.collect()

    except Exception as error:
        print("STATUS: GAGAL")
        print("ERROR :", error)

print("\n" + "=" * 70)
print("PENGUJIAN SELESAI")
print("=" * 70)