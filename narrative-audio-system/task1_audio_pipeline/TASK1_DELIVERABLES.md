# Task 1 Deliverables: Audio Processing Pipeline

This folder contains the implementation and outputs for Task 1.

## 1) Python code for audio processing

Implementation file:
- `task1_audio_pipeline/audio_pipeline.py`

The script performs:
- Audio loading from a directory of `.wav` files
- Peak normalization (`librosa.util.normalize`)
- Optional fixed-window segmentation (`--segment-seconds`)
- Feature extraction for each segment
- Export of a structured CSV dataset suitable for ML

## 2) Description of extracted features

Per segment, the output dataset includes:
- `filename`: source audio file name
- `segment_index`: index of segment within the source file
- `segment_start_seconds`: segment start timestamp
- `segment_end_seconds`: segment end timestamp
- `pitch_mean_hz`: mean pitch from `librosa.yin` (Hz)
- `spectral_centroid_mean_hz`: mean spectral centroid (Hz)
- `energy_rms_mean`: mean RMS energy
- `duration_seconds`: segment duration in seconds
- `mfcc_1 ... mfcc_13`: mean MFCC coefficients

## 3) Example output dataset

Example dataset file:
- `examples/task1_features_dataset_sample.csv`

To generate a full dataset from all available `.wav` files:

```bash
python task1_audio_pipeline/audio_pipeline.py \
  --input-dir ../examples \
  --output-csv ../examples/task1_features_dataset.csv \
  --normalized-dir ../examples/normalized_audio \
  --segment-seconds 2.0
```

For a small quick-run sample:

```bash
python task1_audio_pipeline/audio_pipeline.py \
  --input-dir ../examples \
  --output-csv ../examples/task1_features_dataset_sample.csv \
  --normalized-dir ../examples/normalized_audio_sample \
  --segment-seconds 2.0 \
  --max-files 5
```
