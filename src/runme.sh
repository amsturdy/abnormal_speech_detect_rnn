#!/bin/bash
DEVTEST_DIR="data/mixture_data/devtest"
DEVTRAIN_DIR="data/mixture_data/devtrain"

FEATURES_TESTING_DIR="features/logmel/testing"
FEATURES_TRAINING_DIR="features/logmel/training"

H5_TESTING_PATH="packed_features/logmel/testing.h5"
H5_TRAINING_PATH="packed_features/logmel/training.h5"

SCALER_PATH="scalers/logmel/training.scaler"

:<<not_run

# synthesizer mixtures
python src/synthesizer/generate_mixtures.py

# Extract features
python src/prepare_data.py extract_features --wav_dir=$DEVTEST_DIR"/audio" --out_dir=$FEATURES_TESTING_DIR --recompute=True
python src/prepare_data.py extract_features --wav_dir=$DEVTRAIN_DIR"/audio" --out_dir=$FEATURES_TRAINING_DIR --recompute=True

# Pack features
python src/prepare_data.py pack_features --fe_dir=$FEATURES_TESTING_DIR --yaml_dir=$DEVTEST_DIR --out_path=$H5_TESTING_PATH
python src/prepare_data.py pack_features --fe_dir=$FEATURES_TRAINING_DIR --yaml_dir=$DEVTRAIN_DIR --out_path=$H5_TRAINING_PATH

# Calculate scaler
python src/prepare_data.py calculate_scaler --hdf5_path=$H5_TRAINING_PATH --out_path=$SCALER_PATH
not_run

# Train AED
python src/audio_event_detect.py train --tr_hdf5_path=$H5_TRAINING_PATH --te_hdf5_path=$H5_TESTING_PATH --scaler_path=$SCALER_PATH --out_model_dir="models/crnn"

# Recognize AED
python src/audio_event_detect.py recognize --te_hdf5_path=$H5_TESTING_PATH --scaler_path=$SCALER_PATH --model_dir="models/crnn" --yaml_dir=$DEVTEST_DIR


