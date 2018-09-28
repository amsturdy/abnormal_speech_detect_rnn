#!/bin/bash
DEVTEST_DIR="data/mixture_data/devtest"
DEVTRAIN_DIR="data/mixture_data/devtrain"

FEATURES_TESTING_DIR="features/testing"
FEATURES_TRAINING_DIR="features/training"

H5_TESTING_PATH="h5/testing.h5"
H5_TRAINING_PATH="h5/training.h5"

SCALER_PATH="scalers/training.scaler"

:<<not_run

# synthesizer mixtures
python src/synthesizer/generate_mixtures.py --distribute=True
not_run

# Extract features
python -m pdb src/prepare_data.py extract_features --wav_dir=$DEVTEST_DIR"/audio" --out_dir=$FEATURES_TESTING_DIR --recompute=True
python src/prepare_data.py extract_features --wav_dir=$DEVTRAIN_DIR"/audio" --out_dir=$FEATURES_TRAINING_DIR --recompute=True

# Features2h5
python src/prepare_data.py features2h5 --fe_dir=$FEATURES_TESTING_DIR --yaml_dir=$DEVTEST_DIR --out_path=$H5_TESTING_PATH
python src/prepare_data.py features2h5 --fe_dir=$FEATURES_TRAINING_DIR --yaml_dir=$DEVTRAIN_DIR --out_path=$H5_TRAINING_PATH

# Calculate scaler
python src/prepare_data.py calculate_scaler --hdf5_path=$H5_TRAINING_PATH --out_path=$SCALER_PATH

# Train
python src/audio_event_detect.py train --tr_hdf5_path=$H5_TRAINING_PATH --te_hdf5_path=$H5_TESTING_PATH --scaler_path=$SCALER_PATH --out_model_dir="models"

# detect
python src/audio_event_detect.py detect --te_hdf5_path=$H5_TESTING_PATH --scaler_path=$SCALER_PATH --model_dir="models" --yaml_dir=$DEVTEST_DIR

# evaluate
python src/evaluate.py --param "all"
