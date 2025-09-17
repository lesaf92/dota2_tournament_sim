#!/bin/bash

# ==============================================================================
# Dota 2 ML Project Orchestrator
# ==============================================================================
# This script automates the data generation and model training process.
# It checks if required files already exist and skips steps to save time.
# All output is logged to a file for debugging.

# --- Configuration ---
# Set the names of your scripts, files, and virtual environment directory.
VENV_DIR="/home/luiz/dota2leaguesim"

TEAMS_SCRIPT="teams.py"
DATA_GEN_SCRIPT="data_generator.py"  # Assumes your multi-rating script is named this
TRAIN_SCRIPT="train.py"
APP_SCRIPT="app.py"

TEAMS_FILE="teams.json"
DATASET_FILE="dota2_multi_rating_dataset.csv"
MODEL_FILE="dota2_predictor.pt"
LOG_FILE="execution_$(date +%Y-%m-%d_%H-%M-%S).log"
cd $VENV_DIR
# --- Script Start ---

# Initialize Log File
echo "Starting project execution at $(date)" > "$LOG_FILE"
echo "--------------------------------------------------" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# 1. Activate Virtual Environment
echo "STEP 1: Activating Python virtual environment..." >> "$LOG_FILE"
if [ -d "$VENV_DIR" ]; then
    source "bin/activate"
    echo "Virtual environment activated." >> "$LOG_FILE"
else
    echo "ERROR: Virtual environment directory '$VENV_DIR' not found." >> "$LOG_FILE"
    echo "Please create the virtual environment first." >> "$LOG_FILE"
    exit 1
fi
echo "" >> "$LOG_FILE"


# 2. Generate Dataset (if it doesn't exist)
echo "STEP 2: Checking for dataset file..." >> "$LOG_FILE"
if [ ! -f "$TEAMS_FILE" ]; then
    echo "-> Teams file ('$TEAMS_FILE') not found." >> "$LOG_FILE"
    echo "-> Running teams generation script: '$TEAMS_SCRIPT'" >> "$LOG_FILE"
    echo "------------------ SCRIPT OUTPUT START -------------------" >> "$LOG_FILE"

    # Execute the script and append all output (stdout & stderr) to the log
    python "$TEAMS_SCRIPT" >> "$LOG_FILE" 2>&1

    echo "------------------- SCRIPT OUTPUT END --------------------" >> "$LOG_FILE"
    echo "-> Teams generation script finished." >> "$LOG_FILE"
else
    echo "-> Teams file ('$TEAMS_FILE') already exists. Skipping teams generation." >> "$LOG_FILE"
fi
echo "" >> "$LOG_FILE"
if [ ! -f "$DATASET_FILE" ]; then
    echo "-> Dataset file ('$DATASET_FILE') not found." >> "$LOG_FILE"
    echo "-> Running data generation script: '$DATA_GEN_SCRIPT'" >> "$LOG_FILE"
    echo "------------------ SCRIPT OUTPUT START -------------------" >> "$LOG_FILE"

    # Execute the script and append all output (stdout & stderr) to the log
    python "$DATA_GEN_SCRIPT" >> "$LOG_FILE" 2>&1

    echo "------------------- SCRIPT OUTPUT END --------------------" >> "$LOG_FILE"
    echo "-> Data generation script finished." >> "$LOG_FILE"
else
    echo "-> Dataset file ('$DATASET_FILE') already exists. Skipping data generation." >> "$LOG_FILE"
fi
echo "" >> "$LOG_FILE"


# 3. Train Model (if the model file doesn't exist)
echo "STEP 3: Checking for model file..." >> "$LOG_FILE"
if [ ! -f "$MODEL_FILE" ]; then
    echo "-> Model file ('$MODEL_FILE') not found." >> "$LOG_FILE"
    echo "-> Running model training script: '$TRAIN_SCRIPT'" >> "$LOG_FILE"
    echo "------------------ SCRIPT OUTPUT START -------------------" >> "$LOG_FILE"

    # Execute the script and append all output to the log
    python "$TRAIN_SCRIPT" >> "$LOG_FILE" 2>&1

    echo "------------------- SCRIPT OUTPUT END --------------------" >> "$LOG_FILE"
    echo "-> Model training script finished." >> "$LOG_FILE"
else
    echo "-> Model file ('$MODEL_FILE') already exists. Skipping model training." >> "$LOG_FILE"
fi
echo "" >> "$LOG_FILE"


# 4. Launch web app
echo "STEP 4: Launching webapp application..." >> "$LOG_FILE"
echo "--------------------------------------------------" >> "$LOG_FILE"
echo "The script has completed the setup and training checks." >> "$LOG_FILE"
echo "The webserver will now start in the terminal." >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Notify the user in the console
echo "Setup complete. Log file created at '$LOG_FILE'."
echo "Launching the webapp now..."

# The output of Streamlit itself will go to the console, not the log file,
# so you can see the URL and server status.
python "$APP_SCRIPT"
