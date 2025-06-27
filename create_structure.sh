#!/bin/bash
# chmod +x create_structure.sh

BASE_DIR="/home/woo/KUiotFinalProject/posture-web-analyzer"

mkdir -p $BASE_DIR/app/analyzer
mkdir -p $BASE_DIR/app/camera
mkdir -p $BASE_DIR/app/models
mkdir -p $BASE_DIR/static
mkdir -p $BASE_DIR/templates
mkdir -p $BASE_DIR/test

touch $BASE_DIR/app/__init__.py
touch $BASE_DIR/app/main.py

touch $BASE_DIR/app/analyzer/__init__.py
touch $BASE_DIR/app/analyzer/posture.py
touch $BASE_DIR/app/analyzer/utils.py
touch $BASE_DIR/app/analyzer/draw.py

touch $BASE_DIR/app/camera/__init__.py
touch $BASE_DIR/app/camera/webcam.py

touch $BASE_DIR/app/models/__init__.py
touch $BASE_DIR/app/models/posture_result.py

touch $BASE_DIR/app/config.py

touch $BASE_DIR/static/script.js
touch $BASE_DIR/static/style.css

touch $BASE_DIR/templates/index.html

mkdir -p $BASE_DIR/test
touch $BASE_DIR/test/test_analyzer.py

touch $BASE_DIR/requirements.txt
touch $BASE_DIR/README.md
touch $BASE_DIR/run.sh
