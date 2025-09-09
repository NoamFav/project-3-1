#!/bin/bash

echo "🚀 Setting up EDMO Project folder structure..."

# Base src structure
mkdir -p src/{go_core,cpp_native,python_nlp,data_pipeline,shared,interfaces}

# Go files
touch src/go_core/{main.go,pipeline.go,video.go,audio.go,utils.go}
cd src/go_core && go mod init edmo-pipeline && cd -

# C++ files
touch src/cpp_native/{whisper_runner.cpp,whisper_runner.hpp,video_tools.cpp,video_tools.hpp,CMakeLists.txt}

# Python microservice
touch src/python_nlp/{app.py,processor.py,strategies.py,requirements.txt}

# Data pipeline scripts
touch src/data_pipeline/{extract_frames.sh,convert_audio.sh,preprocess.py}
chmod +x src/data_pipeline/*.sh

# Shared config/schemas
touch src/shared/{schema.json,config.yaml,constants.go}

# API and data format docs
touch src/interfaces/{nlp_api.md,data_format.md}

echo "✅ Folder structure created."
echo "📦 Go module initialized."
echo "📁 You're ready to build the pipeline."