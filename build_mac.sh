#!/bin/bash
cd "$(dirname "$0")"

# Activate virtual environment
# Assumes venv is created in the same directory or user has it ready
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Warning: venv/bin/activate not found. Assuming environment is already set up."
fi

echo "Installing PyInstaller..."
pip install pyinstaller

echo "Building App..."
# Hidden imports and collections
# startup_timeout is added to prevent timeout on slow machines
# nvidia is removed as it is not applicable for Mac
pyinstaller --noconfirm --clean \
 --onefile \
 --name "ReazonSpeechApp" \
 --hidden-import="sklearn.utils._typedefs" \
 --hidden-import="sklearn.neighbors._partition_nodes" \
 --hidden-import="sklearn.neighbors._quad_tree" \
 --hidden-import="sklearn.tree._utils" \
 --hidden-import="scipy.special.cython_special" \
 --hidden-import="scipy.spatial.transform._rotation_groups" \
 --collect-all="reazonspeech" \
 --collect-all="librosa" \
 --collect-all="resampy" \
 --collect-all="sherpa_onnx" \
 --collect-all="lightning_fabric" \
 --collect-all="pytorch_lightning" \
 reazon_app.py

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo "Build complete!"
echo "The App is in the 'dist' folder."
echo "Copying config.json..."
cp config.json dist/config.json

echo "Done."
