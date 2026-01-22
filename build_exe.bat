@echo off
cd /d %~dp0
call venv\Scripts\activate

echo Installing PyInstaller...
pip install pyinstaller

echo Building EXE...
:: Hidden imports for librosa/sklearn/scipy stack and potential missing modules
pyinstaller --noconfirm --clean ^
 --onefile ^
 --name "ReazonSpeechApp" ^
 --hidden-import="sklearn.utils._typedefs" ^
 --hidden-import="sklearn.neighbors._partition_nodes" ^
 --hidden-import="sklearn.neighbors._quad_tree" ^
 --hidden-import="sklearn.tree._utils" ^
 --hidden-import="scipy.special.cython_special" ^
 --hidden-import="scipy.spatial.transform._rotation_groups" ^
 --collect-all="reazonspeech" ^
 --collect-all="librosa" ^
 --collect-all="resampy" ^
 --collect-all="sherpa_onnx" ^
 --collect-all="nvidia" ^
 reazon_app.py

if %ERRORLEVEL% NEQ 0 (
    echo Build failed!
    pause
    exit /b %ERRORLEVEL%
)

echo Build complete!
echo The EXE is in the 'dist' folder.
echo Copy config.json to the 'dist' folder before running!
copy config.json dist\config.json

pause
