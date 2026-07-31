@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
rem gears2 drag-and-drop compressor. Drop a .safetensors model onto this file.
rem Requires: python with  pip install torch gguf safetensors numpy
if "%~1"=="" (
  echo Drag a .safetensors model onto this file to compress to Q2_K.
  pause
  exit /b
)
echo === gears2 Q2_K ===
python "%~dp0xquant_tool.py" "%~1" Q2_K
echo.
echo Done. Compressed .gguf is next to the source.
pause
