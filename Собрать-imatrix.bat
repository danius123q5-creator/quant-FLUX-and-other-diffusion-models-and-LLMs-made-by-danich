@echo off
chcp 65001 >nul
REM ── Собрать imatrix для FLUX (activation-aware) — двойной клик ──
REM Нужен CUDA-torch + diffusers. Берём python_embeded от ComfyUI-portable,
REM иначе — системный python (должен иметь torch+diffusers).
setlocal
set "SCRIPT=%~dp0collect_imatrix.py"
set "PY="
for %%P in (
  "D:\ComfyBot\comfyui_portable\ComfyUI_windows_portable\python_embeded\python.exe"
  "%~dp0..\comfyui_portable\ComfyUI_windows_portable\python_embeded\python.exe"
  "%~dp0ComfyUI_windows_portable\python_embeded\python.exe"
) do if exist %%~P set "PY=%%~P"
if "%PY%"=="" set "PY=python"
echo Использую python: %PY%
"%PY%" -u "%SCRIPT%" %*
echo.
echo Готово. Файл .imatrix.npy рядом. Укажи его в gears2.exe (поле imatrix).
pause
