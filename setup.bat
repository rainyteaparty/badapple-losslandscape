@echo off
REM Create the venv and check system deps. Safe to re-run.
setlocal EnableDelayedExpansion
cd /d "%~dp0"

REM ---- conda check --------------------------------------------------------
REM A venv built from Anaconda's python inherits conda's DLL environment.
REM torch then loads two Intel OpenMP runtimes and dies with WinError 1114
REM ("DLL initialization routine failed") on import.
if defined CONDA_PREFIX (
  echo.
  echo WARNING: a conda environment is active ^(%CONDA_DEFAULT_ENV%^).
  echo          Building the venv from conda's python usually breaks torch with
  echo          WinError 1114 on import. Recommended:
  echo.
  echo            conda deactivate
  echo            rmdir /s /q .venv
  echo            setup.bat
  echo.
  echo          ...from a plain cmd/PowerShell, using a python.org install.
  echo          Continuing anyway in 5 seconds; Ctrl-C to stop.
  timeout /t 5 >nul
)

REM ---- python -------------------------------------------------------------
set "PY=python"
where python >nul 2>&1
if errorlevel 1 (
  set "PY=py -3"
  py -3 --version >nul 2>&1
  if errorlevel 1 (
    echo ERROR: no Python on PATH. Install from python.org, tick "Add to PATH".
    exit /b 1
  )
)
for /f "delims=" %%v in ('%PY% --version 2^>^&1') do echo python:   %%v

REM ---- ffmpeg -------------------------------------------------------------
where ffmpeg >nul 2>&1
if errorlevel 1 (
  echo MISSING: ffmpeg -- winget install Gyan.FFmpeg   ^(or: scoop install ffmpeg^)
  exit /b 1
)
where ffprobe >nul 2>&1
if errorlevel 1 (
  echo MISSING: ffprobe -- it ships with ffmpeg; check your PATH
  exit /b 1
)
echo ffmpeg:   found
echo ffprobe:  found

REM ---- venv ---------------------------------------------------------------
if not exist ".venv" (
  echo creating .venv
  %PY% -m venv .venv
  if errorlevel 1 exit /b 1
)
set "VPY=.venv\Scripts\python.exe"
if not exist "%VPY%" (
  echo ERROR: %VPY% missing. A venv is not portable between platforms --
  echo        if this folder came from macOS/Linux: rmdir /s /q .venv ^&^& setup.bat
  exit /b 1
)
"%VPY%" -m pip install --quiet --upgrade pip

REM ---- torch: the Windows PyPI wheel is CPU-only, unlike Linux -------------
where nvidia-smi >nul 2>&1
if errorlevel 1 (
  echo torch:    no nvidia-smi found, installing the CPU build
) else (
  echo torch:    NVIDIA GPU detected, installing the CUDA build ^(large download^)
  "%VPY%" -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cu124
  if errorlevel 1 (
    echo ERROR: CUDA torch install failed. Pick the index matching your driver at
    echo        https://pytorch.org/get-started/locally/  then re-run setup.bat
    exit /b 1
  )
)
"%VPY%" -m pip install --quiet -r requirements.txt
if errorlevel 1 (
  echo ERROR: pip install failed
  exit /b 1
)

REM ---- device -------------------------------------------------------------
"%VPY%" -c "import torch; from net import pick_device; d = pick_device(); tail = ('  (' + torch.cuda.get_device_name(0) + ', CUDA ' + str(torch.version.cuda) + ')') if d == 'cuda' else ('  -- no GPU found; training will be slow' if d == 'cpu' else ''); print('torch ' + torch.__version__ + ', device ' + d + tail)"
if errorlevel 1 (
  echo.
  echo ERROR: torch failed to import. Most likely causes, in order:
  echo   1. a conda env was active when .venv was built -- see the warning above
  echo   2. missing Visual C++ runtime -- install "Microsoft Visual C++
  echo      Redistributable for Visual Studio 2015-2022 x64" from microsoft.com
  echo   3. a 32-bit Python -- torch is x64 only
  exit /b 1
)

REM ---- source clip --------------------------------------------------------
if not exist "video\out" mkdir "video\out"
set "CLIP="
for %%F in (video\*.mp4 video\*.mkv video\*.webm video\*.mov video\*.avi) do (
  if not defined CLIP set "CLIP=%%F"
)
echo.
if not defined CLIP (
  echo Next: drop the source clip in  video\   ^(the first one found is used^)
) else (
  echo source clip: !CLIP!
)
echo.
echo Run without activating anything:
echo   .venv\Scripts\python generate.py --loss all
echo   .venv\Scripts\python generate_layers.py
echo   .venv\Scripts\python serve.py
echo.
echo Or activate first ^(note: NOT the same path as on macOS/Linux^):
echo   cmd:         .venv\Scripts\activate.bat
echo   PowerShell:  .venv\Scripts\Activate.ps1
echo                ^(if blocked: Set-ExecutionPolicy -Scope Process Bypass^)
endlocal
