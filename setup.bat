@echo off
REM Create the venv and check system deps. Safe to re-run.
setlocal EnableDelayedExpansion
cd /d "%~dp0"

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
  echo ERROR: %VPY% missing -- delete .venv and re-run
  exit /b 1
)
"%VPY%" -m pip install --quiet --upgrade pip
"%VPY%" -m pip install --quiet -r requirements.txt
if errorlevel 1 (
  echo ERROR: pip install failed
  exit /b 1
)

REM ---- device -------------------------------------------------------------
"%VPY%" -c "import torch; from net import pick_device; d = pick_device(); tail = ('  (' + torch.cuda.get_device_name(0) + ', CUDA ' + str(torch.version.cuda) + ')') if d == 'cuda' else ('  -- no GPU found; training will be slow' if d == 'cpu' else ''); print('torch ' + torch.__version__ + ', device ' + d + tail)"

REM ---- source clip --------------------------------------------------------
if not exist "video\out" mkdir "video\out"
set "CLIP="
for %%F in (video\*.mp4 video\*.mkv video\*.webm video\*.mov video\*.avi) do (
  if not defined CLIP set "CLIP=%%F"
)
echo.
if not defined CLIP (
  echo Next: drop the source clip in  video\   ^(the first one found is used^)
  echo Then:
  echo   .venv\Scripts\python generate.py --loss all
) else (
  echo source clip: !CLIP!
  echo next:  .venv\Scripts\python generate.py --loss all
)
endlocal
