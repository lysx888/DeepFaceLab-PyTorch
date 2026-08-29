@echo off
rem ========== BASE ENV ==========
SET ROOT_DIR=%~dp0
SET ROOT_DIR=%ROOT_DIR:~0,-1%

rem ========== MSVC ENV (for gsplat JIT compile) ==========
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

rem overriding windows user/local environment
SET LOCALENV_DIR=%ROOT_DIR%\_local
SET TMP=%LOCALENV_DIR%\t
SET TEMP=%LOCALENV_DIR%\t
SET HOME=%LOCALENV_DIR%\u
SET HOMEPATH=%LOCALENV_DIR%\u
SET USERPROFILE=%LOCALENV_DIR%\u
SET LOCALAPPDATA=%USERPROFILE%\AppData\Local
SET APPDATA=%USERPROFILE%\AppData\Roaming

rem ========== PYTHON ENV ==========
SET PYTHON_PATH=%ROOT_DIR%\python
SET PYTHONHOME=
SET PYTHONPATH=
SET PYTHONEXECUTABLE=%PYTHON_PATH%\python.exe
SET PYTHONWEXECUTABLE=%PYTHON_PATH%\pythonw.exe
SET PYTHON_EXECUTABLE=%PYTHON_PATH%\python.exe
SET PYTHONW_EXECUTABLE=%PYTHON_PATH%\pythonw.exe
SET PYTHON_BIN_PATH=%PYTHON_EXECUTABLE%
SET PYTHON_LIB_PATH=%PYTHON_PATH%\Lib\site-packages

rem ========== CUDA ENV (from PyTorch) ==========
SET TORCH_LIB_PATH=%PYTHON_LIB_PATH%\torch\lib
SET PATH=%TORCH_LIB_PATH%;%PYTHON_PATH%;%PYTHON_PATH%\Scripts;%PATH%

rem ========== MSVC Redist (for CUDA JIT compiled .pyd) ==========
rem vcruntime140.dll is in VC145.CRT (not VC143), must be on PATH
SET MSVC_REDIST_X64=%ProgramFiles(x86)%\Microsoft Visual Studio\18\BuildTools\VC\Redist\MSVC\14.51.36231\x64\Microsoft.VC145.CRT
SET PATH=%MSVC_REDIST_X64%;%PATH%

rem ========== DLL SEARCH PATH ==========
rem Embedded Python needs os.add_dll_directory for torch
SET DFL_DLL_DIRS=%TORCH_LIB_PATH%

rem CUDA memory allocation - reduce fragmentation
SET PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync,max_split_size_mb:128

rem Suppress ONNX Runtime verbose output (3=ERROR level)
SET ORT_LOGGING_LEVEL=3

rem PyTorch thread control - set by configure_torch() per task
rem Do NOT set OMP/MKL here; each module sets its own value
rem DataLoader workers: 1 thread each (set by worker_init_fn)

rem Force UTF-8 mode for triton/torch.compile on Chinese Windows (fixes GBK UnicodeDecodeError in inductor)
SET PYTHONUTF8=1

rem ========== ADDITIONAL ENV ==========
SET FFMPEG_PATH=%ROOT_DIR%\ffmpeg
SET PATH=%FFMPEG_PATH%;%PATH%

rem ========== PROJECT ENV ==========
SET WORKSPACE=%ROOT_DIR%\..\workspace
SET DFL_ROOT=%ROOT_DIR%

rem create local dirs
if not exist "%LOCALENV_DIR%\t" mkdir "%LOCALENV_DIR%\t"
if not exist "%LOCALENV_DIR%\u" mkdir "%LOCALENV_DIR%\u"
if not exist "%LOCALENV_DIR%\u\Desktop" mkdir "%LOCALENV_DIR%\u\Desktop"