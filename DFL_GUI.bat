@echo off
call "%~dp0setenv.bat"
echo Launching DeepFaceLab PyTorch GUI...
"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\run_gui.py"
if errorlevel 1 (
    echo.
    echo GUI failed to start. Use CLI mode instead:
    echo   python DeepFaceLab\run.py --help
    pause
)
