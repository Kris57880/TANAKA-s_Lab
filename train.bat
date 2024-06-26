@echo off
setlocal enabledelayedexpansion

set /a count=0
set /a max_retries=10



python line_notify.py "Start Training"
python main.py
if %errorlevel% neq 0 (
    echo main.py crashed with exit code %errorlevel%. Restarting...
    python line_notify.py "Error during training, Restarting... "
    set /a count+=1
    echo Waiting for 30 seconds... Press CTRL+C to stop.
    
    REM Wait for 30 seconds with the ability to interrupt using CTRL+C
    choice /c YN /n /d Y /t 30 >nul
    if %errorlevel% neq 1 (
        call :terminate
    )
    
    goto loop
)

echo Training completed successfully.
exit /b 0
REM Function to handle script termination
:terminate
echo.
echo Process interrupted by user. Exiting...
python line_notify.py "Training interrupted by user."
exit /b 1

REM Main loop
:loop
if !count! geq %max_retries% (
    echo Maximum retries reached. Exiting...
    python line_notify.py "Maximum retries reached. Training stopped."
    exit /b 1
)