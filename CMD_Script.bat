@echo off
setlocal enabledelayedexpansion

:: Step 1: Check for the script argument
set "SCRIPT_NAME=%1"
if "%SCRIPT_NAME%"=="" (
    echo ERROR: No python script specified. You need to specify a python script.
    exit /b 1
) else (
    echo Running
)

:: Step 2: Dynamically locate Anaconda
echo Searching for Anaconda installation in %LOCALAPPDATA%...

set "ANACONDA_PATH="
for /r "%LOCALAPPDATA%" %%D in (anaconda3/Scripts/activate.bat) do (
    if exist "%%D" (
        set "ANACONDA_PATH=%%~dpD.."
        goto :found
    )
)

:found
if "%ANACONDA_PATH%"=="" (
    echo Anaconda installation not found in %LOCALAPPDATA%! Ensure it is installed and accessible.
    exit /b 1
)

echo Anaconda found at: %ANACONDA_PATH%

:: Step 3: Add Anaconda to the PATH temporarily
set "PATH=%ANACONDA_PATH%\Scripts;%ANACONDA_PATH%\Library\bin;%PATH%"


:: Step 4: Activate the environments
:: Base Environment
@REM call conda init
call "%ANACONDA_PATH%\Scripts\activate.bat" base
if errorlevel 1 (
    echo ERROR: Failed to activate the base environment
    exit /b 1
)

:: Step 5: Check if a command line Argument for the environment was passed in
:: If none is passed default to the default (postprocessing)
set "ENV_NAME=%2"
if "%ENV_NAME%"=="" (
    echo No seperate environment specified, activating default environment ..postprocessing..
    call conda activate postprocessing
    if errorlevel 1 (
        echo ERROR: Failed to activate the postprocessing environment
        goto :cleanup
    )
) else (
    echo Activating environment "%ENV_NAME%"
    call conda activate "%ENV_NAME%"
    if errorlevel 1 (
        echo ERROR: Failed to activate the "%ENV_NAME%" environment
        goto :cleanup
    )
)

:: Step 6: Run the Python script
python "%SCRIPT_NAME%"
if errorlevel 1 (
    echo ERROR: Python Script execution failed 
    goto :cleanup
)

echo Python script executed successfully.

:cleanup
:: Step 7: Deactivate the environment
call conda deactivate
if errorlevel 1 (
    echo WARNING: Possible Issues to related to deactivating the environment!
    echo Please manually deactivate if necessary.
)

:: Step 8: Ensure we exit Anaconda and return to CMD
echo Exiting Script
exit /b 0

