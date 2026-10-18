@echo off
setlocal enabledelayedexpansion

rem ===========================================================================
rem  IFAN defocus deblurring - drop-zone runner
rem
rem  EASIEST USE - no arguments at all:
rem      Put images, or whole folders of images, into the "input" folder that
rem      sits next to this file, then double-click this file.
rem
rem  OR - drag folders (or images) onto this file in Explorer.
rem
rem  Either way the results land beside each input, in a "deblurred" subfolder
rem  and with a "_db" name suffix, so they can never be mixed up with the
rem  originals:
rem
rem      D:\photos\IMG_0001.jpg   ->  D:\photos\deblurred\IMG_0001_db.jpg
rem      input\holiday\a.png      ->  input\holiday\deblurred\a_db.png
rem
rem  Re-running is safe: files that are already results are skipped, so the
rem  folders do not grow every time.
rem
rem  Explicit paths also work - any argument that exists on disk is an input.
rem      run_infer.bat "D:\photos" "E:\more photos"
rem
rem  No options are needed.  Large images are tiled automatically, results are
rem  always full resolution, and an image past the 100 megapixel ceiling is
rem  reported instead of being shrunk.
rem
rem  Bit depth needs no setting either: a 16-bit TIFF comes back 16-bit, an 8-bit
rem  JPEG stays 8-bit, and a folder holding both is handled in one run.
rem
rem  Any argument starting with "-" is passed straight to local_infer.py and
rem  applied to every run:
rem      --dest output           write to the project "output" folder instead
rem      --dest subfolder        subfolder only, no "_db"
rem      --dest suffix           "_db" beside the originals, no subfolder
rem      --subfolder sharp       rename the subfolder
rem      --suffix _sharp         change the suffix
rem      --tile 512              smaller tiles, less GPU memory
rem      --overlap 384           wider feather between tiles
rem      --max-pixels 200000000  raise the ceiling
rem      --ext png               write PNG instead of keeping the input format
rem
rem  This wrapper always processes subfolders.  For everything else run
rem      .venv\Scripts\python.exe local_infer.py --help
rem  or see LOCAL_SETUP.md.
rem ===========================================================================

set "HERE=%~dp0"
set "PY=%HERE%.venv\Scripts\python.exe"
set "INROOT=%HERE%input"

rem Keep the window open when the file was double-clicked, but not when the
rem command was typed into an already-open prompt.  Set IFAN_NO_PAUSE=1 to
rem suppress the pause unconditionally, e.g. when calling this from a script.
set "HOLD="
if defined IFAN_NO_PAUSE goto :no_pause
echo %cmdcmdline% | find /i "%~nx0" >nul
if not errorlevel 1 set "HOLD=1"
:no_pause

if not exist "%PY%" (
    echo [ERROR] Python environment not found:
    echo         %PY%
    echo         See section 4 of LOCAL_SETUP.md.
    goto :end
)
if not exist "%HERE%ckpt\IFAN.pytorch" if not exist "%HERE%ckpt\IFAN_16bit.pytorch" (
    echo [ERROR] No pretrained checkpoint found in:
    echo         %HERE%ckpt
    echo         See section 4.2 of LOCAL_SETUP.md.
    goto :end
)

if not exist "%INROOT%" mkdir "%INROOT%"

rem ---- sort the arguments into options and inputs ---------------------------
rem An argument beginning with "-" is an option.  Whatever follows an option is
rem treated as its value, even if a file of that name happens to exist, so that
rem things like "--dest output" and "--tile 512" are not mistaken for inputs.
rem Every remaining argument must exist on disk to count as an input.
set "OPTIONS="
set "COUNT=0"
set "PREVOPT="
:parse
if "%~1"=="" goto :parsed
set "ARG=%~1"
if "!ARG:~0,1!"=="-" goto :opt
if defined PREVOPT goto :opt_value
if exist "!ARG!" goto :inp
echo [WARN] not found on disk, passing through as an option: !ARG!
goto :opt

:opt
set "PREVOPT=1"
set "OPTIONS=!OPTIONS! !ARG!"
shift
goto :parse

:opt_value
set "PREVOPT="
set "OPTIONS=!OPTIONS! !ARG!"
shift
goto :parse

:inp
set "PREVOPT="
set /a COUNT+=1
set "INPUT_!COUNT!=!ARG!"
shift
goto :parse
:parsed

if !COUNT! EQU 0 (
    rem Nothing dropped, so use the drop-zone folder.
    set /a COUNT=1
    set "INPUT_1=%INROOT%"
    echo Nothing was dropped, so processing the "input" folder.
    dir /b /s "%INROOT%\*.jpg" "%INROOT%\*.jpeg" "%INROOT%\*.png" "%INROOT%\*.bmp" "%INROOT%\*.tif" "%INROOT%\*.tiff" "%INROOT%\*.webp" >nul 2>&1
    if errorlevel 1 (
        echo.
        echo No images found in:
        echo     %INROOT%
        echo Put image files, or folders of images, in there and run again.
        goto :end
    )
)

echo.
for /l %%K in (1,1,!COUNT!) do call :run_one "!INPUT_%%K!"

goto :end

rem ---------------------------------------------------------------------------
rem  Process one input folder or image.  %1 = path.
rem ---------------------------------------------------------------------------
:run_one
set "SRC=%~1"
if "%SRC:~-1%"=="\" set "SRC=%SRC:~0,-1%"
echo =====================================================================
echo Source : %SRC%
echo =====================================================================
"%PY%" "%HERE%local_infer.py" --input "%SRC%" --recursive !OPTIONS!
if errorlevel 1 echo [ERROR] local_infer.py exited with code %ERRORLEVEL%
echo.
exit /b 0

:end
echo.
if defined HOLD pause
exit /b 0
