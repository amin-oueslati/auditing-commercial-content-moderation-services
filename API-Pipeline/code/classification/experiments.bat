@echo off

rem Define datasets and APIs
set datasets=toxigen jigsaw megaspeech
set apis=google microsoft gpt amazon

rem Activate Conda environment
call conda activate AuditNLP

rem Create a directory for logs if it doesn't exist
if not exist logs mkdir logs

rem Loop through datasets (sequential)
for %%d in (%datasets%) do (
    rem Start a new command prompt window for each API within the dataset
    for %%a in (%apis%) do (
        rem Execute classification.py for each combination and redirect output to a log file
        start cmd /k python classification.py --dataset %%d --sample_frac 1 --api %%a > "logs\%%d_%%a.log" 2>&1
        rem Add a small delay to ensure APIs are not overloaded
        timeout /t 5 /nobreak >nul
    )
)

rem Deactivate Conda environment
conda deactivate
