@echo off
REM ============================================================
REM Multi-seed campaign, RESUME from judging (generation + merge + L1
REM already completed and verified). Runs the four remaining judging
REM passes (L2/L3 x seeds 1,7), DeepSeek-V4-Flash, max_tokens 1500,
REM 4 workers. Resumable: re-run after any interruption.
REM ETA ~3.5 h.
REM ============================================================

set PY=python
cd /d "%~dp0src"

echo === Layer-2 judging seed 1 started %date% %time% ===
%PY% layer2_eval.py reports_main_glm_n100_B3toB7_seed1.parquet --judge deepseek --seed 42 --max-tokens 1500 --workers 4 > ..\logs\multiseed_l2_s1.log 2>&1
echo === Layer-2 judging seed 1 finished %date% %time% ===

echo === Layer-3 judging seed 1 started %date% %time% ===
%PY% layer3_eval.py reports_main_glm_n100_B3toB7_seed1.parquet --judge deepseek --seed 42 --max-tokens 1500 --workers 4 > ..\logs\multiseed_l3_s1.log 2>&1
echo === Layer-3 judging seed 1 finished %date% %time% ===

echo === Layer-2 judging seed 7 started %date% %time% ===
%PY% layer2_eval.py reports_main_glm_n100_B3toB7_seed7.parquet --judge deepseek --seed 42 --max-tokens 1500 --workers 4 > ..\logs\multiseed_l2_s7.log 2>&1
echo === Layer-2 judging seed 7 finished %date% %time% ===

echo === Layer-3 judging seed 7 started %date% %time% ===
%PY% layer3_eval.py reports_main_glm_n100_B3toB7_seed7.parquet --judge deepseek --seed 42 --max-tokens 1500 --workers 4 > ..\logs\multiseed_l3_s7.log 2>&1
echo === Layer-3 judging seed 7 finished %date% %time% ===

echo === ALL DONE %date% %time% ===
pause
