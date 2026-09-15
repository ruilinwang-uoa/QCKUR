@echo off
REM ============================================================
REM Multi-seed baseline experiment (SERIAL fallback)
REM   Same campaign as run_multiseed_baselines.bat but sequential:
REM   use this if the parallel version triggers API rate limits.
REM   Generation ~5 h + judging ~3.5 h. Fully resumable: re-run this
REM   file after any interruption; completed triples are skipped.
REM   NOTE: generation writes the same per-seed merged output names,
REM   so slices from the parallel run are NOT reused here (per-system
REM   slice files differ from this file's combined output). If you
REM   switch from parallel to serial mid-way, delete
REM   runs/reports_main_glm_n100_B3toB7_seed{1,7}.parquet first ONLY
REM   if they are partial merges; per-system slices are still usable
REM   via merge_multiseed.py.
REM ============================================================

set PY=python
cd /d "%~dp0src"

echo === [1/6] Generation seed 1 started %date% %time% ===
%PY% run_experiment.py full --n 100 --systems B3,B4,B5,B6,B7 --seeds 1 --out reports_main_glm_n100_B3toB7_seed1.parquet > ..\logs\multiseed_gen_s1.log 2>&1
echo === [1/6] Generation seed 1 finished %date% %time% ===

echo === [2/6] Generation seed 7 started %date% %time% ===
%PY% run_experiment.py full --n 100 --systems B3,B4,B5,B6,B7 --seeds 7 --out reports_main_glm_n100_B3toB7_seed7.parquet > ..\logs\multiseed_gen_s7.log 2>&1
echo === [2/6] Generation seed 7 finished %date% %time% ===

echo === [3/6] Layer-1 scoring (programmatic, local) %date% %time% ===
%PY% layer1_reports.py reports_main_glm_n100_B3toB7_seed1.parquet > ..\logs\multiseed_l1_s1.log 2>&1
%PY% layer1_reports.py reports_main_glm_n100_B3toB7_seed7.parquet > ..\logs\multiseed_l1_s7.log 2>&1

echo === [4/6] Layer-2 judging seed 1 %date% %time% ===
%PY% layer2_eval.py reports_main_glm_n100_B3toB7_seed1.parquet --judge deepseek --seed 42 --max-tokens 1500 --workers 4 > ..\logs\multiseed_l2_s1.log 2>&1
echo === [5/6] Layer-3 judging seed 1 %date% %time% ===
%PY% layer3_eval.py reports_main_glm_n100_B3toB7_seed1.parquet --judge deepseek --seed 42 --max-tokens 1500 --workers 4 > ..\logs\multiseed_l3_s1.log 2>&1
echo === Layer-2 judging seed 7 %date% %time% ===
%PY% layer2_eval.py reports_main_glm_n100_B3toB7_seed7.parquet --judge deepseek --seed 42 --max-tokens 1500 --workers 4 > ..\logs\multiseed_l2_s7.log 2>&1
echo === [6/6] Layer-3 judging seed 7 %date% %time% ===
%PY% layer3_eval.py reports_main_glm_n100_B3toB7_seed7.parquet --judge deepseek --seed 42 --max-tokens 1500 --workers 4 > ..\logs\multiseed_l3_s7.log 2>&1

echo === ALL DONE %date% %time% ===
pause
