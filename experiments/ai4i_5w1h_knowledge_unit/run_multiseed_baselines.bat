@echo off
REM ============================================================
REM Multi-seed baseline experiment (PARALLEL version)
REM   - Systems : B3,B4,B5,B6,B7 x seeds 1,7 = 10 concurrent generation
REM               workers (one minimized window each), GLM-4-Flash
REM   - Then    : merge per seed -> L1 (local) -> L2/L3 judging with
REM               DeepSeek-V4-Flash, max_tokens 1500, 4 workers
REM   - Resume  : generation workers skip completed (method, instance,
REM               seed) triples; judging skips scored pairs. If anything
REM               is interrupted, just double-click this file again.
REM   - ETA     : generation ~2.6 h (B7 is the bottleneck) + judging
REM               ~3.5 h  =>  ~6 h total
REM   - Fallback: run_multiseed_baselines_serial.bat (sequential, ~9 h)
REM ============================================================

set PY=python
set ROOT=%~dp0.
cd /d %ROOT%\src
if not exist %ROOT%\logs\flags mkdir %ROOT%\logs\flags

echo === Launching 10 generation workers %date% %time% ===
for %%S in (1 7) do (
  for %%M in (B3 B4 B5 B6 B7) do (
    if exist %ROOT%\logs\flags\%%M_%%S.done del %ROOT%\logs\flags\%%M_%%S.done
    start "gen %%M s%%S" /min %ROOT%\gen_seed_worker.bat %%M %%S
  )
)

echo === Waiting for generation workers to finish (checks every 60 s) ===
:waitloop
timeout /t 60 /nobreak >nul
for %%S in (1 7) do for %%M in (B3 B4 B5 B6 B7) do (
  if not exist %ROOT%\logs\flags\%%M_%%S.done goto waitloop
)
echo === All generation workers done %date% %time% ===

echo === Merging slices %date% %time% ===
%PY% merge_multiseed.py 1
%PY% merge_multiseed.py 7

echo === Layer-1 scoring (programmatic, local) %date% %time% ===
%PY% layer1_reports.py reports_main_glm_n100_B3toB7_seed1.parquet > ..\logs\multiseed_l1_s1.log 2>&1
%PY% layer1_reports.py reports_main_glm_n100_B3toB7_seed7.parquet > ..\logs\multiseed_l1_s7.log 2>&1

echo === Layer-2 judging seed 1 %date% %time% ===
%PY% layer2_eval.py reports_main_glm_n100_B3toB7_seed1.parquet --judge deepseek --seed 42 --max-tokens 1500 --workers 4 > ..\logs\multiseed_l2_s1.log 2>&1
echo === Layer-3 judging seed 1 %date% %time% ===
%PY% layer3_eval.py reports_main_glm_n100_B3toB7_seed1.parquet --judge deepseek --seed 42 --max-tokens 1500 --workers 4 > ..\logs\multiseed_l3_s1.log 2>&1
echo === Layer-2 judging seed 7 %date% %time% ===
%PY% layer2_eval.py reports_main_glm_n100_B3toB7_seed7.parquet --judge deepseek --seed 42 --max-tokens 1500 --workers 4 > ..\logs\multiseed_l2_s7.log 2>&1
echo === Layer-3 judging seed 7 %date% %time% ===
%PY% layer3_eval.py reports_main_glm_n100_B3toB7_seed7.parquet --judge deepseek --seed 42 --max-tokens 1500 --workers 4 > ..\logs\multiseed_l3_s7.log 2>&1

echo === ALL DONE %date% %time% ===
pause
