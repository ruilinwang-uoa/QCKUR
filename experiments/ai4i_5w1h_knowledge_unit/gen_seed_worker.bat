@echo off
REM Worker: generate one (system, seed) slice of the multi-seed campaign.
REM Usage: gen_seed_worker.bat <SYSTEM> <SEED>
set PY=python
cd /d "%~dp0src"
%PY% run_experiment.py full --n 100 --systems %1 --seeds %2 --out reports_main_glm_n100_%1_seed%2.parquet > ..\logs\multiseed_%1_s%2.log 2>&1
echo done %date% %time% > ..\logs\flags\%1_s%2.done
