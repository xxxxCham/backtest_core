@echo off
REM Script de profiling pour Windows
REM Usage: tools\profile.bat simple ema_cross
REM        tools\profile.bat grid ema_cross 100

cd /d %~dp0..
python tools/profiler.py %*
