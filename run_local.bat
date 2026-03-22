@echo off
setlocal

set PROJECT_ROOT=d:\ai-gout-management-agent
set CONDA_ENV=py310

cd /d %PROJECT_ROOT%

echo Project root: %PROJECT_ROOT%
echo Conda env: %CONDA_ENV%
echo Local LLM base URL: http://127.0.0.1:1234/v1
echo Local LLM model: HuatuoGPT-o1-7B

echo Activating conda environment...
call conda activate %CONDA_ENV%
if errorlevel 1 (
  echo Failed to activate conda env: %CONDA_ENV%
  pause
  exit /b 1
)

set LOCAL_LLM_BASE_URL=http://127.0.0.1:1234/v1
set LOCAL_LLM_API_KEY=lm-studio
set LOCAL_LLM_MODEL=HuatuoGPT-o1-7B
set LOCAL_LLM_TIMEOUT_SECONDS=60

echo Starting Streamlit...
python -m streamlit run streamlit_app.py

pause