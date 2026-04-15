@echo off
setlocal

set PROJECT_ROOT=d:\ai-gout-management-agent
set CONDA_EXE=D:\Miniconda3\Scripts\conda.exe
set CONDA_ENV=gout-agent

cd /d %PROJECT_ROOT%

echo Project root: %PROJECT_ROOT%
echo Conda env: %CONDA_ENV%
echo Local LLM base URL: http://127.0.0.1:1234/v1
echo Local LLM model: HuatuoGPT-o1-7B

if not exist "%CONDA_EXE%" (
  echo Conda executable not found: %CONDA_EXE%
  pause
  exit /b 1
)

set LOCAL_LLM_BASE_URL=http://127.0.0.1:1234/v1
set LOCAL_LLM_API_KEY=lm-studio
set LOCAL_LLM_MODEL=HuatuoGPT-o1-7B
set LOCAL_LLM_TIMEOUT_SECONDS=60

echo Starting Streamlit...
"%CONDA_EXE%" run -n %CONDA_ENV% python -m streamlit run streamlit_app.py

pause
