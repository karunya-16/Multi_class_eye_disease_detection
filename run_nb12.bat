@echo off
cd /d "d:\Practice Projects\Disease Detection"
.venv311\Scripts\python.exe -c "import pandas, tensorflow; print('pandas', pandas.__version__, 'tf', tensorflow.__version__)"
