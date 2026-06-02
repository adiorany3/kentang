@echo off
python -m pip install --upgrade pip
pip install -r requirements.txt
python download_model.py
streamlit run main.py
pause
