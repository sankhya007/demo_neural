@echo off
echo Installing required packages...
pip install -r requirements.txt

echo Running Neural Network...
python main.py

pause