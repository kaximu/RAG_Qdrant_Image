@echo off
echo ================================================
echo 🚀 Setting up Python environment for RAG Chatbot
echo ================================================

:: Step 1: Upgrade pip
python -m pip install --upgrade pip

:: Step 2: Install Python requirements
pip install -r requirements.txt

:: Step 3: Install OCR dependencies
pip install pytesseract pillow pdf2image

:: Step 4: Check if Tesseract OCR is installed (Windows default path)
if exist "C:\Program Files\Tesseract-OCR\tesseract.exe" (
    echo ✅ Tesseract OCR already installed.
) else (
    echo ⚠️ Tesseract OCR not found!
    echo 👉 Please download and install from: https://github.com/UB-Mannheim/tesseract/wiki
    echo Default path: C:\Program Files\Tesseract-OCR
)

echo ================================================
echo ✅ Setup complete! You can now run:
echo streamlit run chatbot_app_Qdrant.py
echo ================================================
pause
