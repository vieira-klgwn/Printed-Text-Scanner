# 📄 Image & PDF Text Extraction Tool (PyQt5 + OpenCV + Tesseract OCR)

This project is a beginner-friendly graphical application that extracts **text from images** (JPG, PNG, etc.) and **PDF files** using:

- **OpenCV (cv2)**
- **Pytesseract (OCR)**
- **NumPy**
- **Pillow (PIL)**
- **PyQt5 (GUI)**

The app allows you to upload images or PDF pages and automatically extract text using OCR.

---

## 🚀 Features

- Extract text from images  
- Extract text from PDF files  
- Supports JPG, PNG, JPEG, PDF  
- Clean PyQt5 user interface  
- High-quality OCR using Tesseract  
- Image preview before extraction  
- Copy or save extracted text  

---

## 📦 Installation

### **1. Create and activate a virtual environment**

python3 -m venv venv
source venv/bin/activate
venv\Scripts\activate


### **2. Install required Python packages**

pip install numpy pillow pytesseract PyQt5 opencv-python


### **3. Install Tesseract OCR (required)**

#### **Linux (Kali / Ubuntu / Debian)**

sudo apt update
sudo apt install tesseract-ocr


#### **Windows**

Download Tesseract OCR:  
https://github.com/tesseract-ocr/tesseract

Add this to PATH:

C:\Program Files\Tesseract-OCR


---

## ▶️ Run the Application

python main.py


> Replace `main.py` with your script name if different.

---

## 📁 Project Structure

📦 ocr-text-extractor
┣ main.py
┣ README.md
┣ requirements.txt
┗ assets/


---

## 📝 Example `requirements.txt`

numpy
pillow
pytesseract
PyQt5
opencv-python


---

## 🔧 Troubleshooting

### **cv2 fails to import**

sudo apt install libgl1
pip install opencv-python


### **TesseractNotFoundError**
You forgot to install Tesseract OCR.  
See the installation section above.

---

## ❤️ Contributing

Pull requests are welcome.  
You can improve the UI, add dark mode, text export, or image preprocessing features.

---

## 📜 License

This project is open-source.  
You are free to modify and improve it.

