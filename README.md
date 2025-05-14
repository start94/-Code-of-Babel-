# 🧠 Code of Babel – Language Detection API for Museums

**Code of Babel** is a FastAPI-based REST service designed to automatically detect the language of short museum-related texts.  
It leverages a pre-trained scikit-learn pipeline for multilingual classification with high confidence.

---

## 🚀 Features

- 🔍 Detects the language of any museum/cultural text snippet
- 📊 Returns ISO language code and confidence score
- ⚡ Built with FastAPI (automatic docs & validation)
- 📁 Includes logging for audit/debugging
- 🧪 Ready for Docker deployment

---

## 🛠️ Tech Stack

- `FastAPI`
- `scikit-learn`
- `pydantic`
- `uvicorn`
- `numpy`

---

## 📦 Setup (Local)

### 1. Clone the repo
```bash
git clone https://github.com/start94/Code-of-Babel.git
cd Code-of-Babel
2. Install dependencies
bash
Copia
Modifica
pip install fastapi uvicorn scikit-learn numpy pydantic
3. Run the API
bash
Copia
Modifica
uvicorn museum_lang_api:app --reload
4. Try it in your browser
Visit 👉 http://localhost:8000/docs for Swagger UI.

🐳 Docker Setup
Build the image:

bash
Copia
Modifica
docker build -t code-of-babel .
Run the container:

bash
Copia
Modifica
docker run -d -p 8000:8000 code-of-babel
📥 API Usage
POST /identify-language
Detect the language of a text.

Request:

json
Copia
Modifica
{
  "text": "Questo è un testo in italiano."
}
Response:

json
Copia
Modifica
{
  "language_code": "IT",
  "confidence": 0.9978
}
📂 File Overview
File	Description
museum_lang_api.py	Main FastAPI app with endpoints
language_detection_pipeline.pkl	Pretrained language classifier
Dockerfile	Docker build instructions
README.md	Project documentation

⚠️ Notes
The language_detection_pipeline.pkl is required to run the API.

Not suitable for long documents or paragraphs (optimized for short text).

📄 License
This project is currently unlicensed. Add a LICENSE file if needed.

🙋‍♂️ Author
Developed by Raffaele Diomaiuto
GitHub: start94
Project: MuseumLangAPI – School collaboration


