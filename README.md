# HR Assistant Chatbot 🤖

## 📌 Overview
HR Assistant Chatbot is a Streamlit-based HR analytics assistant that answers questions in English or Arabic.
Users can choose between a local offline model (Qwen) or a cloud model (Groq). The chatbot converts questions into safe SQL,
queries an SQLite employee database, and returns readable HR insights.

---

## 🎯 Project Goal
The main goal is to make HR data accessible through natural language.
Instead of writing SQL, users can ask questions like “average salary by department” or “how many employees left” and get instant results.

---

## ✨ Key Features
- English + Arabic questions
- Local mode: Qwen (Offline)
- Cloud mode: Groq
- SQL cleaning + column name normalization to reduce failures
- Friendly formatted answers
- Download chat history from the sidebar
- Clean UI theme 

---
## 🧠 How It Works
1) The user types a question (English or Arabic).
2) The selected LLM generates an SQL query.
3) The SQL is cleaned to keep only a single safe SELECT statement.
4) Common column name variants are normalized to match the dataset schema.
5) The SQL query runs on the SQLite database (hr.db).
6) Results are formatted into a readable response and displayed in the chat UI.


---
## 🧩 Tech Stack

### Frontend
- Streamlit: Used to build the interactive chatbot UI, sidebar controls, and chat interface.
### Backend
- Python: Core programming language used for application logic and data processing.
### Database
- SQLite: Lightweight relational database used to store employee HR data.
### Large Language Models
- Qwen (Local / Offline): Used for on-device SQL generation without internet access.
- Groq (Cloud): Used for fast cloud-based SQL generation via API.
### Data & Query Layer
- SQL (SQLite dialect):Used for querying employee data.
- Regular Expressions:Used to clean and validate generated SQL queries.
### Security & Safety
- Read-only SQL guardrails (SELECT only)
- Column name normalization to avoid invalid queries
- GitHub Push Protection for secret detection

---

## 📂 Project Structure

```text
Chatbot_Project/
├── app.py            — Main Streamlit application (UI and chat flow)
├── db.py             — Database initialization and SQL execution
├── hr.db             — SQLite database with employee data
├── sql_utils.py      — SQL normalization and answer formatting
├── local_sql.py      — Local Qwen pipeline (offline SQL generation)
├── cloud_sql.py      — Cloud Groq pipeline (online SQL generation)
├── setup_models.py   — Model and client loading utilities
├── load_data.py      — Script to load HR data into the database
├── requirements.txt  — Python dependencies
├── README.md         — Project documentation
└── .gitignore        — Ignored files and folders
```

---

## 🔐 Security Notes
- API keys must never be committed to GitHub.
- This project is designed to allow read-only queries only (no data modification SQL).

---
## 🚀 Installation & Setup

### 1. Clone the Repository
    git clone https://github.com/janaalmotairi/Chatbot_Project.git
    cd Chatbot_Project


### 2. Install dependencies
    pip install -r requirements.txt

### 3. Run the app:
     streamlit run app.py

### 3. Setup the local model
**Note:** The local model files are large and are not included in this repository due to GitHub limits.
Download the Qwen 0.5B model from Hugging Face.
Place the model files inside a folder named `local_qwen_model` in the root directory.


---

## ▶️ Usage

1. **Select Model:** Choose between "Local Qwen" or "Cloud Qwen" from the sidebar.
2. **Ask Questions:** Type your HR-related query in the chat input.
3. **Download Chat:** download the chat fron the sidebar.

---
## 📸 Screenshots

### 1) Home Screen
<img width="1902" height="992" alt="Screenshot 2026-02-05 164904" src="https://github.com/user-attachments/assets/42d76037-2120-4876-b469-e3b32ada49e0" />

### 2) Example Results (Local Mode)
<img width="1900" height="986" alt="Screenshot 2026-02-05 164845" src="https://github.com/user-attachments/assets/dc4a0ed1-aac4-4915-8977-77059aa4ef1b" />


### 3) Cloud Mode (Groq)
<img width="1903" height="981" alt="Screenshot 2026-02-05 163842" src="https://github.com/user-attachments/assets/5ccc857f-19c0-48bc-a8f3-e39459f5a74b" />


## 👩‍💻 Developed By
Jana

