# 🛡️ VeriPulse ML: Fake News Detection & TF-IDF Attribution Workbench

VeriPulse is a full-stack machine learning web application designed to detect fake news articles, evaluate model robustness using feature ablation, and provide deep token attribution insights through a dual-engine inference architecture.

---

## 🚀 Key Features

* **Dual-Engine Inference:** Switch seamlessly between a client-side JavaScript TF-IDF simulation and a live Python FastAPI backend powered by XGBoost.
* **Feature Ablation Studies:** Test model robustness and keyword bias by programmatically stripping anchor words (such as `"Reuters"`) before classification.
* **Multi-Model Agreement Lab:** Compare live predictions across multiple classifiers (`XGBoost`, `LinearSVC`, `Random Forest`, and `Logistic Regression`).
* **Interactive Token Attribution Map:** Visualize which words drive predictions toward `REAL` vs `FAKE`.

---

## 🛠️ Tech Stack

* **Frontend:** React, Vite, Tailwind CSS / Custom UI
* **Backend:** Python, FastAPI, Uvicorn, Pydantic
* **Machine Learning:** Scikit-Learn, XGBoost, Joblib, NumPy, Pandas
* **Integrations:** LangChain, Google Gemini API, Tavily Search (`news_verifier.py`)

---

## 📁 Project Structure

```text
CLG-PROJ/
│
├── frontend/             # React & Vite frontend dashboard
├── models/               # Serialized .pkl artifacts (excluded from git)
├── app.py                # FastAPI backend server
├── train_model.py        # ML training script (XGBoost + TF-IDF pipeline)
├── Model_experiments.py  # Model experimentation and ablation logic
├── news_verifier.py      # LLM & search integration module
└── requirements.txt      # Python dependencies
