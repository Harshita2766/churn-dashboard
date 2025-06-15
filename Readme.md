# 📊 Telco Customer Churn Prediction Dashboard

A complete end-to-end machine learning project that predicts customer churn using a trained classification model and visualizes insights through a beautiful **dark-themed Streamlit dashboard**.

<img src="https://img.shields.io/badge/streamlit-dashboard-brightgreen?style=flat-square" /> <img src="https://img.shields.io/badge/status-active-success?style=flat-square" /> <img src="https://img.shields.io/badge/python-3.10-blue?style=flat-square" />  

---

## 📌 Project Overview

Telecom companies often struggle with customer retention due to high churn rates. This project provides:

- A trained ML model for churn prediction
- A **dashboard** to visualize churn distribution, risk levels, and key customers
- A user-friendly interface to explore predictions and download results

---

## 🧠 Model Summary

- **Algorithm**: Logistic Regression  
- **Features Used**: Demographic + service usage + contract info  
- **Metrics**: F1-Score, Accuracy, Precision/Recall 

---

## 🚀 Dashboard Features

✅ Churn Probability Distribution  
✅ Churn vs Retain Pie Chart  
✅ Top-10 High-Risk Customers Table  
✅ Downloadable Predictions    
✅ Dark Mode Theme  
✅ Clean Error Handling  
✅ Mobile Responsive *(via Streamlit)*  
✅ Real-time insights from predictions.csv  

---

## 🗂️ Project Structure

churn-prediction/
│
├── data/
│ ├── raw/
│ │ └── telco_train.csv # Input data
│ ├── models/
│ │ └── notebooks/
│ │ ├── model.pkl # Trained model
│ │ ├── model_features.pkl # Features used in model
│ │ ├── predictions.csv # Churn predictions
│ │ └── eda_model.ipynb # Training and EDA notebook
│
├── apps/
│ └── dashboard.py # Streamlit dashboard app
│
├── requirements.txt
└── README.md

yaml
Copy
Edit

---

## 🛠️ Setup Instructions

Follow these steps to run the project locally:

### 1. Clone the Repository

```bash
git clone https://github.com/Harshita2766/churn-dashboard.git
cd churn-dashboard

2. Create and Activate a Virtual Environment (optional)
bash
Copy
Edit
python -m venv venv
venv\Scripts\activate        # On Windows
# OR
source venv/bin/activate     # On macOS/Linux
3. Install Required Packages
bash
Copy
Edit
pip install -r requirements.txt
If requirements.txt is missing, generate it using:

bash
Copy
Edit
pip freeze > requirements.txt
4. Run the Streamlit Dashboard
bash
Copy
Edit
streamlit run data/models/notebooks/apps/dashboard.py

🧾 License
This project is licensed under the MIT License – feel free to use, share, or improve!

🙋‍♀️ Author
Harshita
🔗 GitHub: @Harshita2766
📧 LinkedIn: Harshita Hemnani

Made with ❤️ using Python, Streamlit, Pandas, and Scikit-learn
