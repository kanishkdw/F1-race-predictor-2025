# 🏎️ F1 Race Predictor

An ML-powered Formula 1 race prediction system built with **FastF1** and **Python**.  
This project uses historical race data to predict lap times and race outcomes, with evaluation and visualization support.

---

## 📌 Features
- Data extraction from **FastF1 API**
- Data preprocessing and feature engineering
- Machine learning models for race time and outcome prediction
- Evaluation using **Mean Absolute Error (MAE)**
- Visualization of results with **Streamlit dashboard**

## ⚙️ Installation
1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/f1-race-predictor.git
   cd f1-race-predictor
  
2. Install dependencies:
     ```bash
     Copy code
     pip install -r requirements.txt
      ```
## 🚀 Usage

Run predictions for a specific race:
    
    python src/main_china.py

Launch the dashboard:

    streamlit run dashboard.py


## 📊 Results
-Predictions for Chinese GP, Japanese GP, and Australian GP
-Evaluated using MAE
-Visualization of predicted vs actual results

## 🛠️ Tech Stack
Python 3.x

FastF1

scikit-learn

Streamlit

Matplotlib / Seaborn
