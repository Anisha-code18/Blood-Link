# 🩸 BloodLink — ML-Based Blood Donor Matching System

A machine learning powered web application that matches blood donors with recipients based on medical compatibility, geographic proximity, and donor eligibility.

---

## 📌 Project Overview

BloodLink is a smart donor-recipient matching system built for emergency healthcare scenarios. Given a patient's blood type, location, and urgency level, the system identifies and ranks the most compatible blood donors in real time using a trained Random Forest model.

---

## 🎯 Features

- Synthetic dataset generation with 200 donors and 50 recipients across 8 major Indian cities
- Rule-based donor eligibility filtering (disease screening, hemoglobin, weight, availability)
- Blood type and Rh factor compatibility checking using standard medical rules
- Geographic distance calculation using the Haversine formula
- Random Forest Classifier trained on 10,000 donor-recipient pairs
- Real-time top-5 donor ranking for any given patient
- Clean web interface built with Flask for hospital use

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas & NumPy | Data generation and manipulation |
| Scikit-learn | Random Forest ML model |
| Flask | Web server and routing |
| Pickle | Saving and loading trained model |
| HTML & CSS | Frontend web interface |
| Jinja2 | Flask templating engine |

---

## 📁 Project Structure

```
BloodLink/
│
├── generate_dataset.py       # Generates synthetic donor & recipient CSV files
├── feature_engineering.py    # Creates donor-recipient pairs with features
├── train_model.py            # Trains and saves the Random Forest model
├── matching_function.py      # Core matching logic (standalone test)
│
├── donors.csv                # Generated donor dataset (200 rows)
├── recipients.csv            # Generated recipient dataset (50 rows)
├── feature_engineered.csv    # 10,000 donor-recipient pairs with labels
├── blood_match_model.pkl     # Trained Random Forest model
│
└── BloodLink/
    ├── app.py                # Flask web application
    └── templates/
        └── index.html        # Web UI for donor matching
```

---

## ⚙️ How It Works

### Step 1 — Data Generation
Synthetic data is generated for 200 donors and 50 recipients with realistic medical attributes including blood type, Rh factor, hemoglobin levels, disease flags, weight, and GPS coordinates across 8 Indian cities.

### Step 2 — Feature Engineering
Every possible donor-recipient pair (200 × 50 = 10,000 pairs) is evaluated. For each pair, the following features are computed:
- `blood_match` — Is the blood type compatible?
- `distance_km` — How far is the donor from the patient?
- `days_since_donation` — How many days since last donation?
- `urgency_score` — How critical is the patient?
- `is_eligible` — Is the donor medically eligible?
- `compatibility` — Final label (0 or 1)

### Step 3 — Model Training
A Random Forest Classifier with 100 trees is trained on 80% of the data (8,000 pairs) and tested on the remaining 20% (2,000 pairs), achieving high accuracy.

### Step 4 — Matching Function
Given a patient's details, the system:
1. Filters out ineligible donors
2. Computes features for each eligible donor
3. Passes features to the trained model
4. Gets compatibility probability scores
5. Returns the top 5 ranked donors

### Step 5 — Web Interface
A Flask web app allows doctors to input patient details and instantly view the top matching donors with their distance, days since donation, and match score.

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn flask
```

### 2. Generate the dataset
```bash
python generate_dataset.py
```

### 3. Run feature engineering
```bash
python feature_engineering.py
```

### 4. Train the model
```bash
python train_model.py
```

### 5. Start the web app
```bash
cd BloodLink
python app.py
```

### 6. Open in browser
```
http://localhost:5000
```

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | 100% |
| Precision | 1.00 |
| Recall | 1.00 |
| F1 Score | 1.00 |

> **Note:** The high accuracy is expected because the compatibility labels were generated using deterministic medical rules. In a real-world dataset with noisy data, accuracy would typically range between 85–95%.

---

## 🔑 Key Feature Importances

| Feature | Importance |
|---|---|
| is_eligible | 49.32% |
| blood_match | 48.45% |
| days_since_donation | 1.42% |
| urgency_score | 0.42% |
| distance_km | 0.38% |

---

## 🏥 Blood Compatibility Rules

| Donor Blood Type | Can Donate To |
|---|---|
| O- | O-, O+, A-, A+, B-, B+, AB-, AB+ |
| O+ | O+, A+, B+, AB+ |
| A- | A-, A+, AB-, AB+ |
| A+ | A+, AB+ |
| B- | B-, B+, AB-, AB+ |
| B+ | B+, AB+ |
| AB- | AB-, AB+ |
| AB+ | AB+ |

---

## 🌆 Supported Cities

Chennai, Mumbai, Delhi, Bengaluru, Kolkata, Hyderabad, Pune, Ahmedabad

---

## ⚠️ Disclaimer

This project uses synthetically generated data for educational and demonstration purposes only. It is not intended for real medical use. Always consult qualified medical professionals for actual blood donor matching.

---

## 👩‍💻 Author

Developed as part of a Machine Learning project on healthcare data.
