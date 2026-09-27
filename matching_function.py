
#Matching function · PY
import pandas as pd
import numpy as np
import pickle
from math import radians, sin, cos, sqrt, atan2
 
# ─── Load Model and Donors ─────────────────────────────────────────
 
with open("blood_match_model.pkl", "rb") as f:
    model = pickle.load(f)
 
donors_df = pd.read_csv("donors.csv")
 
# ─── Blood Compatibility Rules ─────────────────────────────────────
 
COMPATIBILITY = {
    "O-":  ["O-", "O+", "A-", "A+", "B-", "B+", "AB-", "AB+"],
    "O+":  ["O+", "A+", "B+", "AB+"],
    "A-":  ["A-", "A+", "AB-", "AB+"],
    "A+":  ["A+", "AB+"],
    "B-":  ["B-", "B+", "AB-", "AB+"],
    "B+":  ["B+", "AB+"],
    "AB-": ["AB-", "AB+"],
    "AB+": ["AB+"],
}
 
# ─── Helper Functions ──────────────────────────────────────────────
 
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return round(R * c, 2)
 
def is_donor_eligible(donor):
    if donor["hiv"] == 1: return 0
    if donor["hepatitis_b"] == 1: return 0
    if donor["hepatitis_c"] == 1: return 0
    if donor["hemoglobin"] < 12.5: return 0
    if donor["weight_kg"] < 50: return 0
    if donor["is_available"] == 0: return 0
    return 1
 
def days_since_donation(date_str):
    last_date = pd.to_datetime(date_str)
    today = pd.Timestamp.today()
    return (today - last_date).days
 
# ─── Main Matching Function ────────────────────────────────────────
 
def find_best_donors(recipient_blood_type, recipient_rh,
                     recipient_lat, recipient_lon,
                     urgency_score, top_n=5):
 
    recipient_blood = recipient_blood_type + recipient_rh
 
    results = []
    for _, donor in donors_df.iterrows():
 
        eligible = is_donor_eligible(donor)
        if eligible == 0:
            continue
 
        days_gap = days_since_donation(donor["last_donated_date"])
        if days_gap < 56:
            continue
 
        donor_blood = donor["blood_type"] + donor["rh_factor"]
        blood_match = 1 if recipient_blood in COMPATIBILITY[donor_blood] else 0
 
        distance = calculate_distance(
            donor["latitude"], donor["longitude"],
            recipient_lat, recipient_lon
        )
 
        features = pd.DataFrame([{
            "blood_match":         blood_match,
            "distance_km":         distance,
            "days_since_donation": days_gap,
            "urgency_score":       urgency_score,
            "is_eligible":         eligible
        }])
 
        score = model.predict_proba(features)[0][1]
 
        results.append({
            "donor_id":            donor["donor_id"],
            "blood_type":          donor["blood_type"] + donor["rh_factor"],
            "city":                donor["city"],
            "distance_km":         distance,
            "days_since_donation": days_gap,
            "match_score":         round(score * 100, 2)
        })
 
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("match_score", ascending=False)
    return results_df.head(top_n)
 
# ─── Test the Matching Function ────────────────────────────────────
 
print("=" * 60)
print("  BEST DONORS FOR PATIENT")
print("  Blood Type : O+  |  City : Chennai  |  Urgency : 9")
print("=" * 60)
 
top_donors = find_best_donors(
    recipient_blood_type = "O",
    recipient_rh         = "+",
    recipient_lat        = 13.0827,
    recipient_lon        = 80.2707,
    urgency_score        = 9
)
 
print(top_donors.to_string(index=False))
print("\nMatching complete ✅")
 
