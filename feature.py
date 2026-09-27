import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# ─── Load Datasets ─────────────────────────────────────────────────

donors_df = pd.read_csv("/mnt/user-data/outputs/donors.csv")
recipients_df = pd.read_csv("/mnt/user-data/outputs/recipients.csv")

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

# ─── Distance Function (Haversine Formula) ─────────────────────────

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return round(R * c, 2)

# ─── Eligibility Check Function ────────────────────────────────────

def is_donor_eligible(donor):
    if donor["hiv"] == 1: return 0
    if donor["hepatitis_b"] == 1: return 0
    if donor["hepatitis_c"] == 1: return 0
    if donor["hemoglobin"] < 12.5: return 0
    if donor["weight_kg"] < 50: return 0
    if donor["is_available"] == 0: return 0
    return 1

# ─── Days Since Last Donation ──────────────────────────────────────

def days_since_donation(date_str):
    last_date = pd.to_datetime(date_str)
    today = pd.Timestamp.today()
    return (today - last_date).days

# ─── Create Donor-Recipient Pairs ──────────────────────────────────

pairs = []
for _, recipient in recipients_df.iterrows():
    for _, donor in donors_df.iterrows():

        donor_blood     = donor["blood_type"] + donor["rh_factor"]
        recipient_blood = recipient["blood_type"] + recipient["rh_factor"]

        blood_match = 1 if recipient_blood in COMPATIBILITY[donor_blood] else 0

        distance = calculate_distance(
            donor["latitude"], donor["longitude"],
            recipient["latitude"], recipient["longitude"]
        )

        days_gap = days_since_donation(donor["last_donated_date"])
        eligible = is_donor_eligible(donor)

        label = 1 if (blood_match == 1 and eligible == 1 and days_gap >= 56) else 0

        pairs.append({
            "donor_id":            donor["donor_id"],
            "recipient_id":        recipient["recipient_id"],
            "blood_match":         blood_match,
            "distance_km":         distance,
            "days_since_donation": days_gap,
            "urgency_score":       recipient["urgency_score"],
            "is_eligible":         eligible,
            "compatibility":       label
        })

# ─── Save to CSV ───────────────────────────────────────────────────

pairs_df = pd.DataFrame(pairs)
pairs_df.to_csv("/mnt/user-data/outputs/feature_engineered.csv", index=False)

print("=" * 55)
print("  FEATURE ENGINEERED DATASET")
print("=" * 55)
print(pairs_df.head(5).to_string(index=False))
print(f"\nTotal pairs generated : {len(pairs_df)}")
print(f"Compatible pairs      : {pairs_df['compatibility'].sum()}")
print(f"Incompatible pairs    : {(pairs_df['compatibility'] == 0).sum()}")