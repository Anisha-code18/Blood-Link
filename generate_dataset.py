import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# ─── Reproducibility ───────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ─── Config ────────────────────────────────────────────────────────
NUM_DONORS     = 200
NUM_RECIPIENTS = 50

BLOOD_TYPES = ["A", "B", "AB", "O"]
RH_FACTORS  = ["+", "-"]

# Indian cities with approximate coordinates
CITIES = {
    "Chennai":   (13.0827, 80.2707),
    "Mumbai":    (19.0760, 72.8777),
    "Delhi":     (28.6139, 77.2090),
    "Bengaluru": (12.9716, 77.5946),
    "Kolkata":   (22.5726, 88.3639),
    "Hyderabad": (17.3850, 78.4867),
    "Pune":      (18.5204, 73.8567),
    "Ahmedabad": (23.0225, 72.5714),
}

# ─── Helper Functions ──────────────────────────────────────────────

def random_date(start_days_ago, end_days_ago):
    """Returns a random date between two ranges in the past."""
    days_ago = random.randint(end_days_ago, start_days_ago)
    return (datetime.today() - timedelta(days=days_ago)).strftime("%Y-%m-%d")

def jitter_coords(lat, lon, radius=0.05):
    """Slightly randomize coordinates so donors in same city aren't identical."""
    return (
        round(lat + random.uniform(-radius, radius), 5),
        round(lon + random.uniform(-radius, radius), 5),
    )

# ─── Generate Donors ──────────────────────────────────────────────

donors = []
for i in range(1, NUM_DONORS + 1):
    city = random.choice(list(CITIES.keys()))
    lat, lon = jitter_coords(*CITIES[city])

    # ~5% chance of disease flag (realistic minority)
    hiv   = 1 if random.random() < 0.03 else 0
    hep_b = 1 if random.random() < 0.04 else 0
    hep_c = 1 if random.random() < 0.03 else 0

    donors.append({
        "donor_id":          f"D{i:03d}",
        "blood_type":        random.choice(BLOOD_TYPES),
        "rh_factor":         random.choice(RH_FACTORS),
        # Hemoglobin: normal range 12–17 g/dL; occasionally low (anemic)
        "hemoglobin":        round(random.gauss(13.5, 1.5), 1),
        # Weight: mostly 50–90 kg; occasionally underweight
        "weight_kg":         round(random.gauss(65, 12), 1),
        # Last donated: anywhere from 2 months to 2 years ago
        "last_donated_date": random_date(730, 60),
        "hiv":               hiv,
        "hepatitis_b":       hep_b,
        "hepatitis_c":       hep_c,
        "city":              city,
        "latitude":          lat,
        "longitude":         lon,
        # ~80% of donors are currently available
        "is_available":      1 if random.random() < 0.80 else 0,
    })

donors_df = pd.DataFrame(donors)

# ─── Generate Recipients ──────────────────────────────────────────

recipients = []
for i in range(1, NUM_RECIPIENTS + 1):
    city = random.choice(list(CITIES.keys()))
    lat, lon = jitter_coords(*CITIES[city])

    recipients.append({
        "recipient_id":  f"R{i:03d}",
        "blood_type":    random.choice(BLOOD_TYPES),
        "rh_factor":     random.choice(RH_FACTORS),
        # Urgency: 1 (low) to 10 (critical), skewed slightly higher (hospital context)
        "urgency_score": random.choices(range(1, 11), weights=[1,1,2,2,3,4,5,5,4,3])[0],
        # Units needed: 1 to 4 bags
        "units_needed":  random.randint(1, 4),
        "city":          city,
        "latitude":      lat,
        "longitude":     lon,
    })

recipients_df = pd.DataFrame(recipients)

# ─── Save to CSV ──────────────────────────────────────────────────

donors_df.to_csv("donors.csv", index=False)
recipients_df.to_csv("recipients.csv", index=False)

# ─── Preview ──────────────────────────────────────────────────────

print("=" * 55)
print("  DONORS DATASET")
print("=" * 55)
print(donors_df.head(5).to_string(index=False))
print(f"\nTotal donors: {len(donors_df)}")
print(f"Available donors: {donors_df['is_available'].sum()}")
print(f"Disqualified (any disease): {((donors_df['hiv']==1)|(donors_df['hepatitis_b']==1)|(donors_df['hepatitis_c']==1)).sum()}")

print("\n" + "=" * 55)
print("  RECIPIENTS DATASET")
print("=" * 55)
print(recipients_df.head(5).to_string(index=False))
print(f"\nTotal recipients: {len(recipients_df)}")
print(f"Critical patients (urgency >= 8): {(recipients_df['urgency_score'] >= 8).sum()}")
