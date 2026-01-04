"""
Data preprocessing for Remote Job Matching Recommender System

- Loads raw clickstream and job metadata
- Converts implicit feedback to explicit ratings (1–5)
- Cleans and prepares content text
- Generates basic statistics and required visualizations
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Paths (RELATIVE ONLY)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DIR = os.path.join(BASE_DIR, "..", "archive")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

EVENTS_FILE = os.path.join(RAW_DIR, "event_data.csv")
ITEMS_FILE = os.path.join(RAW_DIR, "job_information.csv")

# -----------------------------
# Utility functions
# -----------------------------
def strip_html(text):
    if pd.isna(text):
        return ""
    return re.sub(r"<.*?>", " ", str(text))

def map_event_to_rating(event):
    if event == "purchase":
        return 5
    if event == "click":
        return 3
    return 1

# -----------------------------
# Load data
# -----------------------------
events = pd.read_csv(EVENTS_FILE)
items = pd.read_csv(ITEMS_FILE)

# -----------------------------
# Process interactions
# -----------------------------
events = events.rename(columns={
    "client_id": "user_id",
    "item_id": "item_id"
})

events["rating"] = events["event_type"].apply(map_event_to_rating)

interactions = (
    events.groupby(["user_id", "item_id"])["rating"]
    .max()
    .reset_index()
)

# -----------------------------
# Process items (content)
# -----------------------------
items = items.rename(columns={
    "pozisyon_adi": "title",
    "item_id_aciklama": "description"
})

items["title"] = items["title"].fillna("")
items["description"] = items["description"].fillna("").apply(strip_html)

items["text"] = items["title"] + " " + items["description"]

items_clean = items[["item_id", "text"]].drop_duplicates()

# -----------------------------
# Save processed data
# -----------------------------
interactions_path = os.path.join(PROCESSED_DIR, "interactions.csv")
items_path = os.path.join(PROCESSED_DIR, "items.csv")

interactions.to_csv(interactions_path, index=False)
items_clean.to_csv(items_path, index=False)

print("Saved processed files:")
print(interactions_path)
print(items_path)

# -----------------------------
# Basic statistics
# -----------------------------
print("\nDataset statistics:")
print("Users:", interactions["user_id"].nunique())
print("Items:", interactions["item_id"].nunique())
print("Interactions:", len(interactions))

# -----------------------------
# Visualizations (REQUIRED)
# -----------------------------
# Rating distribution
plt.figure()
interactions["rating"].value_counts().sort_index().plot(kind="bar")
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "rating_distribution.png"))
plt.close()

# User activity distribution
user_activity = interactions.groupby("user_id").size()
plt.figure()
user_activity.plot(kind="hist", bins=50)
plt.title("User Activity Distribution")
plt.xlabel("Interactions per User")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "user_activity_distribution.png"))
plt.close()

# Item popularity distribution
item_popularity = interactions.groupby("item_id").size()
plt.figure()
item_popularity.plot(kind="hist", bins=50)
plt.title("Item Popularity Distribution")
plt.xlabel("Interactions per Item")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "item_popularity_distribution.png"))
plt.close()

print("Saved plots to results/")
