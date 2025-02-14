from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import xgboost as xgb
import requests
import os
from collections import Counter
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Google Drive file IDs
USERS_CSV_ID = "15jVGtI8f9heb3W944skG8_qJajiFDaD0"
MODEL_JSON_ID = "1hZcUKiI_pGnAwopu1JqyikGe09wx3t2m"

# Function to download files from Google Drive
def download_from_drive(file_id, dest_path):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print(f"✅ Downloaded: {dest_path}")
    else:
        print(f"❌ Failed to download {dest_path}")

# Download users.csv if not exists
users_csv_path = "users.csv"
if not os.path.exists(users_csv_path):
    download_from_drive(USERS_CSV_ID, users_csv_path)

# Download recommendation_model.json if not exists
model_json_path = "recommendation_model.json"
if not os.path.exists(model_json_path):
    download_from_drive(MODEL_JSON_ID, model_json_path)

# Load user metadata
users_df = pd.read_csv(users_csv_path)

# Label Encoding for categorical variables
label_encoders = {}
for col in ["Gender", "Marital_Status", "Sect", "Caste", "State"]:
    le = LabelEncoder()
    users_df[col] = le.fit_transform(users_df[col])
    label_encoders[col] = le

# Load trained XGBoost model
bst = xgb.Booster()
if os.path.exists(model_json_path):
    bst.load_model(model_json_path)
else:
    bst = None  # Handle missing model scenario

def get_recommendations(member_id):
    """Fetch user details and recommendations using the trained model."""
    # Ensure user exists
    user_row = users_df[users_df["Member_ID"] == member_id]
    if user_row.empty:
        return {"error": "User not found"}
    
    # Extract user details
    user_details = user_row.iloc[0][["Member_ID", "Gender", "Age", "Marital_Status", "Sect", "Caste", "State"]].to_dict()
    user_gender = user_details["Gender"]
    user_age = user_details["Age"]
    user_caste = user_details["Caste"]
    user_sect = user_details["Sect"]
    user_state = user_details["State"]
    
    # Get opposite gender
    opposite_gender_encoded = 1 - user_gender  # Assuming binary gender (0,1)
    
    # Filter opposite-gender users
    eligible_profiles = users_df[users_df["Gender"] == opposite_gender_encoded].copy()

    if eligible_profiles.empty:
        return {"user_details": user_details, "recommended_profiles": [], "statistics": {}}

    # Compute required features
    eligible_profiles["Age_Diff"] = abs(eligible_profiles["Age"] - user_age)
    eligible_profiles["Same_Caste"] = (eligible_profiles["Caste"] == user_caste).astype(int)
    eligible_profiles["Same_Sect"] = (eligible_profiles["Sect"] == user_sect).astype(int)
    eligible_profiles["Same_State"] = (eligible_profiles["State"] == user_state).astype(int)
    eligible_profiles["Target_Popularity"] = 0.5  # Placeholder

    # Ensure we have only the model's required features
    if bst is None:
        return {"error": "Model file not found. Please check `recommendation_model.json`."}
    
    model_features = bst.feature_names
    try:
        X_test = eligible_profiles[model_features]
    except KeyError as e:
        return {"error": f"Feature mismatch: {str(e)}"}

    # Convert to DMatrix for XGBoost
    dtest = xgb.DMatrix(X_test)

    # Get predictions
    preds = bst.predict(dtest)

    # Rank profiles by prediction score
    ranked_profiles = sorted(
        zip(eligible_profiles["Member_ID"], preds), key=lambda x: x[1], reverse=True
    )[:100]

    recommended_ids = [profile_id for profile_id, _ in ranked_profiles]

    # Compute statistics
    recommended_profiles_df = users_df[users_df["Member_ID"].isin(recommended_ids)]
    age_distribution = dict(Counter(recommended_profiles_df["Age"]))
    sect_distribution = dict(Counter(recommended_profiles_df["Sect"]))
    state_distribution = dict(Counter(recommended_profiles_df["State"]))

    return {
        "user_details": user_details,
        "recommended_profiles": recommended_ids,
        "statistics": {
            "age_distribution": age_distribution,
            "sect_distribution": sect_distribution,
            "state_distribution": state_distribution,
        },
    }

@app.get("/recommend/{member_id}")
def recommend_profiles(member_id: int):
    """API endpoint to get user details and recommendations."""
    return get_recommendations(member_id)

# Vercel handler
def handler():
    return app
