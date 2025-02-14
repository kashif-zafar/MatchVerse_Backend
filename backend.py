from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import xgboost as xgb
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import os

app = FastAPI()

# Enable CORS (Modify if hosted elsewhere)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load user metadata (Handle file path for Vercel)
users_file = os.path.join(os.path.dirname(__file__), "users.csv")
users_df = pd.read_csv(users_file)

# Label Encoding for categorical variables
label_encoders = {}
for col in ["Gender", "Marital_Status", "Sect", "Caste", "State"]:
    le = LabelEncoder()
    users_df[col] = le.fit_transform(users_df[col])
    label_encoders[col] = le

# Load trained XGBoost model (Handle missing file error)
model_file = os.path.join(os.path.dirname(__file__), "recommendation_model.json")
bst = xgb.Booster()
if os.path.exists(model_file):
    bst.load_model(model_file)
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
        return {"error": "Model file not found. Please upload `recommendation_model.json`."}
    
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
