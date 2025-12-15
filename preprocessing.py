import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/cleaned_all_participants.csv")

features = [
    "calories", "heart_rate", "steps",
    "basal_rate", "bolus_volume_delivered", "carb_input"
]
target = "glucose"

X = df[features]
y = df[target]

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (time-aware)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=False
)

print("Preprocessing completed successfully")
