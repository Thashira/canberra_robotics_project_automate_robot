import os
import numpy as np
import joblib

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------------------------------------------------
# 1. Prepare dataset
# --------------------------------------------------

folder = "./processed_dataset"

failed = {"episode_007", "episode_021"}

episodes = sorted(
    d for d in os.listdir(folder)
    if d.startswith("episode_") and d not in failed
)

rng = np.random.default_rng(42)
rng.shuffle(episodes)

train_eps = episodes[:15]
val_eps = episodes[15:17]
test_eps = episodes[17:19]


def load_episodes(selected):
    X_list, y_list = [], []

    for ep in selected:
        path = os.path.join(folder, ep)

        states = np.load(os.path.join(path, "states.npy"))
        actions = np.load(os.path.join(path, "actions.npy"))

        X_list.append(states[:-1])
        y_list.append(actions)

    return np.concatenate(X_list), np.concatenate(y_list)


X_train, y_train = load_episodes(train_eps)
X_val, y_val = load_episodes(val_eps)
X_test, y_test = load_episodes(test_eps)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 2. Normalize input states
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# 3. Create MLP model
# --------------------------------------------------

model = MLPRegressor(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=100,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42,
    verbose=True
)


# --------------------------------------------------
# 4. Train
# --------------------------------------------------

print("\nStarting Behavior Cloning training...\n")

model.fit(X_train_scaled, y_train)


# --------------------------------------------------
# 5. Evaluate
# --------------------------------------------------

val_predictions = model.predict(X_val_scaled)
test_predictions = model.predict(X_test_scaled)

print("\n========== VALIDATION RESULTS ==========")

print("MAE:", mean_absolute_error(y_val, val_predictions))
print("MSE:", mean_squared_error(y_val, val_predictions))

print("\n========== TEST RESULTS ==========")

print("MAE:", mean_absolute_error(y_test, test_predictions))
print("MSE:", mean_squared_error(y_test, test_predictions))


# --------------------------------------------------
# 6. Save model
# --------------------------------------------------

joblib.dump(model, "bc_model.pkl")
joblib.dump(scaler, "state_scaler.pkl")

print("\nModel saved successfully!")

print("\n========== PER-ACTION EVALUATION ==========")

action_names = [
    "Action 1",
    "Action 2",
    "Action 3",
    "Action 4",
    "Action 5",
    "Action 6",
    "Action 7"
]

for i, name in enumerate(action_names):

    mae = mean_absolute_error(
        y_test[:, i],
        test_predictions[:, i]
    )

    mse = mean_squared_error(
        y_test[:, i],
        test_predictions[:, i]
    )

    print(f"{name}: MAE={mae:.6f}, MSE={mse:.6f}")