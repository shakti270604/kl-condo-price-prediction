# ============================================================
# STEP 11: MODEL TRAINING, HYPERPARAMETER OPTIMIZATION & EVALUATION (FINAL)
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
import joblib

# --- Optional: XGBoost and ANN ---
try:
    import xgboost as xgb
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam

    HAS_XGB = True
    HAS_ANN = True
except ImportError:
    HAS_XGB = False
    HAS_ANN = False
    print("⚠️ Install xgboost and tensorflow for full functionality.")

# --- 1️⃣ Load Preprocessed Data ---
train_scaled = pd.read_csv("train_preprocessed_scaled.csv")
test_scaled = pd.read_csv("test_preprocessed_scaled.csv")
train_unscaled = pd.read_csv("train_preprocessed_unscaled.csv")
test_unscaled = pd.read_csv("test_preprocessed_unscaled.csv")

X_train_scaled = train_scaled.drop(columns=["Target"])
y_train_scaled = train_scaled["Target"]
X_test_scaled = test_scaled.drop(columns=["Target"])
y_test_scaled = test_scaled["Target"]

X_train_unscaled = train_unscaled.drop(columns=["Target"])
y_train_unscaled = train_unscaled["Target"]
X_test_unscaled = test_unscaled.drop(columns=["Target"])
y_test_unscaled = test_unscaled["Target"]

print("✅ Data loaded successfully (scaled & unscaled).")

# --- 2️⃣ Define Models + Hyperparameter Grids ---
models = {
    "Linear Regression": (LinearRegression(), {}),  # No hyperparameters to tune
    "Lasso Regression": (Lasso(random_state=42, max_iter=3000), {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],  # Hyperparameter grid for alpha
    }),
    "Random Forest": (RandomForestRegressor(random_state=42, n_jobs=-1), {
        "n_estimators": [200, 300],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
    }),
}

if HAS_XGB:
    models["XGBoost"] = (
        xgb.XGBRegressor(random_state=42, n_jobs=-1),
        {
            "n_estimators": [200, 300],
            "learning_rate": [0.05, 0.1],
            "max_depth": [4, 6, 8],
            "subsample": [0.8, 1.0],
        },
    )

print("✅ Models and parameter grids prepared.")

# --- 3️⃣ Train + Optimize + Evaluate ---
results = []
trained_models = {}  # Store fitted models

for name, (model, param_grid) in models.items():
    print(f"\n🚀 Training model: {name}")

    # Choose correct dataset
    if name in ["Linear Regression", "Lasso Regression"]:
        X_train, y_train, X_test, y_test = X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled
    else:
        X_train, y_train, X_test, y_test = X_train_unscaled, y_train_unscaled, X_test_unscaled, y_test_unscaled

    # Grid search if applicable
    if param_grid:
        grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, scoring="neg_mean_squared_error")
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        print(f"🔧 Best Params: {grid.best_params_}")
    else:
        model.fit(X_train, y_train)

    trained_models[name] = model  # Save trained version

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({"Model": name, "RMSE": rmse, "MAE": mae, "R²": r2})
    print(f"{name} → RMSE: {rmse:,.2f}, MAE: {mae:,.2f}, R²: {r2:.3f}")

# --- 4️⃣ ANN (Scaled Data Only) - IMPROVED VERSION ---
# --- 4️⃣ ANN (Scaled Data Only) WITH HYPERPARAMETER TUNING ---
if HAS_ANN:
    import keras_tuner as kt

    print("\n🤖 Hyperparameter Tuning for Artificial Neural Network (ANN)...")

    # -------- Build Model Function for Keras Tuner --------
    def build_ann(hp):
        model = Sequential()

        # Input layer
        model.add(Dense(
            units=hp.Int("units_input", min_value=32, max_value=128, step=32),
            activation="relu",
            input_dim=X_train_scaled.shape[1]
        ))

        # Hidden layers (1–3 layers)
        for i in range(hp.Int("num_layers", 1, 3)):
            model.add(Dense(
                units=hp.Int(f"units_{i}", min_value=16, max_value=128, step=16),
                activation="relu"
            ))
            model.add(Dropout(hp.Float(f"dropout_{i}", 0.0, 0.4, step=0.1)))

        # Output layer
        model.add(Dense(1))

        # Compile with different learning rates
        model.compile(
            optimizer=Adam(
                learning_rate=hp.Choice("learning_rate", [0.001, 0.0005, 0.0001])
            ),
            loss="mse",
            metrics=["mae"],
        )
        return model

    # -------- Keras Tuner Random Search --------
    tuner = kt.RandomSearch(
        build_ann,
        objective="val_loss",
        max_trials=10,
        executions_per_trial=1,
        directory="ann_tuning",
        project_name="property_price_ann"
    )

    tuner.search(
        X_train_scaled, y_train_scaled,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[EarlyStopping(monitor="val_loss", patience=10)],
        verbose=1
    )

    # Get best model
    ann = tuner.get_best_models(num_models=1)[0]

    print("\n🎯 Best ANN Hyperparameters:")
    print(tuner.get_best_hyperparameters()[0].values)

    # Predict and evaluate
    y_pred_ann = ann.predict(X_test_scaled).flatten()

    rmse = np.sqrt(mean_squared_error(y_test_scaled, y_pred_ann))
    mae = mean_absolute_error(y_test_scaled, y_pred_ann)
    r2 = r2_score(y_test_scaled, y_pred_ann)

    results.append({"Model": "ANN (Tuned)", "RMSE": rmse, "MAE": mae, "R²": r2})
    trained_models["ANN (Tuned)"] = ann

    print(f"ANN (Tuned) → RMSE: {rmse:,.2f}, MAE: {mae:,.2f}, R²: {r2:.3f}")

# --- 5️⃣ Compare Model Results ---
eval_df = pd.DataFrame(results).sort_values("RMSE")
print("\n🏁 MODEL PERFORMANCE SUMMARY:")
print(eval_df)

best_model_name = eval_df.iloc[0]["Model"]
print(f"\n🎯 Best Model: {best_model_name}")

# --- 6️⃣ Refit Best Random Forest (Guaranteed Fitted for Streamlit) ---
if best_model_name == "Random Forest":
    print("\n🧩 Re-training final Random Forest for deployment...")
    rf_best = RandomForestRegressor(
        n_estimators=300,
        min_samples_split=5,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    rf_best.fit(pd.concat([X_train_unscaled, X_test_unscaled]), pd.concat([y_train_unscaled, y_test_unscaled]))
    best_model = rf_best
else:
    best_model = trained_models[best_model_name]

# --- 7️⃣ Save Best Model ---
joblib.dump(best_model, "best_model.pkl")
joblib.dump(X_train_unscaled.columns.tolist(), "model_columns.pkl")

print(f"\n✅ Best model ({best_model_name}) saved successfully!")
print("💾 Files generated: best_model.pkl, model_columns.pkl")