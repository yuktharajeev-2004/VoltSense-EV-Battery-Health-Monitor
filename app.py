from pathlib import Path
import math
import joblib
import pandas as pd

from flask import Flask, render_template, request

app = Flask(__name__)

# Project folder
BASE_DIR = Path(__file__).resolve().parent

# Load your existing trained files
model = joblib.load(BASE_DIR / "Random forest.pkl")
encoder1 = joblib.load(BASE_DIR / "Obj1_encoder.pkl")
encoder2 = joblib.load(BASE_DIR / "Obj2_encoder.pkl")
scaler = joblib.load(BASE_DIR / "StandardScaler.pkl")
# Keep exactly the same feature order as your training code
FEATURES = [
    "manufacturer",
    "chemistry_type",
    "nominal_capacity_Ah",
    "nominal_voltage_V",
    "cycle_number",
    "voltage_V",
    "current_A",
    "c_rate",
    "state_of_charge_%",
    "depth_of_discharge_%",
    "cell_temperature_C",
    "internal_resistance_mOhm",
]

# Choices from your saved LabelEncoders
manufacturers = encoder1.classes_.tolist()
chemistries = encoder2.classes_.tolist()

# Default input values
defaults = {
    "nominal_capacity_Ah": 3.2,
    "nominal_voltage_V": 3.7,
    "cycle_number": 500,
    "voltage_V": 3.6,
    "current_A": 0.5,
    "c_rate": 0.5,
    "state_of_charge_%": 70,
    "depth_of_discharge_%": 30,
    "cell_temperature_C": 25,
    "internal_resistance_mOhm": 52,
}


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    form = {
        "manufacturer": "",
        "chemistry_type": "",
        **{key: "" for key in defaults},
    }

    if request.method == "POST":
        form.update(request.form.to_dict())

        try:
            manufacturer = form["manufacturer"]
            chemistry = form["chemistry_type"]

            if manufacturer not in manufacturers:
                raise ValueError("Please select a valid manufacturer.")

            if chemistry not in chemistries:
                raise ValueError("Please select a valid battery chemistry.")

            # Read and validate numeric inputs
            numeric = {}

            for key in defaults:
                value = float(form[key])

                if not math.isfinite(value):
                    raise ValueError(f"{key} must be a valid finite number.")

                numeric[key] = value

            if numeric["nominal_capacity_Ah"] <= 0:
                raise ValueError("Nominal capacity must be positive.")

            if numeric["nominal_voltage_V"] <= 0:
                raise ValueError("Nominal voltage must be positive.")

            if numeric["cycle_number"] < 0:
                raise ValueError("Cycle number cannot be negative.")

            if not 0 <= numeric["state_of_charge_%"] <= 100:
                raise ValueError("State of charge must be between 0 and 100.")

            if not 0 <= numeric["depth_of_discharge_%"] <= 100:
                raise ValueError(
                    "Depth of discharge must be between 0 and 100."
                )

            # Build input row in the original training feature order
            row = {
                "manufacturer": manufacturer,
                "chemistry_type": chemistry,
                **numeric,
            }

            input_data = pd.DataFrame([row], columns=FEATURES)

            # Apply the SAME LabelEncoders used during training
            input_data["manufacturer"] = encoder1.transform(
                input_data["manufacturer"]
            )

            input_data["chemistry_type"] = encoder2.transform(
                input_data["chemistry_type"]
            )

            # Apply the SAME fitted scaler; do not fit it again
            scaled_input = scaler.transform(input_data)

            # Predict using the existing trained Random Forest
            prediction = float(model.predict(scaled_input)[0])

            if not math.isfinite(prediction):
                raise ValueError("The model returned an invalid prediction.")

            displayed_soh = max(0.0, min(100.0, prediction))

            if displayed_soh >= 80:
                status = "Healthy"
                guidance = (
                    "The predicted SOH is in a generally healthy range."
                )
            elif displayed_soh >= 60:
                status = "Moderate"
                guidance = (
                    "Monitor battery performance and degradation."
                )
            else:
                status = "Low"
                guidance = (
                    "Consider a qualified battery health inspection."
                )

            
            result = {
                "soh": round(displayed_soh, 2),
                "raw_soh": round(prediction, 4),
                "status": status,
                "guidance": guidance
            }

            # Reset dropdowns after a successful prediction
            form["manufacturer"] = ""
            form["chemistry_type"] = ""

        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc) or "Please check your input values."

    return render_template(
        "index.html",
        result=result,
        error=error,
        form=form,
        manufacturers=manufacturers,
        chemistries=chemistries,
        metrics=None,
    )


if __name__ == "__main__":
    app.run(debug=True)
