"""Score a new patient with the trained model.
Usage:  python predict.py            (runs a demo patient)
        or import predict_patient(dict) from another script.
"""
import joblib, pandas as pd

_bundle = joblib.load("models/heart_model.joblib")

def predict_patient(p: dict) -> dict:
    X = pd.DataFrame([p])[_bundle["features"]]
    prob = float(_bundle["model"].predict_proba(X)[0, 1])
    thr = _bundle["screening_threshold"]
    return {"probability_of_heart_disease": round(prob, 3),
            "prediction_at_0.5": int(prob >= 0.5),
            "screening_flag": int(prob >= thr),
            "risk_band": "HIGH" if prob >= 0.7 else "MODERATE" if prob >= 0.4 else "LOW"}

if __name__ == "__main__":
    demo = dict(age=58, sex=1, cp=0, trestbps=150, chol=270, fbs=0, restecg=0,
                thalach=111, exang=1, oldpeak=2.5, slope=1, ca=2, thal=3)
    print(predict_patient(demo))
