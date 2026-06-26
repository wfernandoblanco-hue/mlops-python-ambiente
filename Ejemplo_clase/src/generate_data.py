import argparse

import numpy as np
import pandas as pd

from src.config import ensure_parent, load_params


SEGMENTS = ["new", "active", "at_risk", "churn"]
DEVICE_OS = ["android", "ios", "web"]
SITES = ["home", "search", "product", "content"]
ENTRY_POINTS = ["home", "search", "recommendation", "notification"]


def build_mock_sessions(n_samples: int, random_state: int, noise_std: float) -> pd.DataFrame:
    """Generate a reproducible mock dataset for session duration regression."""
    rng = np.random.default_rng(random_state)

    data = pd.DataFrame(
        {
            "segment": rng.choice(SEGMENTS, size=n_samples, p=[0.25, 0.45, 0.2, 0.1]),
            "historical_avg_session_minutes": rng.uniform(2.0, 35.0, size=n_samples),
            "historical_sessions_last_7d": rng.poisson(4, size=n_samples),
            "days_since_last_session": rng.integers(0, 31, size=n_samples),
            "hour_of_day": rng.integers(0, 24, size=n_samples),
            "day_of_week": rng.integers(0, 7, size=n_samples),
            "device_os": rng.choice(DEVICE_OS, size=n_samples, p=[0.45, 0.35, 0.2]),
            "site": rng.choice(SITES, size=n_samples),
            "entry_point": rng.choice(ENTRY_POINTS, size=n_samples),
            "push_received_last_24h": rng.integers(0, 2, size=n_samples),
        }
    )

    segment_effect = data["segment"].map(
        {"new": -2.5, "active": 4.0, "at_risk": -4.0, "churn": -7.5}
    )
    entry_effect = data["entry_point"].map(
        {"home": 1.5, "search": 2.0, "recommendation": 5.5, "notification": 3.0}
    )
    site_effect = data["site"].map({"home": 1.0, "search": 1.5, "product": 2.5, "content": 3.0})
    evening_effect = np.where(data["hour_of_day"].between(18, 22), 3.0, 0.0)
    weekend_effect = np.where(data["day_of_week"].isin([5, 6]), 2.0, 0.0)
    noise = rng.normal(0.0, noise_std, size=n_samples)

    session_minutes = (
        6.0
        + 0.58 * data["historical_avg_session_minutes"]
        + 0.85 * data["historical_sessions_last_7d"]
        - 0.18 * data["days_since_last_session"]
        + 2.7 * data["push_received_last_24h"]
        + segment_effect
        + entry_effect
        + site_effect
        + evening_effect
        + weekend_effect
        + noise
    )

    data["session_minutes"] = np.clip(session_minutes, 1.0, None).round(2)
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    data_params = params["data"]

    data = build_mock_sessions(
        n_samples=int(data_params["n_samples"]),
        random_state=int(data_params["random_state"]),
        noise_std=float(data_params["noise_std"]),
    )

    output_path = ensure_parent(data_params["raw_path"])
    data.to_csv(output_path, index=False)
    print(f"Dataset mock creado en {output_path} con {len(data)} filas.")


if __name__ == "__main__":
    main()