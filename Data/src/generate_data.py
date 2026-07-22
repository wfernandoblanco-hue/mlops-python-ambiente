import argparse

import numpy as np
import pandas as pd

from config import ensure_parent, load_params

SEGMENTS = ["new", "active", "at_risk", "churn"]
DEVICE_OS = ["android", "ios", "web"]
SITES = ["home", "search", "product", "content"]
CAMPAIGN_TYPES = ["promotional", "transactional", "reengagement", "newsletter"]


def build_mock_push_notifications(
    n_samples: int, random_state: int, noise_std: float
) -> pd.DataFrame:
    """Generate a reproducible mock dataset for push notification open classification."""
    rng = np.random.default_rng(random_state)

    # 1. Creación de variables especificadas
    data = pd.DataFrame(
        {
            "user_id": [f"usr_{i:06d}" for i in range(1, n_samples + 1)],
            "site": rng.choice(SITES, size=n_samples),
            "campaign_type": rng.choice(
                CAMPAIGN_TYPES, size=n_samples, p=[0.40, 0.25, 0.20, 0.15]
            ),
            "device_os": rng.choice(
                DEVICE_OS, size=n_samples, p=[0.45, 0.35, 0.20]
            ),
            "hour_of_day": rng.integers(0, 24, size=n_samples),
            "day_of_week": rng.integers(0, 7, size=n_samples),
            # Distribución Beta para simular tasas de apertura reales (sesgadas hacia valores bajos)
            "historical_open_rate": rng.beta(a=2, b=8, size=n_samples).round(4),
            "historical_push_count": rng.integers(1, 100, size=n_samples),
            "days_since_last_open": rng.integers(0, 61, size=n_samples),
            "segment": rng.choice(
                SEGMENTS, size=n_samples, p=[0.25, 0.45, 0.20, 0.10]
            ),
        }
    )

    # 2. Mapeo de efectos categóricos sobre el Log-Odds (Logit)
    campaign_effect = data["campaign_type"].map(
        {
            "transactional": 1.2,
            "reengagement": 0.3,
            "promotional": 0.0,
            "newsletter": -0.3,
        }
    )
    segment_effect = data["segment"].map(
        {"active": 0.5, "new": 0.0, "at_risk": -0.5, "churn": -1.2}
    )
    peak_hour_effect = np.where(
        data["hour_of_day"].between(10, 14)
        | data["hour_of_day"].between(18, 21),
        0.4,
        0.0,
    )

    noise = rng.normal(0.0, noise_std, size=n_samples)

    # 3. Cálculo de los Log-Odds (z)
    log_odds = (
        -1.8  # Sesgo base (tasa de conversión ~15-20%)
        + 3.5 * data["historical_open_rate"]
        - 0.03 * data["days_since_last_open"]
        + campaign_effect
        + segment_effect
        + peak_hour_effect
        + noise
    )

    # 4. Transformación Sigmoide a probabilidades [0, 1]
    probabilities = 1 / (1 + np.exp(-log_odds))

    # 5. Generación del target binario (0 o 1) mediante ensayo de Bernoulli
    data["target_opened"] = rng.binomial(n=1, p=probabilities)

    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    data_params = params["data"]

    data = build_mock_push_notifications(
        n_samples=int(data_params["n_samples"]),
        random_state=int(data_params["random_state"]),
        noise_std=float(data_params["noise_std"]),
    )

    output_path = ensure_parent(data_params["raw_path"])
    data.to_csv(output_path, index=False)
    print(f"Dataset mock creado en {output_path} con {len(data)} filas.")


if __name__ == "__main__":
    main()