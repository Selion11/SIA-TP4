import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def load_europe_data(csv_path):
    """Load europe.csv, keep country names, and return standardized numeric features."""
    df = pd.read_csv(csv_path)

    if "Country" not in df.columns:
        raise ValueError("Expected a 'Country' column in europe.csv.")

    country_names = df["Country"].astype(str)
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_columns:
        raise ValueError("No numeric columns found in europe.csv.")

    x_raw = df[numeric_columns].to_numpy(dtype=float)

    # Standardization is required because variables have very different scales.
    scaler = StandardScaler()
    x_std = scaler.fit_transform(x_raw)

    return country_names, numeric_columns, x_std


def train_oja(x, epochs=500, learning_rate=0.01, lr_decay=0.995, seed=42):
    """Train a single Oja neuron to estimate the first principal component."""
    if x.ndim != 2:
        raise ValueError("Input data must be a 2D array.")

    n_features = x.shape[1]
    rng = np.random.default_rng(seed)

    w = rng.normal(0.0, 1.0, size=n_features)
    w /= np.linalg.norm(w)

    for epoch in range(epochs):
        eta = learning_rate * (lr_decay ** epoch)
        for i in rng.permutation(x.shape[0]):
            xi = x[i]
            y = np.dot(w, xi)
            w += eta * (y * xi - (y ** 2) * w)

            norm_w = np.linalg.norm(w)
            if norm_w > 0:
                w /= norm_w

    return w / np.linalg.norm(w)


def build_loadings_table(feature_names, weights):
    loadings = pd.DataFrame(
        {
            "feature": feature_names,
            "pc1_weight": weights,
            "abs_weight": np.abs(weights),
        }
    )
    return loadings.sort_values("abs_weight", ascending=False).reset_index(drop=True)


def build_country_scores(country_names, x_std, weights):
    scores = x_std @ weights
    result = pd.DataFrame({"country": country_names, "pc1_score": scores})
    return result.sort_values("pc1_score", ascending=False).reset_index(drop=True)


def interpret_component(loadings_table):
    top = loadings_table.head(3)
    positive = loadings_table.sort_values("pc1_weight", ascending=False).head(3)
    negative = loadings_table.sort_values("pc1_weight", ascending=True).head(3)

    top_names = ", ".join(top["feature"].tolist())
    pos_text = ", ".join(f"{row.feature} ({row.pc1_weight:.3f})" for row in positive.itertuples())
    neg_text = ", ".join(f"{row.feature} ({row.pc1_weight:.3f})" for row in negative.itertuples())

    lines = [
        "PC1 interpretation (Oja):",
        f"- Variables with highest absolute contribution: {top_names}.",
        f"- Positive side of PC1 is mainly associated with: {pos_text}.",
        f"- Negative side of PC1 is mainly associated with: {neg_text}.",
        "- Countries with high PC1 scores tend to have high values on positive-load features",
        "  and/or low values on negative-load features (after standardization).",
        "- Note: the sign of a principal component is arbitrary; only relative directions matter.",
    ]
    return "\n".join(lines)


def plot_oja_results(loadings_table, country_scores, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    loadings_sorted = loadings_table.sort_values("pc1_weight")
    colors = ["#1f77b4" if w >= 0 else "#d62728" for w in loadings_sorted["pc1_weight"]]

    ax1.barh(loadings_sorted["feature"], loadings_sorted["pc1_weight"], color=colors)
    ax1.axvline(0, color="black", linewidth=1)
    ax1.set_title("PC1 Weights Learned by Oja")
    ax1.set_xlabel("Weight")
    ax1.set_ylabel("Feature")

    score_view = country_scores.sort_values("pc1_score")
    score_colors = ["#1f77b4" if s >= 0 else "#d62728" for s in score_view["pc1_score"]]

    ax2.barh(score_view["country"], score_view["pc1_score"], color=score_colors)
    ax2.axvline(0, color="black", linewidth=1)
    ax2.set_title("Countries Projected on PC1")
    ax2.set_xlabel("PC1 score")
    ax2.set_ylabel("Country")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(project_root, "data", "europe.csv")
    output_dir = os.path.join(project_root, "oja", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    country_names, feature_names, x_std = load_europe_data(csv_path)

    weights = train_oja(
        x_std,
        epochs=500,
        learning_rate=0.01,
        lr_decay=0.995,
        seed=42,
    )

    loadings_table = build_loadings_table(feature_names, weights)
    country_scores = build_country_scores(country_names, x_std, weights)

    loadings_csv = os.path.join(output_dir, "oja_pc1_loadings.csv")
    scores_csv = os.path.join(output_dir, "oja_country_scores.csv")
    figure_png = os.path.join(output_dir, "oja_pc1_analysis.png")

    loadings_table.to_csv(loadings_csv, index=False)
    country_scores.to_csv(scores_csv, index=False)
    plot_oja_results(loadings_table, country_scores, figure_png)

    print("=== Oja Model - Exercise 1.2 ===")
    print(f"Dataset: {csv_path}")
    print(f"Samples: {x_std.shape[0]} | Numeric features: {x_std.shape[1]}")

    print("\nFinal normalized weight vector (PC1 estimate):")
    print(np.array2string(weights, precision=6, suppress_small=True))

    print("\nPC1 loadings (sorted by absolute value):")
    print(loadings_table[["feature", "pc1_weight", "abs_weight"]].to_string(index=False))

    print("\nTop 5 countries with highest PC1 score:")
    print(country_scores.head(5).to_string(index=False))

    print("\nTop 5 countries with lowest PC1 score:")
    print(country_scores.tail(5).sort_values("pc1_score").to_string(index=False))

    print("\n" + interpret_component(loadings_table))

    print("\nSaved files:")
    print(f"- {loadings_csv}")
    print(f"- {scores_csv}")
    print(f"- {figure_png}")


if __name__ == "__main__":
    main()
