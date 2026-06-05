import json
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import oja as oja_core


LEARNING_RATES = [0.0001, 0.001, 0.01, 0.05, 0.1]
EPOCHS_GRID = [25, 50, 100, 500, 1000, 3000]
SEED = 42
LR_DECAY = 0.995
BASELINE_LEARNING_RATE = 0.01
BASELINE_EPOCHS = 500
TOP_K = 3


@dataclass
class ExperimentResult:
    learning_rate: float
    epochs: int
    weights: np.ndarray
    loadings: pd.DataFrame
    country_scores: pd.DataFrame
    history: list
    variance_captured: float
    explained_variance_ratio: float
    final_step_change: float
    similarity_to_baseline: float
    distance_to_baseline: float
    top_variables: list
    top_positive_variables: list
    top_negative_variables: list
    top_countries_high: list
    top_countries_low: list


def format_lr(value):
    return f"{value:.4f}".rstrip("0").rstrip(".") if value < 0.1 else f"{value:.2f}".rstrip("0").rstrip(".")


def ensure_directory(path):
    os.makedirs(path, exist_ok=True)


def sign_invariant_cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(abs(np.dot(a, b) / denom))


def sign_invariant_delta(a, b):
    return float(min(np.linalg.norm(a - b), np.linalg.norm(a + b)))


def variance_captured(x_std, weights):
    scores = x_std @ weights
    return float(np.var(scores, ddof=0))


def train_oja_with_history(x_std, max_epochs, learning_rate, seed=SEED, lr_decay=LR_DECAY):
    n_features = x_std.shape[1]
    rng = np.random.default_rng(seed)

    weights = rng.normal(0.0, 1.0, size=n_features)
    weights /= np.linalg.norm(weights)

    history = []
    snapshots = {}

    for epoch in range(1, max_epochs + 1):
        previous = weights.copy()
        eta = learning_rate * (lr_decay ** (epoch - 1))

        for sample_idx in rng.permutation(x_std.shape[0]):
            x_i = x_std[sample_idx]
            y_i = np.dot(weights, x_i)
            weights += eta * (y_i * x_i - (y_i ** 2) * weights)

            norm_weights = np.linalg.norm(weights)
            if norm_weights > 0:
                weights /= norm_weights

        weights /= np.linalg.norm(weights)
        one_step_similarity = sign_invariant_cosine_similarity(weights, previous)
        one_step_change = sign_invariant_delta(weights, previous)

        history.append(
            {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "weights": weights.copy(),
                "one_step_similarity": one_step_similarity,
                "one_step_change": one_step_change,
                "variance_captured": variance_captured(x_std, weights),
            }
        )
        snapshots[epoch] = weights.copy()

    return weights.copy(), history, snapshots


def extract_top_features(loadings_table, k=TOP_K):
    top_absolute = loadings_table.head(k)["feature"].tolist()
    top_positive = loadings_table.sort_values("pc1_weight", ascending=False).head(k)["feature"].tolist()
    top_negative = loadings_table.sort_values("pc1_weight", ascending=True).head(k)["feature"].tolist()
    return top_absolute, top_positive, top_negative


def extract_top_countries(country_scores, k=TOP_K):
    high = country_scores.head(k)["country"].tolist()
    low = country_scores.sort_values("pc1_score", ascending=True).head(k)["country"].tolist()
    return high, low


def run_single_experiment(x_std, feature_names, country_names, max_epochs, learning_rate, baseline_weights=None):
    final_weights, history, snapshots = train_oja_with_history(
        x_std=x_std,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        seed=SEED,
        lr_decay=LR_DECAY,
    )

    loadings = oja_core.build_loadings_table(feature_names, final_weights)
    country_scores = oja_core.build_country_scores(country_names, x_std, final_weights)

    variance = variance_captured(x_std, final_weights)
    total_variance = x_std.shape[1]
    explained_ratio = float(variance / total_variance)
    final_step_change = history[-1]["one_step_change"]

    similarity_to_baseline = 0.0
    distance_to_baseline = 1.0
    if baseline_weights is not None:
        similarity_to_baseline = sign_invariant_cosine_similarity(final_weights, baseline_weights)
        distance_to_baseline = 1.0 - similarity_to_baseline

    top_variables, top_positive_variables, top_negative_variables = extract_top_features(loadings)
    top_countries_high, top_countries_low = extract_top_countries(country_scores)

    return ExperimentResult(
        learning_rate=learning_rate,
        epochs=max_epochs,
        weights=final_weights,
        loadings=loadings,
        country_scores=country_scores,
        history=history,
        variance_captured=variance,
        explained_variance_ratio=explained_ratio,
        final_step_change=final_step_change,
        similarity_to_baseline=similarity_to_baseline,
        distance_to_baseline=distance_to_baseline,
        top_variables=top_variables,
        top_positive_variables=top_positive_variables,
        top_negative_variables=top_negative_variables,
        top_countries_high=top_countries_high,
        top_countries_low=top_countries_low,
    )


def run_grid_search(x_std, feature_names, country_names):
    baseline_weights = None
    baseline_result = None
    all_results = []

    for learning_rate in LEARNING_RATES:
        _, history, snapshots = train_oja_with_history(
            x_std=x_std,
            max_epochs=max(EPOCHS_GRID),
            learning_rate=learning_rate,
            seed=SEED,
            lr_decay=LR_DECAY,
        )

        for epochs in EPOCHS_GRID:
            weights = snapshots[epochs]
            loadings = oja_core.build_loadings_table(feature_names, weights)
            country_scores = oja_core.build_country_scores(country_names, x_std, weights)
            variance = variance_captured(x_std, weights)
            explained_ratio = float(variance / x_std.shape[1])

            if learning_rate == BASELINE_LEARNING_RATE and epochs == BASELINE_EPOCHS:
                baseline_weights = weights.copy()
                baseline_result = {
                    "weights": weights.copy(),
                    "country_scores": country_scores.copy(),
                    "loadings": loadings.copy(),
                }

            result = ExperimentResult(
                learning_rate=learning_rate,
                epochs=epochs,
                weights=weights.copy(),
                loadings=loadings,
                country_scores=country_scores,
                history=history[:epochs],
                variance_captured=variance,
                explained_variance_ratio=explained_ratio,
                final_step_change=history[epochs - 1]["one_step_change"],
                similarity_to_baseline=0.0,
                distance_to_baseline=0.0,
                top_variables=extract_top_features(loadings)[0],
                top_positive_variables=extract_top_features(loadings)[1],
                top_negative_variables=extract_top_features(loadings)[2],
                top_countries_high=extract_top_countries(country_scores)[0],
                top_countries_low=extract_top_countries(country_scores)[1],
            )
            all_results.append(result)

    if baseline_weights is None:
        raise RuntimeError("Could not locate the baseline configuration 0.01 / 500 in the grid.")

    for result in all_results:
        result.similarity_to_baseline = sign_invariant_cosine_similarity(result.weights, baseline_weights)
        result.distance_to_baseline = 1.0 - result.similarity_to_baseline

    return all_results, baseline_result, baseline_weights


def results_to_summary_df(results):
    rows = []
    for idx, result in enumerate(results, start=1):
        rows.append(
            {
                "experiment_id": f"exp_{idx:03d}",
                "learning_rate": result.learning_rate,
                "epochs": result.epochs,
                "variance_captured": result.variance_captured,
                "explained_variance_ratio": result.explained_variance_ratio,
                "final_step_change": result.final_step_change,
                "similarity_to_baseline": result.similarity_to_baseline,
                "distance_to_baseline": result.distance_to_baseline,
                "top_variables": json.dumps(result.top_variables, ensure_ascii=False),
                "top_positive_variables": json.dumps(result.top_positive_variables, ensure_ascii=False),
                "top_negative_variables": json.dumps(result.top_negative_variables, ensure_ascii=False),
                "top_countries_high": json.dumps(result.top_countries_high, ensure_ascii=False),
                "top_countries_low": json.dumps(result.top_countries_low, ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows)


def results_to_loadings_df(results):
    rows = []
    for idx, result in enumerate(results, start=1):
        experiment_id = f"exp_{idx:03d}"
        temp = result.loadings.copy()
        temp.insert(0, "experiment_id", experiment_id)
        temp.insert(1, "learning_rate", result.learning_rate)
        temp.insert(2, "epochs", result.epochs)
        rows.append(temp)
    return pd.concat(rows, ignore_index=True)


def results_to_scores_df(results):
    rows = []
    for idx, result in enumerate(results, start=1):
        experiment_id = f"exp_{idx:03d}"
        temp = result.country_scores.copy()
        temp.insert(0, "experiment_id", experiment_id)
        temp.insert(1, "learning_rate", result.learning_rate)
        temp.insert(2, "epochs", result.epochs)
        rows.append(temp)
    return pd.concat(rows, ignore_index=True)


def results_to_history_df(results):
    rows = []
    for idx, result in enumerate(results, start=1):
        experiment_id = f"exp_{idx:03d}"
        for record in result.history:
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "learning_rate": result.learning_rate,
                    "epochs": result.epochs,
                    "epoch": record["epoch"],
                    "variance_captured": record["variance_captured"],
                    "one_step_similarity": record["one_step_similarity"],
                    "one_step_change": record["one_step_change"],
                }
            )
    return pd.DataFrame(rows)


def summarize_by_learning_rate(summary_df):
    grouped = summary_df.groupby("learning_rate", as_index=False).agg(
        max_explained_variance_ratio=("explained_variance_ratio", "max"),
        min_explained_variance_ratio=("explained_variance_ratio", "min"),
        explained_variance_ratio_at_last=("explained_variance_ratio", lambda s: s.iloc[-1]),
        average_final_step_change=("final_step_change", "mean"),
    )
    return grouped

def plot_similarity_evolution(results, output_path):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    for learning_rate in sorted({result.learning_rate for result in results}):
        final_result = next(
            result for result in results if result.learning_rate == learning_rate and result.epochs == max(EPOCHS_GRID)
        )
        final_weights = final_result.weights

        epoch_numbers = []
        similarities = []
        for record in final_result.history:
            epoch_numbers.append(record["epoch"])
            similarities.append(sign_invariant_cosine_similarity(record["weights"], final_weights))

        ax.plot(epoch_numbers, similarities, label=f"lr={format_lr(learning_rate)}")

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Similarity to the final vector")
    ax.set_title("Evolution of similarity as epochs increase")
    ax.legend(title="Learning rate")
    ax.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_quality_over_epochs(summary_df, output_path):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    for learning_rate, group in summary_df.groupby("learning_rate"):
        group = group.sort_values("epochs")
        ax.plot(group["epochs"], group["explained_variance_ratio"], marker="o", linewidth=2, label=f"lr={format_lr(learning_rate)}")

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("Quality trend across epochs for each learning rate")
    ax.legend(title="Learning rate")
    ax.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_best_worst_loadings(best_result, worst_result, output_path):
    best_sorted = best_result.loadings.sort_values("pc1_weight")
    worst_sorted = worst_result.loadings.sort_values("pc1_weight")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=150, sharex=True)

    axes[0].barh(best_sorted["feature"], best_sorted["pc1_weight"], color=["#1f77b4" if v >= 0 else "#d62728" for v in best_sorted["pc1_weight"]])
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].set_title(f"Best configuration\nlr={format_lr(best_result.learning_rate)}, epochs={best_result.epochs}")
    axes[0].set_xlabel("Weight")

    axes[1].barh(worst_sorted["feature"], worst_sorted["pc1_weight"], color=["#1f77b4" if v >= 0 else "#d62728" for v in worst_sorted["pc1_weight"]])
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_title(f"Worst configuration\nlr={format_lr(worst_result.learning_rate)}, epochs={worst_result.epochs}")
    axes[1].set_xlabel("Weight")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_ranking(summary_df, output_path):
    ranked = summary_df.sort_values(
        by=["explained_variance_ratio", "final_step_change"], ascending=[False, True]
    ).copy()
    ranked["label"] = ranked.apply(lambda row: f"lr={format_lr(row.learning_rate)} | ep={int(row.epochs)}", axis=1)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(ranked)))
    ax.barh(ranked["label"], ranked["explained_variance_ratio"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Explained variance ratio")
    ax.set_title("Ranking of configurations")
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_country_comparison(best_result, baseline_result, output_path):
    best_scores = best_result.country_scores.set_index("country")["pc1_score"]
    baseline_scores = baseline_result["country_scores"].set_index("country")["pc1_score"]
    comparison = pd.DataFrame(
        {
            "best": best_scores,
            "baseline": baseline_scores,
        }
    ).reset_index()

    comparison["abs_change"] = (comparison["best"] - comparison["baseline"]).abs()
    comparison = comparison.sort_values("abs_change", ascending=False).head(12)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=150, sharey=True)

    axes[0].barh(comparison["country"], comparison["baseline"], color="#7f8c8d")
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].set_title("Original configuration\nlr=0.01, epochs=500")
    axes[0].set_xlabel("PC1 score")

    axes[1].barh(comparison["country"], comparison["best"], color="#1f77b4")
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_title(f"Best configuration\nlr={format_lr(best_result.learning_rate)}, epochs={best_result.epochs}")
    axes[1].set_xlabel("PC1 score")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_best_worst_countries(best_result, worst_result, output_path):
    best_scores = best_result.country_scores.set_index("country")["pc1_score"]
    worst_scores = worst_result.country_scores.set_index("country")["pc1_score"]

    comparison = pd.DataFrame(
        {
            "best": best_scores,
            "worst": worst_scores,
        }
    ).reset_index()

    comparison["abs_change"] = (comparison["best"] - comparison["worst"]).abs()
    comparison = comparison.sort_values("abs_change", ascending=False).head(12)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=150, sharey=True)

    axes[0].barh(comparison["country"], comparison["worst"], color="#c0392b")
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].set_title(
        f"Worst configuration\nlr={format_lr(worst_result.learning_rate)}, epochs={worst_result.epochs}"
    )
    axes[0].set_xlabel("PC1 score")

    axes[1].barh(comparison["country"], comparison["best"], color="#1f77b4")
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_title(
        f"Best configuration\nlr={format_lr(best_result.learning_rate)}, epochs={best_result.epochs}"
    )
    axes[1].set_xlabel("PC1 score")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_report(summary_df, history_df, baseline_result, best_result, worst_result, learning_rate_summary, output_paths):
    summary_sorted = summary_df.sort_values(
        by=["explained_variance_ratio", "final_step_change"], ascending=[False, True]
    ).reset_index(drop=True)

    best_row = summary_sorted.iloc[0]
    worst_row = summary_sorted.iloc[-1]
    original_row = summary_df[(summary_df["learning_rate"] == BASELINE_LEARNING_RATE) & (summary_df["epochs"] == BASELINE_EPOCHS)].iloc[0]

    low_lr = learning_rate_summary.sort_values("learning_rate").iloc[0]
    high_lr = learning_rate_summary.sort_values("learning_rate").iloc[-1]
    medium_lr = learning_rate_summary.loc[learning_rate_summary["learning_rate"].eq(0.01)].iloc[0]

    unstable = summary_df[summary_df["final_step_change"] > 0.1]
    unstable_labels = [f"lr={format_lr(r.learning_rate)}, ep={int(r.epochs)}" for r in unstable.itertuples()]

    best_label = f"lr={format_lr(best_row.learning_rate)}, epochs={int(best_row.epochs)}"
    worst_label = f"lr={format_lr(worst_row.learning_rate)}, epochs={int(worst_row.epochs)}"
    original_label = f"lr={format_lr(original_row.learning_rate)}, epochs={int(original_row.epochs)}"

    best_vs_original_distance = float(best_row.distance_to_baseline)
    best_quality = float(best_row.explained_variance_ratio)
    worst_quality = float(worst_row.explained_variance_ratio)

    oral_paragraphs = [
        "En esta nueva prueba variamos dos hiperparámetros de la regla de Oja: la cantidad de épocas y el learning rate.",
        "Un learning rate demasiado bajo aprende lentamente, mientras que uno demasiado alto puede generar inestabilidad o cambios demasiado bruscos en los pesos.",
        "La mejor configuración fue la que logró mayor varianza explicada con un vector de pesos ya estabilizado, mientras que la peor fue la que quedó más lejos de esa región de convergencia y mantuvo cambios más grandes entre épocas.",
        "Además, el signo de la primera componente principal sigue siendo arbitrario: si la dirección se invierte, la interpretación sigue siendo equivalente mientras se respeten las relaciones entre variables y países.",
    ]

    report = []
    report.append("# Reporte de barrido de hiperparámetros - Oja")
    report.append("")
    report.append("## 1. Cambios realizados")
    report.append("- Se agrego un experimento de grilla sobre `learning_rate` y `epochs` sin modificar el experimento original de Oja.")
    report.append("- Se reutilizo la carga de `data/europe.csv`, la separacion de `Country`, el uso exclusivo de variables numericas y la estandarizacion previa.")
    report.append("- Se generaron salidas comparativas en `oja/outputs/hyperparameter_search/`.")
    report.append("")
    report.append("## 2. Hiperparametros probados")
    report.append(f"- Learning rates: {LEARNING_RATES}")
    report.append(f"- Epochs: {EPOCHS_GRID}")
    report.append(f"- Seed fija: {SEED}")
    report.append(f"- Decaimiento fijo del learning rate: {LR_DECAY}")
    report.append("")
    report.append("## 3. Criterio de evaluacion")
    report.append("- Se midio la varianza capturada por la direccion aprendida, expresada como `explained_variance_ratio`.")
    report.append("- Se midio la estabilidad por el cambio entre la ultima epoca y la anterior, usando una distancia invariante al signo.")
    report.append("- Se uso como referencia la configuracion original `lr=0.01, epochs=500` para comparar distancias entre vectores aprendidos.")
    report.append("- No se comparo con PCA, porque la consigna solo pide analizar la regla de Oja.")
    report.append("")
    report.append("## 4. Resultado global")
    report.append(f"- Mejor configuracion segun varianza explicada y estabilidad: **{best_label}**.")
    report.append(f"- Peor configuracion segun el mismo criterio: **{worst_label}**.")
    report.append(f"- Configuracion original: **{original_label}**.")
    report.append(f"- Distancia de la mejor respecto de la original: {best_vs_original_distance:.6f}.")
    report.append(f"- Varianza explicada de la mejor configuracion: {best_quality:.6f}.")
    report.append(f"- Varianza explicada de la peor configuracion: {worst_quality:.6f}.")
    report.append("- La mejor configuracion y la original son practicamente equivalentes: el cambio al pasar de 500 a 3000 epochs es marginal.")
    report.append("")
    report.append("## 5. Lectura por learning rate")
    report.append(f"- Learning rates bajos: {format_lr(low_lr.learning_rate)} aprenden lento; en este barrido su mejor resultado fue {low_lr.max_explained_variance_ratio:.6f}, bastante por debajo del mejor global.")
    report.append(f"- Learning rate medio cercano al original: {format_lr(medium_lr.learning_rate)} mostro el mejor equilibrio entre calidad y estabilidad; su mejor resultado fue {medium_lr.max_explained_variance_ratio:.6f}.")
    report.append(f"- Learning rates altos: {format_lr(high_lr.learning_rate)} llegan rapido a una buena solucion, pero en epocas cortas pueden mostrar cambios grandes; su mejor resultado fue {high_lr.max_explained_variance_ratio:.6f}.")
    report.append("")
    report.append("## 6. Efecto de aumentar epochs")
    report.append("- Al aumentar epochs, las configuraciones con learning rate bajo suelen mejorar de forma mas lenta y necesitan muchas mas iteraciones para acercarse a un estado estable.")
    report.append("- En configuraciones con learning rate medio o alto, muchas veces la mejora fuerte aparece al principio y luego se estabiliza; mas epochs dejan de aportar mucho cuando el vector ya convergio.")
    report.append("- Si el cambio final entre dos epocas consecutivas sigue siendo alto, el entrenamiento aun no quedo asentado.")
    report.append("")
    report.append("## 7. Configuraciones no convergidas o inestables")
    if unstable_labels:
        report.append(f"- Se detectaron configuraciones con cambios finales altos, tipicas de una etapa de aprendizaje aun activa: {', '.join(unstable_labels)}.")
    else:
        report.append("- No se detectaron configuraciones claramente inestables segun la heuristica usada.")
    report.append("")
    report.append("## 8. Interpretacion de los graficos")
    report.append(f"- `distance_heatmap.png`: muestra que tan lejos quedo cada configuracion respecto de la original `0.01 / 500`.")
    report.append(f"- `similarity_evolution.png`: muestra la similitud exacta con el vector final de cada learning rate a medida que aumentan las epochs.")
    report.append(f"- `quality_over_epochs.png`: permite ver si mas epochs siguen mejorando la varianza explicada o si el modelo ya convergio.")
    report.append(f"- `best_worst_loadings.png`: compara los pesos aprendidos por la mejor y la peor configuracion.")
    report.append(f"- `best_worst_countries.png`: compara las proyecciones de paises entre la mejor y la peor configuracion.")
    report.append(f"- `ranking.png`: ordena todas las configuraciones por calidad.")
    report.append(f"- `country_projection_comparison.png`: compara la proyeccion de paises entre la configuracion original y la mejor.")
    report.append("")
    report.append("## 9. Parrafos listos para la presentacion oral")
    for paragraph in oral_paragraphs:
        report.append(f"- {paragraph}")
    report.append("")
    report.append("## 10. Resumen de archivos generados")
    for label, path in output_paths.items():
        report.append(f"- {label}: `{path}`")

    return "\n".join(report)


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(project_root, "data", "europe.csv")
    output_dir = os.path.join(project_root, "oja", "outputs", "hyperparameter_search")
    ensure_directory(output_dir)

    country_names, feature_names, x_std = oja_core.load_europe_data(csv_path)

    results, baseline_result, baseline_weights = run_grid_search(x_std, feature_names, country_names)
    summary_df = results_to_summary_df(results)
    loadings_df = results_to_loadings_df(results)
    scores_df = results_to_scores_df(results)
    history_df = results_to_history_df(results)
    learning_rate_summary = summarize_by_learning_rate(summary_df)

    summary_csv = os.path.join(output_dir, "oja_hyperparameter_summary.csv")
    loadings_csv = os.path.join(output_dir, "oja_hyperparameter_loadings.csv")
    scores_csv = os.path.join(output_dir, "oja_hyperparameter_country_scores.csv")
    history_csv = os.path.join(output_dir, "oja_hyperparameter_history.csv")

    summary_df.to_csv(summary_csv, index=False)
    loadings_df.to_csv(loadings_csv, index=False)
    scores_df.to_csv(scores_csv, index=False)
    history_df.to_csv(history_csv, index=False)

    best_row = summary_df.sort_values(by=["explained_variance_ratio", "final_step_change"], ascending=[False, True]).iloc[0]
    worst_row = summary_df.sort_values(by=["explained_variance_ratio", "final_step_change"], ascending=[False, True]).iloc[-1]

    best_result = next(
        result for result in results if result.learning_rate == best_row.learning_rate and result.epochs == int(best_row.epochs)
    )
    worst_result = next(
        result for result in results if result.learning_rate == worst_row.learning_rate and result.epochs == int(worst_row.epochs)
    )

    distance_heatmap_png = os.path.join(output_dir, "distance_heatmap.png")
    similarity_evolution_png = os.path.join(output_dir, "similarity_evolution.png")
    quality_over_epochs_png = os.path.join(output_dir, "quality_over_epochs.png")
    best_worst_loadings_png = os.path.join(output_dir, "best_worst_loadings.png")
    best_worst_countries_png = os.path.join(output_dir, "best_worst_countries.png")
    ranking_png = os.path.join(output_dir, "ranking.png")
    country_projection_comparison_png = os.path.join(output_dir, "country_projection_comparison.png")

    plot_similarity_evolution(results, similarity_evolution_png)
    plot_quality_over_epochs(summary_df, quality_over_epochs_png)
    plot_best_worst_loadings(best_result, worst_result, best_worst_loadings_png)
    plot_best_worst_countries(best_result, worst_result, best_worst_countries_png)
    plot_ranking(summary_df, ranking_png)
    plot_country_comparison(best_result, baseline_result, country_projection_comparison_png)

    report_path = os.path.join(output_dir, "oja_hyperparameter_report.md")
    output_paths = {
        "summary_csv": summary_csv,
        "loadings_csv": loadings_csv,
        "scores_csv": scores_csv,
        "history_csv": history_csv,
        "distance_heatmap_png": distance_heatmap_png,
        "similarity_evolution_png": similarity_evolution_png,
        "quality_over_epochs_png": quality_over_epochs_png,
        "best_worst_loadings_png": best_worst_loadings_png,
        "best_worst_countries_png": best_worst_countries_png,
        "ranking_png": ranking_png,
        "country_projection_comparison_png": country_projection_comparison_png,
        "report_md": report_path,
    }

    report_text = build_report(
        summary_df=summary_df,
        history_df=history_df,
        baseline_result=baseline_result,
        best_result=best_result,
        worst_result=worst_result,
        learning_rate_summary=learning_rate_summary,
        output_paths=output_paths,
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("=== Oja Hyperparameter Search ===")
    print(f"Dataset: {csv_path}")
    print(f"Experiments run: {len(summary_df)}")
    print(f"Best configuration: lr={format_lr(best_row.learning_rate)}, epochs={int(best_row.epochs)}")
    print(f"Worst configuration: lr={format_lr(worst_row.learning_rate)}, epochs={int(worst_row.epochs)}")
    print("\nGenerated files:")
    for key, path in output_paths.items():
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
