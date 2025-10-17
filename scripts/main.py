import os
import pandas as pd
import numpy as np
from data_utils import load_data, extract_vectors
from model import train_model
from plots import (
    plot_learning_curves,
    plot_parity_grid,
    plot_error_histograms,
    plot_residuals_grid,
    plot_1d_scans,
    plot_2d_pes
)

if __name__ == "__main__":
    # Ensure plots folder exists
    PLOTS_DIR = "../plots"
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Training/testing combinations
    combos = [
        ("../data/H2O_unrotated.xyz", "../data/H2O_unrotated.ener",
         "../data/H2O_unrotated.xyz", "../data/H2O_unrotated.ener"),
        ("../data/H2O_rotated.xyz", "../data/H2O_rotated.ener",
         "../data/H2O_rotated.xyz", "../data/H2O_rotated.ener"),
        ("../data/H2O_unrotated.xyz", "../data/H2O_unrotated.ener",
         "../data/H2O_rotated.xyz", "../data/H2O_rotated.ener"),
        ("../data/H2O_rotated.xyz", "../data/H2O_rotated.ener",
         "../data/H2O_unrotated.xyz", "../data/H2O_unrotated.ener"),
    ]

    results, resid_list, labels, parity_cases = [], [], [], []

    for tx, te, xx, ee in combos:
        # Load data
        geom_tr, E_tr = load_data(tx, te)
        geom_te, E_te = load_data(xx, ee)
        X_tr = extract_vectors(geom_tr)
        X_te = extract_vectors(geom_te)

        # Train model
        model, mu, sigma, y_mean, y_std, hist, _ = train_model(X_tr, E_tr)

        # Learning curve
        plot_learning_curves(hist, filename=os.path.join(PLOTS_DIR, f"{os.path.basename(tx)}_learning.png"))

        # Predictions
        E_pred = (model.predict((X_te - mu)/sigma).flatten() * y_std) + y_mean
        resid  = E_pred - E_te

        # Store results
        tag = f"{os.path.splitext(os.path.basename(tx))[0]}→{os.path.splitext(os.path.basename(xx))[0]}"
        resid_list.append(resid)
        labels.append(tag)
        parity_cases.append((E_te, E_pred, tag))

        results.append({
            "train": os.path.basename(tx),
            "test": os.path.basename(xx),
            "MAE": np.mean(np.abs(resid)),
            "MSE": np.mean(resid**2),
        })

    # Write summary CSV
    df = pd.DataFrame(results)
    print(df[["train", "test", "MAE", "MSE"]])
    df.to_csv("../experiment_summary.csv", index=False)

    # Combined diagnostic plots
    plot_error_histograms(resid_list, labels, filename=os.path.join(PLOTS_DIR, "error_histograms.png"))
    plot_parity_grid(parity_cases, filename=os.path.join(PLOTS_DIR, "parity_grid.png"))
    plot_residuals_grid(parity_cases, X_te, resid_list, ['r1', 'r2', 'θ'], filename=os.path.join(PLOTS_DIR, "residuals_grid.png"))

    # Final model trained on full unrotated dataset
    geom_fin, E_fin = load_data("../data/H2O_unrotated.xyz", "../data/H2O_unrotated.ener")
    X_fin = extract_vectors(geom_fin)
    model_fin, mu_fin, sigma_fin, ymean_fin, ystd_fin, hist_fin, _ = train_model(X_fin, E_fin)

    # Learning curve for final model
    plot_learning_curves(hist_fin, filename=os.path.join(PLOTS_DIR, "fig1_learning_curves.png"))

    # 1D scans & 2D PES
    plot_1d_scans(model_fin, mu_fin, sigma_fin, filename=os.path.join(PLOTS_DIR, "fig5_1d_scans.png"))
    plot_2d_pes(model_fin, mu_fin, sigma_fin, filename=os.path.join(PLOTS_DIR, "fig6_2d_pes.png"))
