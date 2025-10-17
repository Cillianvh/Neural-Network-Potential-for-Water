import matplotlib.pyplot as plt
import numpy as np
import os


# -----------------------
# Learning curves
# -----------------------
def plot_learning_curves(hist, filename="../plots/learning_curve.png"):
    plt.figure(figsize=(6, 4))
    plt.plot(hist.history['loss'], label='Train')
    plt.plot(hist.history['val_loss'], label='Validation')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300)
    plt.close()


# -----------------------
# Parity grid
# -----------------------
def plot_parity_grid(cases, filename="../plots/parity_grid.png"):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (E_true, E_pred, title) in zip(axs.flat, cases):
        mn, mx = E_true.min(), E_true.max()
        ax.scatter(E_true, E_pred, s=5)
        ax.plot([mn, mx], [mn, mx], 'k--')
        ax.set_title(title)
        ax.set_xlabel('True (Eh)')
        ax.set_ylabel('Pred (Eh)')
    plt.tight_layout()

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300)
    plt.close()


# -----------------------
# Error histograms
# -----------------------
def plot_error_histograms(resid_list, labels, filename="../plots/error_histograms.png"):
    plt.figure(figsize=(6, 4))
    bins = np.linspace(min(r.min() for r in resid_list),
                       max(r.max() for r in resid_list), 50)
    for r, label in zip(resid_list, labels):
        plt.hist(r, bins=bins, alpha=0.5, label=label)
    plt.xlabel('Error (Eh)')
    plt.ylabel('Count')
    plt.legend()
    plt.title('Error Distribution Comparison')
    plt.tight_layout()

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300)
    plt.close()


# -----------------------
# Residuals grid
# -----------------------
def plot_residuals_grid(cases, feature_array, resid_array, names, filename="../plots/residuals_grid.png"):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    # first three panels
    for ax, feat_idx, name, resid in zip(axs.flat, [0, 1, 2], names, resid_array):
        ax.scatter(feature_array[:, feat_idx], resid, s=5)
        ax.axhline(0, color='k', ls='--')
        ax.set_xlabel(name)
        ax.set_ylabel('Error (Eh)')
        ax.set_title(f'Residuals vs {name}')
    # last panel
    E_true, E_pred, _ = cases[-1]
    resid = resid_array[-1]
    ax = axs.flat[3]
    ax.scatter(E_true, resid, s=5)
    ax.axhline(0, color='k', ls='--')
    ax.set_xlabel('True Energy (Eh)')
    ax.set_ylabel('Error (Eh)')
    ax.set_title('Residuals vs True Energy')
    plt.tight_layout()

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300)
    plt.close()


# -----------------------
# 1D scans
# -----------------------
def plot_1d_scans(model, mean, std, filename="../plots/fig5_1d_scans.png"):
    r = np.linspace(0.85, 1.10, 100)
    theta_eq = 107.5
    fig, axs = plt.subplots(3, 1, figsize=(6, 12), sharex=False)

    Xsym = np.stack([r, r, theta_eq * np.ones_like(r)], 1)
    Esym = model.predict((Xsym - mean) / std).flatten()
    axs[0].plot(r, Esym)
    axs[0].set_title('Symmetric Stretch')
    axs[0].set_xlabel('r (Å)');
    axs[0].set_ylabel('Energy (Eh)')

    Xasym = np.stack([r, 0.96 * np.ones_like(r), theta_eq * np.ones_like(r)], 1)
    Easym = model.predict((Xasym - mean) / std).flatten()
    axs[1].plot(r, Easym)
    axs[1].set_title('Asymmetric Stretch')
    axs[1].set_xlabel('r1 (Å)');
    axs[1].set_ylabel('Energy (Eh)')

    a = np.linspace(95, 120, 100)
    Xbend = np.stack([0.96 * np.ones_like(a), 0.96 * np.ones_like(a), a], 1)
    Ebend = model.predict((Xbend - mean) / std).flatten()
    axs[2].plot(a, Ebend)
    axs[2].set_title('Angle Bend')
    axs[2].set_xlabel('θ (°)');
    axs[2].set_ylabel('Energy (Eh)')

    plt.tight_layout()

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300)
    plt.close()


# -----------------------
# 2D PES
# -----------------------
def plot_2d_pes(model, mu, sigma, filename="../plots/fig6_2d_pes.png"):
    r1_vals = np.linspace(0.85, 1.10, 60)
    theta_vals = np.linspace(95, 120, 60)
    r1g, theta_g = np.meshgrid(r1_vals, theta_vals)
    Xgrid = np.stack([r1g.ravel(), 0.96 * np.ones(r1g.size), theta_g.ravel()], 1)
    Egrid = model.predict((Xgrid - mu) / sigma).flatten()
    Esurf = Egrid.reshape(r1g.shape)
    plt.figure(figsize=(6, 5))
    plt.contourf(r1g, theta_g, Esurf, 50, cmap='viridis')
    plt.xlabel('r1 (Å)');
    plt.ylabel('θ (°)')
    plt.title('2D PES')
    plt.colorbar(label='Energy (Eh)')
    plt.tight_layout()

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300)
    plt.close()
