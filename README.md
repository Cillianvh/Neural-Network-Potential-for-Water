# Neural Network Potential for Water (H₂O)

This project implements a feed-forward neural network using **TensorFlow/Keras** to model the **potential energy surface (PES)** of a water molecule.  
The model predicts molecular energies from geometric features (O–H bond lengths and the H–O–H angle) and produces diagnostic plots to visualize performance.

---

## 🧩 Project Structure

- **`data/`** — Input XYZ geometries and energy files.  
- **`plots/`** — Folder for generated plots (learning curves, parity grids, residuals, 1D scans, 2D PES).  
- **`scripts/`** — Python source code:  
  - `main.py` — Runs training, evaluation, and plot generation.  
  - `data_utils.py` — Handles data loading and feature extraction.  
  - `model.py` — Defines and trains the neural network.  
  - `plots.py` — Generates all visualizations.  
- **`experiment_summary.csv`** — Automatically generated summary table of model metrics (ignored by Git).  
- **`report_water_potential.pdf`** — Project report including methods, results, and analysis.  
- **`requirements.txt`** — Python dependencies.  
- **`.gitignore`** — Specifies ignored files (plots, IDE configs, and generated outputs).

---

## ⚙️ Usage

1. **Set up a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate         # Windows
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the main script:**

   ```bash
   python scripts/main.py
   ```

   This trains the neural network, evaluates performance, generates all plots in the `plots/` directory, and outputs `experiment_summary.csv` in the project root.

---

## 👤 Author

**Cillian Vickers-Hayes**  
*October 2025*
