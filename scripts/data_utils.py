import numpy as np


# -----------------------
# Geometry functions
# -----------------------
def calculate_distance(a, b):
    """
    Calculate Euclidean distance between two 3D points.
    """
    return np.linalg.norm(a - b)


def calculate_angle(a, b, c):
    """
    Calculate angle (in degrees) formed at point b by vectors ba and bc.
    """
    v1, v2 = a - b, c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos = np.clip(cos, -1, 1)
    return np.degrees(np.arccos(cos))


# -----------------------
# Load data
# -----------------------
def load_data(xyz_file, ener_file):
    """
    Load molecular geometries and energies from XYZ and energy files.

    """
    energies = np.loadtxt(ener_file)
    try:
        with open(xyz_file) as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find XYZ file: {xyz_file}")

    geoms, idx = [], 0
    while idx < len(lines):
        idx += 2  # skip atom count + comment
        mol = []
        for _ in range(3):  # H2O only
            parts = lines[idx].split()
            mol.append(np.array(list(map(float, parts[1:4]))))
            idx += 1
        geoms.append(mol)
    return geoms, energies


# -----------------------
# Feature extraction
# -----------------------
def extract_vectors(geometries):
    """
    Convert geometries into input features for the model: [r1, r2, angle].

    """
    feats = []
    for mol in geometries:
        h1, o, h2 = mol
        r1 = calculate_distance(o, h1)
        r2 = calculate_distance(o, h2)
        θ = calculate_angle(h1, o, h2)
        feats.append([r1, r2, θ])
    return np.array(feats)
