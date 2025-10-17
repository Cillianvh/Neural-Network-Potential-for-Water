import numpy as np
from tensorflow import keras

# -----------------------
# Reproducibility
# -----------------------
np.random.seed(42)
keras.utils.set_random_seed(42)


# -----------------------
# Build Model
# -----------------------
def build_model(input_dim=3):
    """
    Create a simple feed-forward neural network.
    """
    model = keras.Sequential([
        keras.Input(shape=(input_dim,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# -----------------------
# Train Model
# -----------------------
def train_model(X, y, epochs=200, batch_size=32, verbose=1):
    """
    Train a neural network model on input features X and target y.
    Normalizes inputs and targets internally.
    """
    # Shuffle and split 80/20
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    split = int(0.8 * len(idx))
    tr_idx, te_idx = idx[:split], idx[split:]
    Xtr, Xte = X[tr_idx], X[te_idx]
    ytr, yte = y[tr_idx], y[te_idx]

    # Normalize inputs
    mean, std = Xtr.mean(axis=0), Xtr.std(axis=0)
    Xtrn, Xten = (Xtr - mean) / std, (Xte - mean) / std

    # Normalize targets
    y_mean, y_std = ytr.mean(), ytr.std()
    ytrn, yten = (ytr - y_mean) / y_std, (yte - y_mean) / y_std

    # Build & train
    model = build_model(input_dim=X.shape[1])
    history = model.fit(Xtrn, ytrn,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(Xten, yten),
                        verbose=verbose)

    # Evaluate normalized test set
    _, mae_norm = model.evaluate(Xten, yten, verbose=0)
    mae = mae_norm * y_std
    print(f"→ Internal Test MAE: {mae:.6f} Eh")

    return model, mean, std, y_mean, y_std, history, (Xte, yte)
