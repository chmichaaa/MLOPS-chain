"""LSTM Autoencoder: an sklearn-compatible anomaly-detection estimator that
scores a trailing WINDOW of rows instead of one row at a time -- unlike the
row-independent scorers this project used previously (IsolationForest/
OneClassSVM, since dropped from training_pipeline.py), which see each row's
engineered features (raw + rolling mean) independently and have no notion of
trajectory. Real incidents in this project's data are gradual, multi-hour
regime shifts (see processing/build_dataset.py's module docstring) rather
than sharp point outliers, which is exactly the shape a sequence-
reconstruction model is built to catch: as the actual trajectory drifts away
from learned-normal dynamics, reconstruction error rises across the window,
not just at one point.

Trained only on X_train (day_block_split guarantees every anomaly day, real
or synthetic, is excluded from it -- see data_handling.day_block_split), so
the reconstruction-error threshold is learned purely from what normal
trajectories look like -- the same unsupervised setup as before.

The sole model type training_pipeline.train_and_select searches over via
Hyperopt (config.MAX_EVALS_LSTM) + MLflow logging + best-hyperparameters-by-F1
selection.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, OutlierMixin
from torch.utils.data import DataLoader, TensorDataset


class _LSTMAE(nn.Module):
    """Sequence-to-sequence LSTM autoencoder (RepeatVector-style decoder):
    an encoder LSTM compresses the whole input window into one latent vector,
    which is then repeated across the window length and fed to a decoder LSTM
    that reconstructs every timestep.
    """

    def __init__(self, n_features, hidden_size, latent_size):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.to_latent = nn.Linear(hidden_size, latent_size)
        self.from_latent = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, n_features)

    def forward(self, x):
        seq_len = x.size(1)
        _, (h_n, _) = self.encoder(x)
        latent = self.to_latent(h_n[-1])
        decoder_input = self.from_latent(latent).unsqueeze(1).repeat(1, seq_len, 1)
        decoded, _ = self.decoder(decoder_input)
        return self.output_layer(decoded)


class LSTMAutoencoder(BaseEstimator, OutlierMixin):
    """window_size trailing rows (left-edge-padded so no row is dropped, same
    convention as preprocessing.RollingWindowFeatures) are reconstructed as a
    sequence; the anomaly score for the row at the end of a window is that
    window's mean squared reconstruction error. Follows sklearn's outlier-
    detector convention: predict() returns 1 (inlier) / -1 (outlier),
    decision_function() is positive for inliers, negative for outliers -- the
    contract training_pipeline.evaluate_pipeline and predict.py expect from
    any pipeline's final estimator, uploaded or trained here.
    """

    def __init__(
        self,
        seq_len=12,
        hidden_size=32,
        latent_size=16,
        epochs=20,
        batch_size=64,
        lr=1e-3,
        threshold_percentile=95,
        random_state=42,
    ):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.threshold_percentile = threshold_percentile
        self.random_state = random_state

    def _make_sequences(self, X):
        X = np.asarray(X, dtype=np.float32)
        pad = np.repeat(X[:1], self.seq_len - 1, axis=0)
        padded = np.concatenate([pad, X], axis=0)
        n = len(X)
        sequences = np.stack([padded[i:i + self.seq_len] for i in range(n)], axis=0)
        return sequences.astype(np.float32)

    def _reconstruction_errors(self, X):
        sequences = torch.from_numpy(self._make_sequences(X))
        self.model_.eval()
        with torch.no_grad():
            recon = self.model_(sequences)
            errors = ((recon - sequences) ** 2).mean(dim=(1, 2))
        return errors.numpy()

    def fit(self, X, y=None):
        torch.manual_seed(self.random_state)
        sequences = self._make_sequences(X)
        n_features = sequences.shape[-1]

        self.model_ = _LSTMAE(n_features, self.hidden_size, self.latent_size)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        dataset = TensorDataset(torch.from_numpy(sequences))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        for _ in range(self.epochs):
            for (batch,) in loader:
                optimizer.zero_grad()
                recon = self.model_(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()

        train_errors = self._reconstruction_errors(X)
        self.threshold_ = float(np.percentile(train_errors, self.threshold_percentile))
        return self

    def decision_function(self, X):
        return self.threshold_ - self._reconstruction_errors(X)

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)
