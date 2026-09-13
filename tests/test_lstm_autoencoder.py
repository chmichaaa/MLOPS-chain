import numpy as np

from prediction_model.processing.lstm_autoencoder import LSTMAutoencoder


def _toy_data(n=60, n_features=3, seed=0):
    rng = np.random.RandomState(seed)
    return rng.normal(loc=0.5, scale=0.05, size=(n, n_features)).astype(np.float32)


def test_fit_predict_shapes():
    X = _toy_data()
    model = LSTMAutoencoder(seq_len=4, hidden_size=4, latent_size=2, epochs=1, batch_size=8, random_state=0)
    model.fit(X)
    preds = model.predict(X)
    scores = model.decision_function(X)
    assert preds.shape == (len(X),)
    assert scores.shape == (len(X),)
    assert set(np.unique(preds)).issubset({-1, 1})


def test_predict_matches_decision_function_sign():
    X = _toy_data()
    model = LSTMAutoencoder(seq_len=4, hidden_size=4, latent_size=2, epochs=1, batch_size=8, random_state=0)
    model.fit(X)
    preds = model.predict(X)
    scores = model.decision_function(X)
    assert np.all((scores >= 0) == (preds == 1))


def test_no_rows_dropped_when_seq_len_exceeds_data_length():
    X = _toy_data(n=5, n_features=2)
    model = LSTMAutoencoder(seq_len=12, hidden_size=4, latent_size=2, epochs=1, batch_size=4, random_state=0)
    model.fit(X)
    assert len(model.predict(X)) == 5
