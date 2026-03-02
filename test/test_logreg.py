"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import numpy as np
import pandas as pd

from regression import LogisticRegressor
from regression import loadDataset

# Load dataset for testing
X_train, X_val, y_train, y_val = loadDataset(
    features=[
        'Penicillin V Potassium 500 MG',
        'Computed tomography of chest and abdomen',
        'Plain chest X-ray (procedure)',
        'Low Density Lipoprotein Cholesterol',
        'Creatinine',
        'AGE_DIAGNOSIS',
        'Triglycerides',
        'Glucose'],
        split_percent=0.8,
        split_seed=42)


# Initialize model
num_features = X_train.shape[1]
model = LogisticRegressor(num_feats=num_features, learning_rate=1e-5, tol=1e-5, max_iter=100, batch_size=10)

def test_prediction():
    model.W = np.zeros(num_features + 1) # zero weights
    y_pred = model.make_prediction(np.hstack([X_train, np.ones((X_train.shape[0], 1))])) # add bias term to input
    
    assert len(y_pred) == X_train.shape[0] # pred should match number of samples
    assert y_pred.shape == (X_train.shape[0],) # pred should be 1D array with length n_samples
    assert not np.isnan(y_pred).any()
    assert np.all((y_pred >= 0.0) & (y_pred <= 1.0)) # all pred should be b/w [0,1]
    assert np.allclose(y_pred, 0.5, atol=1e-5) # with zero weights, should output 0.5

def test_loss_function():
    y_true = np.array([1, 0])
    y_pred = np.array([0.95, 0.05])

    loss_perfect = model.loss_function(y_true, y_true)
    assert np.isclose(loss_perfect, 0, atol=1e-5) # loss should be near zero for perfect pred

    # manually compute and check
    epsilon = 1e-6
    loss_m = -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))
    model_loss = model.loss_function(y_true, y_pred)

    assert np.isclose(loss_m, model_loss, atol=1e-5)

def test_gradient():
    model.W = np.zeros(num_features + 1) # zero weights

    X = np.hstack([X_train, np.ones((X_train.shape[0], 1))])

    # manual predictions & error
    y_pred_manual = model.sigmoid(np.dot(X, model.W))
    error = y_pred_manual - y_train

    # manual gradient with model
    manual_gradient = np.dot(X.T, error) / len(y_train)
    gradient = model.calculate_gradient(y_train, X)

    assert gradient.shape == model.W.shape # check shape match
    assert np.all(np.isfinite(gradient)) # no inf or NaN values
    assert np.any(gradient != 0) # gradient should not be all zeros for non-perfect pred
    assert np.allclose(gradient, manual_gradient, atol=1e-5) # check values match manual calculation

def test_training():
    model.W = np.zeros(num_features + 1) # zero weights
    W_start = model.W.copy()

    # initial loss before training
    initial_loss = model.loss_function(y_val, model.make_prediction(np.hstack([X_val, np.ones((X_val.shape[0], 1))])))

    # train
    model.train_model(X_train, y_train, X_val, y_val)
    W_end = model.W.copy()
    final_loss = model.loss_function(y_val, model.make_prediction(np.hstack([X_val, np.ones((X_val.shape[0], 1))])))

    assert not np.allclose(W_start, W_end) # weights should be different after training
    assert final_loss < initial_loss # loss should decrease after training