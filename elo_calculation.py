from typing import List

import numpy as np
from scipy.optimize import minimize

# Initialize Elo ratings
elo_ratings = np.array([1000] * 7)


# Define expected win probability
def expected_win_probability(elo_i, elo_j):
    return 1 / (1 + 10 ** ((elo_j - elo_i) / 500))


# Define negative log-likelihood function
def negative_log_likelihood(elo_ratings, win_prob_matrix):
    n = len(elo_ratings)
    neg_log_likelihood = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                expected_prob = expected_win_probability(elo_ratings[i], elo_ratings[j])
                actual_prob = win_prob_matrix[i][j]
                neg_log_likelihood -= actual_prob * np.log(expected_prob)
    return neg_log_likelihood


def get_elo_ratings(win_prob_matrix: List[List[float]]):
    win_prob_matrix = np.array(win_prob_matrix)
    elo_ratings = np.array([1000] * len(win_prob_matrix))
    result = minimize(
        negative_log_likelihood, elo_ratings, args=(win_prob_matrix,), method="BFGS"
    )
    return result.x


# Empirical win probability matrix (example values)
win_prob_matrix = np.array(
    [
        [0, 0.37, 0.50, 0.56, 0.49, 0.58, 0.51],  # claude35_sonnet
        [0.63, 0, 0.49, 0.59, 0.69, 0.68, 0.46],  # claude3_sonnet
        [0.50, 0.51, 0, 0.65, 0.60, 0.67, 0.62],  # gpt35_turbo
        [0.44, 0.41, 0.35, 0, 0.50, 0.60, 0.35],  # gpt4o
        [0.51, 0.31, 0.40, 0.50, 0, 0.62, 0.42],  # llama2_13b
        [0.42, 0.32, 0.33, 0.40, 0.38, 0, 0.32],  # llama2_7b
        [0.49, 0.54, 0.38, 0.65, 0.58, 0.68, 0],  # llama3_8b
    ]
)


# Bootstrap for confidence intervals
def bootstrap_elo(win_prob_matrix, n_bootstrap=500):
    n = win_prob_matrix.shape[0]
    bootstrap_samples = []
    for _ in range(n_bootstrap):
        sample_matrix = np.zeros_like(win_prob_matrix)
        for i in range(n):
            for j in range(n):
                if i != j:
                    sample_matrix[i][j] = np.random.binomial(1, win_prob_matrix[i][j])
        result = minimize(
            negative_log_likelihood, elo_ratings, args=(sample_matrix,), method="BFGS"
        )
        bootstrap_samples.append(result.x)
    return np.array(bootstrap_samples)


bootstrap_samples = bootstrap_elo(win_prob_matrix)
confidence_intervals = np.percentile(bootstrap_samples, [2.5, 97.5], axis=0)
print("Confidence Intervals:", confidence_intervals)
