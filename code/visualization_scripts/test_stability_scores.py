"""
A script for some statistical testing with pre-calculated ROI stability scores
"""
import pickle
import numpy as np
from scipy.stats import median_test

data_path = '/m/cs/scratch/networks/nurmit7/artikkeli_onerva/multilayer-brains/code/stab_scores_with_ROI_size.pickle'

p_reference = 'craddock'
p_targets = ['ReHo_seeds_weighted_mean_consistency_voxelwise_thresholding_03_regularization-100', 'ReHo_seeds_min_correlation_voxelwise_thresholding_03']

percentile_to_check = 95
n_iterations = 1000

f = open(data_path, 'rb')
data = pickle.load(f)
f.close()

assert p_reference in data.keys(), 'reference key for statistical testing is not included in data keys, please check'
assert np.all(np.array([target in data.keys() for target in p_targets])), 'some of the target keys are not included in data keys, please check'

reference_data = [d[0] for d in data[p_reference]] # data contaains stability score-size pairs
target_data = [[d[0] for d in data[target]] for target in p_targets]

reference_percentile = np.percentile(reference_data, percentile_to_check)
reference_median = np.median(reference_data)
print(f'{percentile_to_check}th percentile for {p_reference}: {reference_percentile}')
print(f'Median for {p_reference}: {reference_median}')

for target, target_key in zip(target_data, p_targets):
    target_percentile = np.percentile(target, percentile_to_check)
    target_median = np.median(target)

    print(f'{percentile_to_check}th percentile for {target_key}: {target_percentile}')
    print(f'Median for {target_key}: {target_median}')

    _, median_p, _, _ = median_test(reference_data, target)

    actual_percentile_diff = target_percentile - reference_percentile

    pooled_data = np.concatenate([reference_data, target])
    n, m = len(reference_data), len(target)

    simulated_percentile_diffs = []

    for i in range(n_iterations):
        np.random.shuffle(pooled_data)
        shuffled_data1 = pooled_data[:n]
        shuffled_data2 = pooled_data[n:]
        simulated_percentile_diff = np.percentile(shuffled_data1, percentile_to_check) - np.percentile(shuffled_data2, percentile_to_check)
        simulated_percentile_diffs.append(simulated_percentile_diff)

    if actual_percentile_diff > 0:
        percentile_p_one_sided = (simulated_percentile_diffs >= actual_percentile_diff).mean()
    else:
        percentile_p_one_sided = (simulated_percentile_diffs <= actual_percentile_diff).mean()

    percentile_p_two_sided = (np.abs(simulated_percentile_diffs) > np.abs(actual_percentile_diff)).mean()

    print(f'p-value {p_reference} vs {target_key}, median: {median_p:e}')
    print(f'one-sided p-value {p_reference} vs {target_key}, {percentile_to_check}th percentile: {percentile_p_one_sided:e}')
    print(f'two-sided p-value {p_reference} vs {target_key}, {percentile_to_check}th percentile: {percentile_p_two_sided:e}')



