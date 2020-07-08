import numpy as np
import pandas as pd
from scipy.stats import chi2

def mvs(data):
    '''
    Calculate mean, variance, and standard deviation for an array.

    input: the array
    output: dictionary with mean, variation, and standard deviation
    '''

    # size of array and mean
    n = len(data)
    mean = sum(data) / n

    # variance and standard deviation
    sq_differences = [(value - mean)**2 for value in data]
    variance = sum(sq_differences) / (n - 1)
    std = variance ** 0.5

    # package up the results
    results = {'mean': mean, 'variance': variance, 'standard_deviation': std}
    return results



def covariance(data, labels=None):
    '''
    Calculates variances/covariances between arrays

    input: list of arrays
    output: variance-covariance matrix
    '''

    # Get mean and standard deviations for each array and make parallel list
    data_mvs = []
    size = len(data[0])
    for item in data:
    # Verify the size of each array is equal to the first array's size
        assert len(item) == size, f'Array index {data.index(item)} is of different \
                                    length.'   
        data_mvs.append(mvs(item))

    # calculate covariance/variation of each permutation of the two arrays
    # building the matrix
    matrix = []
    for i in range(len(data)):
        row = []
        for j in range(len(data)):
            # Find covariance
            prod = [(data[i][x] - data_mvs[i]['mean']) * (data[j][x] - \
                                    data_mvs[j]['mean']) for x in range(len(data[i]))]
            cov = sum(prod) / (len(data[i]) - 1)
            row.append(cov)
        matrix.append(row)

    return pd.DataFrame(matrix, columns=labels, index=labels)   


def correlation(data, labels=None):
    '''
    Calculates correlations between arrays

    input: list of arrays
    output: variance-covariance matrix
    '''

    # Get mean and standard deviations for each array and make parallel list
    data_mvs = []
    size = len(data[0])
    for item in data:
    # Verify the size of each array is equal to the first array's size
        assert len(item) == size, f'Array index {data.index(item)} is of different \
                                    length.'                                               
        data_mvs.append(mvs(item))

    # calculate correlations of each permutation of the two arrays, building 
    # the matrix
    matrix = []
    for i in range(len(data)):
        row = []
        for j in range(len(data)):
            # covariance
            prod = [(data[i][x] - data_mvs[i]['mean']) * (data[j][x] - \
                                    data_mvs[j]['mean']) for x in range(len(data[i]))]
            cov = sum(prod) / (len(data[i]) - 1)
            # correlation
            r = cov / (data_mvs[i]['standard_deviation'] * data_mvs[j]['standard_deviation'])
            row.append(r)
        matrix.append(row)

    return pd.DataFrame(matrix, columns=labels, index=labels)


def chi_2(data, labels=None):
    '''
    Calculate chi-square given a crosstab of two categorical values.

    Input:
    List of two arrays of same length

    Output:
    crosstab, chi-square statistic, and p-value
    '''

    assert len(data) == 2, f'Other than two input arrays ({len(data)})'
    assert len(data[0]) == len(data[1]), 'Inputs of different lengths!'

    if labels == None:
        labels = ['0', '1']

    colA = pd.Series(data[0], name=labels[0])
    colb = pd.Series(data[1], name=labels[1])
    cross_data = pd.crosstab(colA, colb)

    # get the row and column sums
    row_sums = np.array(cross_data.sum(axis=1))
    col_sums = np.array(cross_data.sum(axis=0))
    total = sum(row_sums)

    # construct expected values table
    expected = []
    for i in row_sums:
        row_expected = []
        for j in col_sums:
            row_expected.append((i * j) / total)
        expected.append(row_expected)

    # Convert cross_data and expected into one dimensional np arrays
    expected = np.array(expected).flatten()
    observed = np.array(cross_data).flatten()

    # Calculate Chi_squared
    chi_square = sum((observed - expected) ** 2 / expected)

    # And p-value (which does use scipy.stats.chi2)
    dof = (cross_data.shape[0] - 1) * (cross_data.shape[1] - 1)
    p_value = chi2.pdf(chi_square, dof)

    return cross_data, chi_square, p_value


# Very quick test to see if all the functions run
if __name__ == '__main__':

    from IPython.display import display

    sales = [3505, 2400, 3027, 2798, 3700, 3250, 2689]
    customers = [3505, 2400, 3027, 2798, 3700, 3250, 2689]

    ctab, chi2, p = chi_2([sales, customers], labels=['sales', 'customers'])
    display(ctab)
    print(f'chi square = {chi2} p_value = {p}')

    print(mvs(sales))
    display(covariance([sales, customers], labels=['sales', 'customers']))
    display(correlation([sales, customers], labels=['sales', 'customers']))
  