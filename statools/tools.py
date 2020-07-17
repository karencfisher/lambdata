import pandas as pd


class DataSet():
    '''
    Maintains a data set given simple arrays or lists of values, and
    provides basic statistics, including covariance and correlation
    matrices.
    '''
    def __init__(self, data, labels=None):
        '''
        Class constructor

        Args:
            data - 2 dimensional Python array (list). List of lists of
            values.

            labels - list of column headings. Optional.

        Returns:
            dictionary of the data set in the form of
            {column-name: [values]}

        Exceptions:
            ValueError if input is invalid
        '''
        # Validate user input. Should be list of at least 2 lists.
        try:
            if len(data) == 1:
                raise ValueError('Input should be two dimensional list')
        except TypeError:
            raise ValueError('Input should be two dimensional list')

        first_length = len(data[0])
        for array in data[1:]:
            if len(array) != first_length:
                raise ValueError('Input arrays should be of same length')
            
        # If labels None, create labels of list form Columni
        if labels == None:
            labels = ['Column' + str(n) for n in range(len(data))]

        # Populate dictionary of data lists by column name
        self.data_set = {}
        for i in range(len(data)):
            self.data_set[labels[i]] = data[i]

    
    ####### Private helper functions
    def __mvs(self, data):
        '''
        Calculate mean, variance, and standard deviation for an array.

        Args:
            data - an 'array' (list) of values

        Returns:
            dictionary with mean, variation, and standard deviation
        '''
        # size of array and mean
        n = len(data)
        mean = sum(data) / n

        # variance and standard deviation
        sq_differences = [(value - mean)**2 for value in data]
        variance = sum(sq_differences) / (n - 1)
        std = variance ** 0.5

        # package up the results
        statistics = {'mean': mean, 'variance': variance, 
                                'standard_deviation': std}
        return statistics


    def __covar(self, colA, colB):
        '''
        Calculate covariance for two arrays.

        Args:
            ColA - an 'array' (list) of values
            ColB - an 'array' (list) of values

        Returns:
            Covariance 
        '''
        assert len(colA) == len(colB), \
            f'Arrays are of unequal length ({len(colA), len(colB)}'

        meanA = self.__mvs(colA)['mean']
        meanB = self.__mvs(colB)['mean']

        prod = [(colA[x] - meanA) * (colB[x] - meanB) 
                                for x in range(len(colA))]
        return sum(prod) / (len(colA) - 1)
        

    def __r(self, colA, colB):
        '''
        Calculate correlation coefficient for two arrays.

        Args:
            ColA - an 'array' (list) of values
            ColB - an 'array' (list) of values

        Returns:
            Correlation coefficient
        '''
        assert len(colA) == len(colB), \
            f'Arrays are of unequal length ({len(colA), len(colB)}'

        covar = self.__covar(colA, colB)

        stdA = self.__mvs(colA)['standard_deviation']
        stdB = self.__mvs(colB)['standard_deviation']
        return covar / (stdA * stdB)
        

    ####### Public functions
    def get_dataframe(self):
        '''
        Returns the data set as a Pandas DataFrame
        '''
        return pd.DataFrame(self.data_set)


    def statistics(self):
        '''
        Gets basic staistics for the data set. Returns as a Pandas
        DataFrame
        '''
        self.stats = {}
        for column in self.data_set.keys():
            self.stats[column] = self.__mvs(self.data_set[column])

        return pd.DataFrame(self.stats).T


    def covariance(self):
        '''
        Calculates variances/covariances between arrays. Returns variance/covariance
        matrix as Pandas Dataframe
        '''
        # calculate covariance/variation of each permutation of the two arrays
        # building the matrix
        self.cov_matrix = []
        for i in self.data_set.values():
            row = []
            for j in self.data_set.values():
                # Find covariance
                row.append(self.__covar(i, j))
            self.cov_matrix.append(row)

        labels = self.data_set.keys()
        return pd.DataFrame(self.cov_matrix, columns=labels, index=labels)


    def correlation(self):
        '''
        Calculates correlation between arrays. Returns correlation
        matrix as Pandas Dataframe
        '''
        # calculate correlations of each permutation of the two arrays, building
        # the matrix
        self.cor_matrix = []
        for i in self.data_set.values():
            row = []
            for j in self.data_set.values():
                # Find r (pearson's correlation)
                row.append(self.__r(i, j))
            self.cor_matrix.append(row)

        labels = self.data_set.keys()
        return pd.DataFrame(self.cor_matrix, columns=labels, index=labels)



if __name__ == '__main__':
    # Very quick unit tests to see if all the functions run and compare 
    # outputs with the equivalent numpy functions

    from IPython.display import display
    import numpy as np

    # Some toy data
    sales = [3505, 2400, 3027, 2798, 3700, 3250, 2689]
    customers = [127, 80, 105, 92, 120, 115, 93]

    # Instantiate
    data = DataSet([sales, customers], ['sales', 'customers'])
    #data = DataSet([sales, customers])

    # Test case 1 -- Try getting dataframe
    print('Get dataframe')
    display(data.get_dataframe())

    # Test case 2 -- And basic statistics
    print('\nGet descriptive statistics')
    display(data.statistics())

    # Test results against numpy equivalents. Print any discrepancies.
    passes = True
    for col in data.stats.keys():
        if data.stats[col]['mean'] != np.mean(data.data_set[col]):
            passes = False
            print(f'\'{col}\': means different\n')
            print(f'Output: {data.stats[col]["mean"]}')
            print(f'Actual: {np.mean(data.data_set[col])}')

        if  data.stats[col]['variance'] != np.var(data.data_set[col], ddof=1):
            passes = False
            print(f'\'{col}\': variances different\n')
            print(f'Output: {data.stats[col]["variance"]}')
            print(f'Actual: {np.var(data.data_set[col], ddof=1)}')

        if  data.stats[col]['standard_deviation'] != np.std(data.data_set[col], ddof=1):
            passes = False
            print(f'\'{col}\': standard deviations different\n')
            print(f'Output: {data.stats[col]["standard_deviation"]}')
            print(f'Actual: {np.std(data.data_set[col], ddof=1)}')
    if not passes:
        print('Some statistics cases fail')
    else:
        print('All statistics cases pass')

    # Test case 3 -- Test covariance method. Compare with np.cov()
    print('\nGet covariance matrix')
    display(data.covariance())

    official_value = np.cov(sales, customers)
    if np.any(official_value != data.cov_matrix):
        print(f'Covariance fails. Actual matrix')
        display(pd.DataFrame(official_value))
    else:
        print(f'Covariance passes')

    # Test case 4 -- Test correlation method. Compare with np.corrcoef()
    print('\nGet correlation matrix')
    display(data.correlation())

    official_value = np.corrcoef(sales, customers)
    if np.any(official_value != data.cor_matrix):
        print(f'Correlation fails. Actual matrix')
        display(pd.DataFrame(official_value))
    else:
        print(f'Correlation passes')
