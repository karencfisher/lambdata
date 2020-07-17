import unittest
import random
import numpy as np
import statools.tools as st

def datagen(n_count, length=None, random_state=None):
    '''
    Helper function to generate n_count random arrays (lists) of 
    length size

    arguments:
        n_count - how many arrays we want
        length - how long each is. Optional. If None, we generate a 
                 random number of them.
        random_state - a random seed for reproducibility. Optional.

    returns:
        2 dimensional list of lists of random values, of equal lengths
    '''
    if random_state != None:
        random.seed(random_state)

    if length == None:
        length = random.randint(0, 1000)

    output = []
    for _ in range(n_count):
        array = [random.random() * 100 for _ in range(length)]
        output.append(array)

    return output


class BoundaryTests(unittest.TestCase):
    '''
    Test if ValueError is raised if invalid input is passed to the 
    constructor. Expected number of arrays > 1, and that all are of
    equal length.
    '''
    
    def test_boundary_1(self):
        ''' Should raise exception if input contains only one array, however long '''
        # Get random data, one array any (random) length
        data = datagen(1, random_state=42)

        # pass if ValueError raised
        with self.assertRaises(ValueError):
            _ = st.DataSet(data)


    def test_boundary_2(self):
        ''' Should raise exception if input contains arrays of unequal length '''
        # Produce randomly 5 - 100 arrays of random length
        n_count = random.randint(5, 100)
        data = datagen(n_count, random_state=42)

        # Shorten one randomly selected array by one element
        index = data.index(random.choice(data))
        del data[index][-1]

         # pass if ValueError raised
        with self.assertRaises(ValueError):
            _ = st.DataSet(data)


    
class MVSOutputTests(unittest.TestCase):
    '''
    Test the values output by the statistics() method. We'll get the 'raw'
    stats from the DataSet object (DataSet.stats), and randomly compare the
    outputs for one of the arrays with the similar numpy function.
    '''
    def setUp(self):
        # We'll create a random number of arrays 
        n_count = random.randint(5, 100)

        # Create the data randomy
        data = datagen(n_count, random_state=42)

        # Instantiate DataSet object and get statistics (in raw form)
        # First call the staistics() method, and then access the instance
        # attribute DataSet.stats. Needs only be done once.
        self.data_set = st.DataSet(data)
        _ = self.data_set.statistics()
        self.mvs = self.data_set.stats


    def test_mean(self):
        '''Should pass if mean == numpy mean (within 7 places)'''
        # In each case, we select one of the arrays at random
        key = random.choice(list(self.data_set.data_set.keys()))

        # Get expected value
        mean = np.mean(self.data_set.data_set[key])

        # Get and check returned value, rounding to 7 places
        mean_test = self.mvs[key]['mean']
        self.assertAlmostEqual(mean, mean_test, 7, 'Mean calculated wrong')


    def test_variance(self):
        '''Should pass if variance == numpy variance (within 7 places)'''
        key = random.choice(list(self.data_set.data_set.keys()))
        variance = np.var(self.data_set.data_set[key], ddof=1)
        variance_test = self.mvs[key]['variance']
        self.assertAlmostEqual(variance, variance_test, 7, 'Variance calculated wrong')
        

    def test_std(self):
        '''Should pass if STD == numpy STD (within 7 places)'''
        key = random.choice(list(self.data_set.data_set.keys()))
        std = np.std(self.data_set.data_set[key], ddof=1)
        std_test = self.mvs[key]['standard_deviation']
        self.assertAlmostEqual(std, std_test, 7, 'Standard deviation calculated wrong')


    def tearDown(self):
        del self.data_set
        

        
class MatriceOutputTests(unittest.TestCase):
    ''' 
    Test the covariance and correlation matrices generated by the
    DataSet class. Create instance of the class with random data, and
    compare outputs from the covariance and correlation methods with
    equivalent numpy methods. 
    '''
    def setUp(self):
        # We'll create a random number of arrays. This all only
        # need be done once
        n_count = random.randint(5, 100)

        # Create the data randomy
        self.data = datagen(n_count, random_state=42)

        # Instantiate the DataSet object.
        self.data_set = st.DataSet(self.data)


    def test_covariance(self):
        '''Passes if generated covariance matrix == numpy covariance matrix'''
        # Get numpy generated covariance matrix on our raw data
        covariance_np = np.cov(self.data, ddof=1)

        # Get covariance matrix from the object
        covariance_st = self.data_set.covariance()

        # If != within 7 places, handle assertion error from numpy as failure
        try:
            np.testing.assert_array_almost_equal(covariance_np, covariance_st,
                                                 decimal=7)
        except AssertionError:
            self.fail('Covariance method fails')


    def test_correlation(self):
        '''Passes if generated correlation matrix == numpy correlation matrix'''
        # Get numpy generated correlation matrix on our raw data
        correlation_np = np.corrcoef(self.data, ddof=1)

        # Get correlation matrix from the object
        correlation_st = self.data_set.correlation()

        # If != within 7 places, handle assertion error from numpy as failure
        try:
            np.testing.assert_array_almost_equal(correlation_np, correlation_st,
                                                 decimal=7)
        except AssertionError:
            self.fail('Correlation method fails')

    
    def tearDown(self):
        del self.data_set




if __name__ == '__main__':
    unittest.main()
    
