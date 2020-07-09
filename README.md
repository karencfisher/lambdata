# lambdata

## statools.tool

#### mvs(data)

Calculates the Mean, Variance, and Standard Deviation for a single list of values. 

**Arguments**

*data* - Python list of values

**Output**

Returns a Python dictionary containing the mean, variance, and standard deviation of the
values. 

![](/assets/mvs_example.jpg)

#### covariance(data, labels=*None*)

Calculates a variance/covariance matrix for a set of arrays of values. 

**Arguments**

*data* - List of lists of values. Can be two or more, and lists must be of equal length.

*labels* - List of labels for the arrays. Optional (default is *None*).

**Output**

Returns a Pandas dataframe of the variance/covariance matrix.

![](/assets/covariance_example.jpg)

#### correlation(data, labels=*None*)

Calculates a correlation matrix (Pearson correlation) for a set of arrays of values. **Arguments**

*data* - List of lists of values. Can be two or more, and lists must be of equal length.

*labels* - List of labels for the arrays. Optional (default is *None*).

**Output**

Returns a Pandas dataframe of the correlation matrix.

![](/assets/correlation_example.jpg)

#### chi_2(data, labels=*None*)

Constructs a crosstab table for two equal length data lists, and obtains chi square score
an p-value. 

**Arguments**

*data* - List of lists of values. Must be two lists of values, and they must be of equal length.

*labels* - List of labels for the arrays for the crosstab output. Optional (default is *None*).

**Output**

Returns a list containing a Pandas dataframe of the crosstab table, chi-square stastic, and the corresponding p-value.

![](/assets/chi2_example.jpg)