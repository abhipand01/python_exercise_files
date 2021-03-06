import numpy as np
import pandas as pd
from pandas import Series, DataFrame
cars_address = 'C:/Users/abhis/Desktop/python_exercise_files/Data/mtcars.csv'
iris_data_address = 'C:/Users/abhis/Desktop/python_exercise_files/Data/iris.data.csv'

############################# Chapter 2: Data Preparation Start ########################################################

####################################### Filtering and Selecting ########################################################

series_obj = Series(np.arange(8), index=['row1', 'row2', 'row3', 'row4', 'row5', 'row6', 'row7', 'row8'])
print(series_obj)

# Selecting the element with the label index as 'row6'

print(series_obj['row6'])

df_obj = DataFrame(np.arange(36).reshape(6, 6),
                   index=['r1', 'r2', 'r3', 'r4', 'r5', 'r6'],
                   columns=['c1', 'c2', 'c3', 'c4', 'c5', 'c6'])
df_obj1 = DataFrame(np.arange(36).reshape(6, 6),
                    index=['1', '2', '3', '4', '5', '6'],
                    columns=['1', '2', '3', '4', '5', '6'])
print(df_obj)

a = df_obj.c1
b = df_obj1.loc[['1'], ['3']]

print(a)
print(df_obj1)
print(b)

############################## Data Slicing - selecting and filtering a range of value #################################

# Selecting between row 3 and row 7
d = series_obj['row3': 'row7']
print(d)


# Comparing with scalars
boolean_lt_point_2 = df_obj < 0.2
print(boolean_lt_point_2)

# Filtering with scalar
e = series_obj[series_obj > 4]
print(e)
f = df_obj[df_obj > 6]
print(f)

# Setting value with scalar
series_obj['row1', 'row5', 'row8'] = 8
print(series_obj)

#######################################################################################################################

########################Treating missing value ########################################################################

df_obj_missing = df_obj[df_obj > 16]
print(df_obj_missing)

np.random.seed(25)
df_obj = DataFrame(np.random.rand(36).reshape(6, 6),
                   index=['r1', 'r2', 'r3', 'r4', 'r5', 'r6'],
                   columns=['c1', 'c2', 'c3', 'c4', 'c5', 'c6'])
print(df_obj)

df_obj.loc['r4':'r6', 'c1'] = np.nan
df_obj.loc['r2':'r5', 'c6'] = np.nan
print(df_obj)

df_filled = df_obj.fillna({'c1': 0.1, 'c6': 1.25})
print(df_filled)

# forward fill
df_ffill = df_obj.ffill()
print(df_ffill)
df_bfill = df_obj.fillna(method="bfill")
print(df_bfill)

# count of all the null value in the dataframe
df_obj.isnull().sum()

## Filtering out missing vaues
df_null_nan = df_obj.dropna()
print(df_null_nan)

df_null_nan_col = df_obj.dropna(axis=1)
print(df_null_nan_col)

############################ Drop duplicates ###########################################################################

df_obj = DataFrame({'c1': [1,1,2,2,3,3,3],
                    'c2': ['a','a','b','b','c','c','c'],
                    'c3': ['A','A','B','B','C','C','C'],
                    'c3': ['A','A','B','B','C','C','C']})
print(df_obj)

df_obj.duplicated()

df_obj.drop_duplicates()

df_obj2 = DataFrame({'c1': [1,1,2,2,3,3,3],
                    'c2': ['a','a','b','b','c','c','c'],
                    'c3': ['A','A','B','B','C','C','C'],
                    'c3': ['A','A','B','B','C','D','C']})

df_obj2.drop_duplicates('c3')
df_obj2.drop_duplicates()

########################################################################################################################

########################## Concatenation and Transformation ############################################################

df_obj = DataFrame(np.arange(36).reshape(6, 6))
print(df_obj)

df_obj_app = DataFrame(np.arange(15).reshape(5, 3))
print(df_obj_app)


# Concatenating: Joins data from separate sources in one object - similar to row or column appending

df_concat_col = pd.concat([df_obj, df_obj_app], axis=1)  # remember to put square brackets around all the dataframes
print(df_concat_col)

df_concat_row = pd.concat([df_obj, df_obj_app])
print(df_concat_row)

# Adding data
series_obj = Series(np.arange(5))
series_obj.name = 'added_variable'
print(series_obj)

# using join method
variable_added_join = DataFrame.join(df_obj,series_obj, sort='added_variable',lsuffix='l', rsuffix='r')
print(variable_added_join)

# using concat method
variable_added_concat_row = pd.concat([df_obj, series_obj], ignore_index=True)
print(variable_added_concat_row)

variable_added_concat_col = pd.concat([df_obj, series_obj], axis=1, ignore_index=True)
print(variable_added_concat_col)

# using append method
variable_added_append = DataFrame.append(df_obj, series_obj, ignore_index=True)
print(variable_added_append)


# Sorting the dataframe
df_sorted = df_obj.sort_values(by=5, ascending=[False],)
print(df_sorted)

########################## Grouping and Aggregation ####################################################################

cars = pd.read_csv(cars_address, header="infer")
cars = cars.rename(columns={'Unnamed: 0':'car_names'})

print(cars.head())

# Group by
cars_group_cyl = cars.groupby(by='cyl')
print(cars_group_cyl.describe())

cars_group_am = cars.groupby(by='am')
print(cars_group_am.mpg.mean())

cars_group_it = cars.groupby(by='item_type')
print(cars_group_it.sum())


############################# Chapter 2: Data Preparation End ##########################################################

########################## Chapter 5: Basic Math and Arithmetic Start ##################################################

np.set_printoptions(precision=2)

# Creating arrays using a list
a = np.arange(1,7)
print(a)
b = np.array([[10,20,30],[40,50,60]])
print(b)

# Creating arrays via assignment
np.random.seed(25)
c = np.random.randn(6)
print(c)
c2 = np.random.rand(6)
print(c2)
d = np.arange(1,35)
print(d)


# Performing arithmetic on arrays
print(a*10)
print(c + a)
print(c-a)
print(c/a)
print(c*a)


## Matrices multiplication and basic algebra

np.set_printoptions(precision=2)

aa = np.array([[2.,4.,6.],[1.,3.,5.],[10.,20.,30.]])
print(aa)

bb = np.arange(9.).reshape(3, 3)
print(bb)

print(aa*bb)

# Dot product
np.dot(aa,bb)


############################## Generate Summary Statistics #############################################################
cars.describe()
cars.mpg.describe()
cars.mpg.max()
cars.mpg.idxmax() #index of the maximum value was found in the mpg variable
cars.cyl.std()
cars.qsec.var()

############################# Summarizing categorical data #############################################################

# cars.groupby('gear').sum()

cars['gear_group'] = pd.Series(cars.gear, dtype="category")
cars['gear_group'].value_counts()

# Crosstab creating
pd.crosstab(cars.am, cars.gear)
pd.pivot_table(data=cars, index=['carb', 'gear'], columns=['am', 'cyl'], values='mpg', aggfunc='sum')


############################### Parametric Correlation #################################################################
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
from pylab import rcParams

import scipy
from scipy.stats.stats import pearsonr, spearmanr

rcParams['figure.figsize'] = 8,4
plt.style.use('seaborn-whitegrid')

# The Pearson Correlation
sb.pairplot(cars)
plt.show()
X = cars[['mpg', 'hp', 'qsec', 'wt']]
sb.pairplot(X)
plt.show()

# Correlation using scipy
scipy_correlation_pearson, pvalue = pearsonr(cars.mpg, cars.hp)
scipy_correlation_pearson

# Pandas for correlation
pandas_correlation_pearson = X.corr()
print(pandas_correlation_pearson)

# Using seaborn to visualize Pearson correlation coefficient
sb.heatmap(pandas_correlation_pearson)
plt.show()


############################### Non-Parametric Correlation #############################################################

# Categorical Variables, non-normally distributed correlation

# Spearman's Correlation : On Ordinal numeric data, but ranked like categorical variables
# Variables are non linearly related
# Data is non-normally distributed


# Chi-Square Test for Independence
# p < 0.05 --> reject null hypothesis and conclude that variables are correlated
# Assumes: Variables are categorical or numeric (you have binned the numeric to create categories)


import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
from pylab import rcParams

import scipy
from scipy.stats.stats import pearsonr, spearmanr

rcParams['figure.figsize'] = 8,4
plt.style.use('seaborn-whitegrid')

X = cars[['cyl', 'vs', 'am', 'gear']]
sb.pairplot(X)
plt.show()

cyl = cars.cyl
vs = cars.vs
am = cars.am
gear = cars['gear']

spearmanr_coeff, pvalue = spearmanr(cyl, vs)
print('Spearman Ranked Correlation Coeff %0.3f' % (spearmanr_coeff))

spearmanr_coeff, pvalue = spearmanr(cyl, am)
print('Spearman Ranked Correlation Coeff %0.3f' % (spearmanr_coeff))

spearmanr_coeff, pvalue = spearmanr(cyl, gear)
print('Spearman Ranked Correlation Coeff %0.3f' % (spearmanr_coeff))

spearmanr_coeff_matrix = X.corr(method="spearman")
spearmanr_coeff_matrix

sb.heatmap(spearmanr_coeff_matrix, xticklabels=spearmanr_coeff_matrix.columns.values,
           yticklabels=spearmanr_coeff_matrix.columns.values)
plt.show()

# Chi-Square Test for Independence
from scipy.stats import chi2_contingency

table = pd.crosstab(cyl, am)
chi2, p, dof, expected = chi2_contingency(table.values)
print('Chi-Square statistic %0.3f p_value %0.3f' % (chi2, p))

table = pd.crosstab(cyl, vs)
chi2, p, dof, expected = chi2_contingency(table.values)
print('Chi-Square statistic %0.3f p_value %0.3f' % (chi2, p))

table = pd.crosstab(cyl, gear)
chi2, p, dof, expected = chi2_contingency(table.values)
print('Chi-Square statistic %0.3f p_value %0.3f' % (chi2, p))

# Since none of the p-values are greater than 0.05, we must reject the null hypothesis and variables are correlated
sb.pairplot(X)
plt.show()


############################### Transforming dataset distribution ######################################################
## Data Scaling: Needed for variable standardization, increase, decrease or standardize spread: Very crucial for ML

# 2 ways to scale the data
#### 1 normalize i.e., between 0 to 1
#### 2 standardize it i.e., zero mean and unit variance

### scikit-learn for scaling, centering, normalizing and binning your data

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sb

import scipy
from scipy.stats.stats import pearsonr, spearmanr

import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import scale

rcParams['figure.figsize'] = 8,4
plt.style.use('seaborn-whitegrid')

cars = pd.read_csv(cars_address,header="infer")
cars = cars.rename(columns={'Unnamed: 0':'car_names'})

# plotting mpg
plt.plot(cars.mpg)
plt.show()

cars['mpg'].describe()

# creating matrix for mpg
mpg_matrix = cars['mpg'].values.reshape(-1,1)

# scale between 0 to 1
scaled_0_1 = preprocessing.MinMaxScaler()
scaled_mpg =  scaled_0_1.fit_transform(mpg_matrix)
plt.plot(scaled_mpg)
plt.show()

# scale between 0 and 10
scaled_0_10 = preprocessing.MinMaxScaler(feature_range=(0, 10))
scaled_mpg_10 =  scaled_0_10.fit_transform(mpg_matrix)
plt.plot(scaled_mpg_10)
plt.show()

# Unscale from 0 to 10 to original
scaled_0_10.fit(mpg_matrix)
scaled_0_10.get_params()
unscaled_mpg = scaled_0_10.inverse_transform(scaled_mpg_10)
plt.plot(unscaled_mpg)
plt.show()


# Using scale() to scale your features
standardise_mpg = scale(cars.mpg, axis=0, with_mean=False, with_std=False)
plt.plot(standardise_mpg)
plt.show()

standardise_mpg = scale(cars.mpg)
plt.plot(standardise_mpg)
plt.show()

# Normalisation is a procedure followed to bring the data closer to the requirements of the algorithms, or at least
# to pre-process data so as to ease the algorithm's job. Variables can be normalised (to unit zero mean and unit
# variable, or to the interval [0, 1]), data elements can be normalised (when all their attributes have the
# same 'units') and the target variable can be normalised too (using a logarithmic transform). The choice to do
# or not to do normalization is of course left to the practitioner, but it can be advised with virtually no risk
# to always perform variable normalisation to [0, 1] when the variable values are bounded, to zero mean and unit
# variance otherwise, and to perform log transform of the target whenever it is skewed.
# http://www.damienfrancois.be/blog/pivot/entry.php?id=8




############################### Outlier Analysis #######################################################################

############################################# Only Univariate methods ################################

#### 1. Box plot methods - remove / cap the outliers at 1.5 * IQR i.e. 25% to 75% range

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sb

rcParams['figure.figsize'] = 5, 4
sb.set_style('whitegrid')

iris_data = pd.read_csv(iris_data_address)
iris_data.columns = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Species']

x = iris_data.iloc[:, 0:4].values
y = iris_data.iloc[:, 4].values

# Identifying outliers from Tukey boxplots
iris_data.boxplot(return_type='dict')
plt.show()
iris_data.boxplot(return_type='axes')
plt.show()
iris_data.boxplot(return_type='both')
plt.show()

# sepal width > 4 and < 2.05 could be potential outliers

sepal_width = x[:, 1]
iris_outliers = (sepal_width > 4.05) | (sepal_width < 2.05)
iris_data[iris_outliers].sort_values(by='Sepal Width', ascending=False)

# Applying Tukey outlier labeling
pd.options.display.float_format = '{:.1f}'.format
x_df = pd.DataFrame(x)
print(x_df.describe())

# 25% is 2.8, 75% is 3.3, IQR = 3.3 - 2.8 = 0.5
# lower bound = 2.8- 1.5*IQR = 2.05
# upper bound = 3.3 + 1.5*IRQ = 4.05


############################################# Multivariate methods ##################################

data = iris_data.iloc[:, 1:4].values
target = iris_data.iloc[:, 4].values
sb.boxplot(x='Species', y='Sepal Length', data=iris_data, palette='hls')
plt.show()

# Scatterplot matrix
sb.pairplot(iris_data, hue='Species', palette='hls')
plt.show()

############################################ Quiz ######################################################################

cars['qsec'].describe()
a = np.array([[1., 3.], [2., 4.]])
b = np.array([[5., 8.], [7., 9.]])

xx = np.array([[7., 9.], [5., 12.]])
yy = np.array([[2., 8.], [7., 4.]])
xx.__matmul__(yy)

xx=np.array([[1.,2.,3.],[4.,5.,6.]])
yy=np.array([[10.,11.],[20.,21.],[30.,31.]])
xx.__matmul__(yy)
np.matmul(xx,yy)

1.936 + 1.5*(1.936 - 1.714)
1.714 - 1.5*(1.936 - 1.714)

cars[(cars.vs == 1) & (cars.cyl == 6)]
cars[cars.cyl == 6]
pd.crosstab(cars.vs, cars.cyl)



