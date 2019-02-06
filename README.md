# prepackPy

## Team
1. Jingyun Chen: [jchen9314](https://github.com/jchen9314)
2. Anthony Chiodo: [apchiodo](https://github.com/apchiodo)
3. Sarah Watts: [smwatts](https://github.com/smwatts)

## Topic

A common rule of thumb for data scientist is that the data preparation process will take approximately 80% of the total time on a project. Not only is this process time consuming, but it is also considered one of the less enjoyable components of a project ([Forbes, 2016](https://www.forbes.com/sites/gilpress/2016/03/23/data-preparation-most-time-consuming-least-enjoyable-data-science-task-survey-says/#3d12fbbf6f63)). To help address this problem, we have decided to build a package that will help improve some of the common techniques used in data preparation. This includes a function that will streamline the process of splitting a dataset into testing and training data (and provide a model ready output!), a function that incorporates more standardization methods then a data scientist could ever want _and_ a function that will allow data scientist to quickly understand the columns and quantity with `NA` values in a dataset.


## Function Descriptions

#### `splitter(X, target_index, split_size, seed)`

**Description:** consolidate scikit-learns current work flow for splitting a data set in to train and test sets, i.e. turn this:

```
data = pd.read_csv('data.csv')

X = data.iloc[:, 0:10]
y = data.iloc[:, 10:11]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

```
into this:

```

data = pd.read_csv('data.csv')

X_train, X_test, y_train, y_test = splitter(data, target_index='y', split_size=0.3, seed=42)

```

| Input Parameters | Input Type             | Output Parameters | Output Type    |
|------------------|------------------------|-------------------|----------------|
| X                | dataframe, numpy array | y train           | 1D numpy array |
| target index     | integer                | y test            | 1D numpy array |
| split size       | float                  | X train           | numpy array    |
| seed             | integer                | X test            | numpy array    |

---

#### `stdizer(X, col_index=None, method, method_args)`



**Description:** standardize features. Accepts both pandas dataframes and numpy arrays as input.  Returns numpy array as output.

Standardization methods:
1. Subtract mean and divide by standard deviation
2. Subtract mean
3. Divide by standard deviation
4. Subtract user specified value and divide by user specified vale
5. Linearly transform the data such that it is scaled to a specific range

These standardization techniques are based on [Minitab documentation](https://support.minitab.com/en-us/minitab/18/help-and-how-to/calculations-data-generation-and-matrices/standardize/standardize-columns-of-data/).

| Input Parameters | Input Type             | Output Parameters | Output Type |
|------------------|------------------------|-------------------|-------------|
| X                | dataframe, numpy array | standardized X    | numpy array |
| col_index        | list of indices        |         ""        |      ""     |
| method           | string                 |         ""        |      ""     |
| method_args      | list of lists          |         ""        |      ""     |
---

#### `na_counter(X)`

**Description:** summarise the missing data (`NA` values) in a dataset.  Accepts both pandas dataframes and numpy arrays as input.  Returns dictionary where the key is the column index, and the value is the NA count as output.

| Input Parameters | Input Type             | Output Parameters                               | Output Type |
|------------------|------------------------|-------------------------------------------------|-------------|
| X                | dataframe, numpy array | dictionary(key= column index, value = NA count) | dictionary  |

## Relationship to the Python ecosystem

#### `splitter`

The existing package/method is [`sklearn.model_selection.train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html), which only splits features/target into train features/target and test features/target.

What `splitter` will improve is that it will be able to separate the target variable from the dataset by specifying the column index of the target variable.

#### `stdizer`

The existing package/method is [`sklearn.preprocessing.StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html), which considers three standardization methods including subtracting mean and dividing by standard deviation, subtracting mean only, and dividing by standard deviation only.

`stdizer` will consider two more standardization techniques including subtracting the maximum value of each column and dividing by the minimum value of each column, and substracting the user specified mean and dividing by the user specified standard deviation.

#### `na_counter`

The existing package/method is [`pandas.DataFrame.describe`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html) or [`pandas.DataFrame.info`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.info.html), which contains a summary of the dataset including information of missing values. However, there is no method for finding and reporting where missing values exist in Python.

`na_counter` will take this problem into consideration. It will be able to return both the indices of columns that contains missing values, number of missing values, as well as the percentage of missing values in the columns.
