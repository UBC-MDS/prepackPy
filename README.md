# prepackPy

## Team
1. Jingyun Chen: [jchen9314](https://github.com/jchen9314)
2. Anthony Chiodo: [apchiodo](https://github.com/apchiodo)
3. Sarah Watts: [smwatts](https://github.com/smwatts)

## Topic

Package that contains methods for standard data staging, preprocessing, and exploratory tasks.

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

#### `stdizer(X)`

**Description:** standardize features by removing the mean (centering on 0), and scaling by the standard deviation.  Accepts both pandas dataframes and numpy arrays as input.  Returns numpy array as output.

| Input Parameters | Input Type             | Output Parameters | Output Type |
|------------------|------------------------|-------------------|-------------|
| X                | dataframe, numpy array | standardized X    | numpy array |

---

#### `na_counter(X)`

**Description:** summarise the missing data (`NA` values) in a dataset.  Accepts both pandas dataframes and numpy arrays as input.  Returns dictionary where the key is the column index, and the value is the NA count as output.

| Input Parameters | Input Type             | Output Parameters                                | Output Type |
|------------------|------------------------|--------------------------------------------------|-------------|
| X                | dataframe, numpy array | dictionary(key = column index, value = NA count) | dictionary  |

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
