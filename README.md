# prepackPy

## Topic:

Package that contains methods for standard data staging, preprocessing, and exploratory tasks.

## Function Descriptions:

---

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
