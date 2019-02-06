# prepackPy

## topic:

Package that contains methods for standard data preprocessing tasks.

## Function Descriptions:

### `splitter`

| Input Parameters | Input Type             | Output Parameters | Output Type    |
|------------------|------------------------|-------------------|----------------|
| X                | dataframe, numpy array | y train           | 1D numpy array |
| target index     | integer                | y test            | 1D numpy array |
| split size       | float                  | X train           | numpy array    |
| seed             | integer                | X test            | numpy array    |

### `stdizer`

| Input Parameters | Input Type             | Output Parameters | Output Type |
|------------------|------------------------|-------------------|-------------|
| X                | dataframe, numpy array | standardized X    | numpy array |

### `na_counter`

| Input Parameters | Input Type             | Output Parameters                                | Output Type |
|------------------|------------------------|--------------------------------------------------|-------------|
| X                | dataframe, numpy array | dictionary(key = column index, value = NA count) | dictionary  |
