# prepackPy

## Team
1. Jingyun Chen: [jchen9314](https://github.com/jchen9314)
2. Anthony Chiodo: [apchiodo](https://github.com/apchiodo)
3. Sarah Watts: [smwatts](https://github.com/smwatts)

## Topic:

Package that contains methods to assist with standard data preprocessing tasks.

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

## Relationship to the Python ecosystem