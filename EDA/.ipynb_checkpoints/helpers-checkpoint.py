import polars as pl
import pandas as pd

def change_column_type(df: pl.DataFrame)->pl.DataFrame:
    transforms = []
    # perform transformations and put it in the lust
    for col in df.columns:
        # Column type determined by last letter
        if col[-1] in ("P", "A"):
            transforms.append(pl.col(col).cast(pl.Float64).alias(col))
        elif col[-1] in ("D") or col == 'date_decision':  # Use 'elif' to ensure mutual exclusivity
            transforms.append(pl.col(col).cast(pl.Date).alias(col))
    
    if transforms:
        df = df.with_columns(transforms)

    return df

def find_duplicate_values(inp_dict):
    # Reverse the dictionary to map values to keys (since values might be the same, use a list to collect keys)
    values_to_keys = {}

    for key, value in inp_dict.items():
        if value not in values_to_keys:
            values_to_keys[value] = []
        # 
        values_to_keys[value].append(key)

    # Filter out entries with unique values.
    return [keys for keys in values_to_keys.values() if len(keys)>1]

def verify_exact_duplicates_from_pairs(df: pl.DataFrame, column_pairs: list):
    exact_duplicates = []
    for pair in column_pairs:
        if len(pair) == 2:  # Ensure we're working with pairs
            # Select the columns based on the pair and check if they are exactly the same
            col1, col2 = pair
            if df.select(col1).equals(df.select(col2), null_equal=True):
                exact_duplicates.append(pair)
    return exact_duplicates

def group_and_sort_columns(df: pd.DataFrame):
    # calculate number of unique values for each column
    unique_counts = df.apply(lambda x: x.nunique())

    # group and hold results
    groups = {}
    for col, count in unique_counts.items():
        if count not in groups:
            groups[count] = []

        groups[count].append(col)

    # sort dictionary by unique values
    sorted_groups = dict(sorted(groups.items()))

    return list(sorted_groups.items())

# converts rows into comma seperated text
def row_to_string(row_info, col_types, col_names)->str:
    row_str = []
    idx = 0
    
    for value, dtype, name in zip(row_info, col_types, col_names):
        if idx == 0:
            idx += 1
            continue
        elif idx == len(col_types)-1:
            break
        idx+=1    
        msg = f"{name} is empty"         
        if isinstance(dtype, (pl.Int64, pl.Float64)):
            if value is not None:
                msg = f"{name} has value" 
        elif isinstance(dtype, pl.String):
            if value is not None:
                msg = f"{name} value is {value}"
        elif isinstance(dtype, pl.Date):
            if value is not None:
                msg = f"{name}: {str(value)}"
        elif isinstance(dtype, pl.Boolean):
            if value is not None:
                msg = f"{name} is {value}"            
        row_str.append(msg)
        
    return ", ".join(row_str) 