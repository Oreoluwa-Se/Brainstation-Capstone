import polars as pl
import pandas as pd
import csv
import os


def change_column_type(df: pl.DataFrame) -> pl.DataFrame:
    transforms = []
    # perform transformations and put it in the lust
    for col in df.columns:
        # Column type determined by last letter
        if col[-1] in ("P", "A"):
            transforms.append(pl.col(col).cast(pl.Float64).alias(col))
        elif (
            col[-1] in ("D") or col == "date_decision"
        ):  # Use 'elif' to ensure mutual exclusivity
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
    return [keys for keys in values_to_keys.values() if len(keys) > 1]


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
def row_to_string(row_info, col_types, col_names, ignore: list = []) -> str:
    """_summary_

    Args:
        row_info (_type_): _description_
        col_types (_type_): _description_
        col_names (_type_): _description_
        ignore (list, str): list of columns to ignore. Defaults to [].

    Returns:
        str: _description_
    """
    row_str = []

    for value, dtype, name in zip(row_info, col_types, col_names):
        if name.strip() in ignore:
            continue

        msg = f"{name} is empty"
        if value is not None:
            if isinstance(dtype, (pl.Int64, pl.Float64)):
                # numeric columns handled in numeric section for now we just acknowledge it
                msg = f"{name} has value"
            elif isinstance(dtype, pl.String):
                if value is not None:
                    msg = f"{name} value is {str(value)}"
            elif isinstance(dtype, pl.Date):
                if value is not None:
                    msg = f"{name}: {str(value)}"
            elif isinstance(dtype, pl.Boolean):
                if value is not None:
                    msg = f"{name} is {str(value)}"

        row_str.append(msg)

    return ", ".join(row_str)


def table_to_text(
    location: str, df: pl.DataFrame, overwrite: bool = False, ignore: list = []
):
    """Converts tabular data to text following row_to_string

    Args:
        location (str): file storage location
        df (pl.DataFrame): dataframe to store
        overwrite (bool, optional): determines if the file should be overwritten. Defaults to False.
        ignore (list, str): list of columns to ignore. Defaults to [].
    """
    if os.path.exists(location) and not overwrite:
        print(f"Warning: The file at {location} already exists")
        return

    with open(location, "w") as file:
        col_types = df.dtypes
        col_names = df.columns

        for row in df.rows():
            row_string = row_to_string(row, col_types, col_names, ignore)
            file.write(row_string + "\n")

        print(f"Transformed rows in location {location}")


def append_csv_to_txt(csv_file_path: str, output_file_path: str):
    """Appends csv to end of txt file

    Args:
        csv_file_path (str): location of csv file
        output_file_path (str): location of output file
    """
    with open(csv_file_path, mode="r", newline="") as csv_file:
        csv_reader = csv.reader(csv_file)

        with open(output_file_path, mode="a", newline="\n") as txt_file:
            for row in csv_reader:
                row_string = ", ".join(row)
                txt_file.write(row_string + "\n")


def append_txt_to_txt(source_file_path: str, target_file_path: str):
    """
    Reads content from the source file and appends it to the target file.

    Parameters:
    source_file_path (str): The path to the source file.
    target_file_path (str): The path to the target file where the content will be appended.
    """
    # Open the source file and read its content
    with open(source_file_path, "r") as source_file:
        content = source_file.read()

    # Open the target file in append mode and write the content from the source file
    with open(target_file_path, "a") as target_file:
        target_file.write(content)


def save_chunk_samples(
    start_loc: int,
    end_loc: int,
    df: pl.DataFrame,
    store_loc: str,
    num_samples: int = 100,
    ignore: list = [],
):
    """Save chunks of the dataframe to storage location

    Args:
        start_loc (int): start point
        end_loc (int): end point
        df (pl.DataFrame): dataframe to chunk
        store_loc (str): storage location
        num_samples (int, optional): number of samples. Defaults to 100.
        ignore (list, str): list of columns to ignore. Defaults to [].
    """
    col = df.select(df.columns[start_loc:end_loc])
    sampled = col.sample(n=num_samples, with_replacement=False)

    # store to a
    loc = store_loc + f"_{start_loc}_{end_loc}.txt"
    table_to_text(loc, sampled, True, ignore)  # store
