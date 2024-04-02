import polars as pl
from typing import Dict, Any, List, Type
import queue


def change_column_type(df: pl.DataFrame) -> pl.DataFrame:
    "Adjusts the type of the column"
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


class MetaData:
    def __init__(self) -> None:
        # Queue for tracking unique numerical information
        self.__number = []
        self.__amount = []
        self.__date = []

        # The converted text value
        self.text = "N/A"
        self.tokens: List[int] = []

    def insert_num(self, value: Any):
        self.__number.append(value)

    def get_number(self):
        return self.__number.pop(0)

    def get_num_array(self):
        return self.__number.copy()

    def insert_amount(self, value: Any):
        self.__amount.append(value)

    def get_amount(self):
        return self.__amount.pop(0)

    def get_amt_array(self):
        return self.__amount.copy()

    def insert_date(self, value: str):
        self.__date.append(value)

    def get_date(self):
        return self.__date.pop(0)

    def get_date_array(self):
        return self.__date.copy()

    def __str__(self) -> str:
        return (
            f"Text: {self.text}\n"
            f"Number Queue Length: {len(self.__number)}\n"
            f"Amount Queue Length: {len(self.__amount)}\n"
            f"Date Queue Length: {len(self.__date)}"
        )

    def __getitem__(self, key: str):
        if key.lower() == "number":
            return self.get_num_array()
        elif key.lower() == "amount":
            return self.get_amt_array()
        elif key.lower() == "date":
            return self.get_date_array()
        else:
            return []


class TableHandler:
    def __init__(self, data) -> None:
        self.data = data
        self.table_extract_idx = 0
        self.processed_row_idx = 0

    def reset(self, only_indexes: bool = True):
        if not only_indexes:
            self.data = None

        self.table_extract_idx = 0
        self.processed_row_idx = 0

    def extract_batches(
        self,
        batch_size: int,
        split_data: bool = True,
        verbose: bool = False,
    ):
        """
        Generator that yields batches of data, split into numeric and categorical,
        for training purposes.

        Args:
        batch_size (int): Number of rows per batch.
        data_type (str): Type of the data to process ('train' or 'test').
        verbose (bool): If True, prints detailed information for each batch.
        split_data (bool): If we want to section things into categorical and numeric cases

        Yields:
        Dict: A dictionary containing two DataFrames, one for numeric and one for categorical data.
        """
        total_rows = self.data.height

        while self.start_idx < total_rows:
            end_idx = min(self.table_extract_idx + batch_size, total_rows)
            data_batch = self.data.slice(self.table_extract_idx, batch_size)

            if not split_data:
                if verbose:
                    print(f"Batch shape: {data_batch.shape}")

                yield {"data": data_batch}
            else:
                yield self.split_data(data_batch, verbose=verbose)

            if verbose:
                print(f"Processed batch: {self.table_extract_idx} to {end_idx}")

            self.table_extract_idx += batch_size

    def split_data(
        self, df: pl.DataFrame, verbose: bool = False
    ) -> Dict[str, pl.DataFrame]:
        """
        Splits the dataframe into numeric and categorical subsets,
        applies preprocessing to numeric columns, and returns both subsets.

        Args:
        df (pl.DataFrame): The input dataframe.
        verbose (bool): If True, prints the shape and column names of the subsets.

        Returns:
        Dict[str, pl.DataFrame]: A dictionary containing the 'categorical' and 'numeric' dataframes.
        """
        # Process numeric columns
        num_df = df.select(self.num_cols)

        num_df = num_df.with_columns(
            [
                pl.when(num_df[col] == 0)
                .then(1e-6)  # Add a small number to zero values
                .otherwise(num_df[col])
                .pipe(
                    lambda col_data: pl.when(col_data > 0)
                    .then(col_data.log10())  # Take the log10 of positive values
                    .otherwise(col_data)
                )
                .alias(col)
                for col in self.num_cols
            ]
        )

        # Select categorical columns
        cat_df = df.select(self.cat_cols)

        if verbose:
            print(f"Shape of numeric dataframe: {num_df.shape}")
            print(f"Categorical columns: {self.cat_cols}")
            print(f"Shape of categorical dataframe: {cat_df.shape}")

        return {"categorical": cat_df, "numeric": num_df}

    @staticmethod
    def row_to_text(
        row_info: pl.DataFrame,
        cols_type: List[Type],
        cols_names: List[str],
        ignore: List[str] = [],
    ) -> MetaData:
        # Process begins
        row_msg = ["This client"]

        td = MetaData()
        for value, dtype, name in zip(row_info, cols_type, cols_names):
            if name.strip() in ignore:
                continue

            msg = f"has no {name} information"
            if isinstance(dtype, (pl.Int64, pl.Float64)):
                if name[-1] in ("A"):
                    if value is not None:
                        td.insert_amount(value)
                        msg = f"has a {name} amount of <|AMOUNT|>"
                elif value is not None:
                    td.insert_num(value)
                    msg = f"has a {name} value of <|NUM|>"

            elif isinstance(dtype, pl.Date):
                if value is not None:
                    td.insert_date(str(value))
                    msg = f"has {name} date of <|DATE|>"

            elif isinstance(dtype, pl.String):
                if value is not None:
                    msg = f"has a {name} value of {str(value)}"

            elif isinstance(dtype, pl.Boolean):
                if value is not None:
                    msg = f"has a {name} value of {str(value)}"

            row_msg.append(msg)

        # update row text information
        td.text = ", ".join(row_msg)

        return td

    def get_meta_data(
        self,
        batch_size: int,
        ignore_list: List[str] = ["case_id", "target"],
        verbose: bool = False,
    ):
        # self.processed_row_idx
        total_rows = self.data.height

        col_types = self.data.dtypes
        col_names = self.data.columns
        processed_list: List[MetaData] = []

        if self.processed_row_idx < total_rows:
            end_idx = min(self.processed_row_idx + batch_size, total_rows)
            data_batch = self.data.slice(self.processed_row_idx, batch_size)

            processed_list: List[MetaData] = []

            for row in data_batch.rows():
                processed_list.append(
                    self.row_to_text(row, col_types, col_names, ignore_list)
                )

            if verbose:
                print(f"Processed batch: {self.processed_row_idx} to {end_idx}")

            self.processed_row_idx += batch_size

        return processed_list


class DataBatcher:
    def __init__(self) -> None:
        self.train = None
        self.test = None
        self.decriptive_tags = {}
        self.cat_cols = []
        self.num_cols = []
        self.col_types = []

    def load_data(self, base_loc: str, category: str = "train"):
        # loading static cases
        df = pl.concat(
            [
                change_column_type(
                    pl.read_csv(
                        base_loc + f"csv_files/{category}/{category}_static_0_0.csv"
                    )
                ),
                change_column_type(
                    pl.read_csv(
                        base_loc + f"csv_files/{category}/{category}_static_0_1.csv"
                    )
                ),
            ],
            how="vertical_relaxed",
        ).join(
            change_column_type(
                pl.read_csv(
                    base_loc + f"csv_files/{category}/{category}_static_cb_0.csv"
                )
            ),
            on="case_id",
            how="left",
        )

        df = df.join(
            change_column_type(
                pl.read_csv(base_loc + f"csv_files/{category}/{category}_base.csv")
            ),
            on="case_id",
            how="inner",
        )

        print(f"------> Size of {category} dataset: {df.shape} <------")
        return df

    def load_and_process(
        self,
        base_loc: str,
        training: bool = True,
        train_test_split: float = 0.8,
        seed: int = 1,
    ):
        full_data = self.load_data(base_loc, "train" if training else "test")

        if training:
            full_data = full_data.sample(n=full_data.height, shuffle=True, seed=seed)
            # Stratified splitting
            train_frames = []
            test_frames = []
            for value in full_data["target"].unique():
                group = full_data.filter(pl.col("target") == value)
                split_point = int(group.height * train_test_split)
                train_frames.append(group[:split_point])
                test_frames.append(group[split_point:])

            self.train = TableHandler(pl.concat(train_frames))
            self.test = TableHandler(pl.concat(test_frames))
        else:
            self.test = TableHandler(full_data)

        # Extract descriptive information for each column
        feat_defs = pl.read_csv(base_loc + "feature_definitions.csv").to_dicts()
        col_defs = {row["Variable"]: row["Description"] for row in feat_defs}

        data_to_use = self.train.data if training else self.test.data
        self.descriptive_tags = {
            col: col_defs.get(col, "No description available")
            for col in data_to_use.columns
        }

        # Update descriptive tags with additional information
        additional_tags = {
            "date_decision": "This refers to the date when a decision was made regarding the approval of the loan.",
            "WEEK_NUM": "This is the week number used for aggregation. In the test sample, WEEK_NUM continues sequentially from the last training value of WEEK_NUM",
            "target": "This is the target value, determined after a certain period based on whether or not the client defaulted on the specific credit case (loan)",
        }

        self.descriptive_tags.update(additional_tags)

        # Get categorical and numeric columns
        self.cat_cols = [
            col
            for col in data_to_use.columns
            if col.endswith(("P", "M", "D", "T", "L"))
        ]

        self.num_cols = [col for col in data_to_use.columns if col.endswith("A")]
        self.col_types = data_to_use.dtypes

        print(f"------> Size of categorical columns: {len(self.cat_cols)} <------")
        print(f"------> Size of numeric columns: {len(self.num_cols)} <------")

        # Dropping the 'MONTH' column from both train and test data
        self.test.data = self.test.data.drop(["MONTH", "WEEK_NUM"])
        if training:
            self.train.data = self.train.data.drop(["MONTH", "WEEK_NUM"])

    def extract_batches(
        self,
        batch_size: int,
        data_type: str = "train",
        split_data: bool = True,
        verbose: bool = False,
    ):
        """
        Generator that yields batches of data, split into numeric and categorical,
        for training purposes.

        Args:
        batch_size (int): Number of rows per batch.
        data_type (str): Type of the data to process ('train' or 'test').
        verbose (bool): If True, prints detailed information for each batch.
        split_data (bool): If we want to section things into categorical and numeric cases

        Yields:
        Dict: A dictionary containing two DataFrames, one for numeric and one for categorical data.
        """
        dtp = self.train if data_type == "train" else self.test
        yield dtp.extract_batches(batch_size, split_data, verbose)

    def get_meta_data(
        self,
        batch_size: int,
        data_type="train",
        ignore_list: List[str] = ["case_id", "target"],
        verbose: bool = False,
    ):
        # self.processed_row_idx
        dtp = self.train if data_type == "train" else self.test
        return dtp.get_meta_data(batch_size, ignore_list, verbose)

    def reset(self, only_indexes: bool = True):
        if self.train is not None:
            self.train.reset(only_indexes)

        self.test.reset(only_indexes)
