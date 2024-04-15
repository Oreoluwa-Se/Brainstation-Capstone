from collections import deque
from datetime import datetime
from enum import Enum, auto
from typing import Dict, Any, List, Type, Set
import os
import pickle
import polars as pl
import random
import numpy as np


class TableInfoManagers:
    @staticmethod
    def infer_dtype_from_sample(df: pl.DataFrame) -> pl.DataFrame:
        for column in df.columns:
            # Attempt to get the first non-null value
            sample_series = df.select(
                pl.col(column).filter(pl.col(column).is_not_null()).head(1)
            )

            if not sample_series[column].is_empty():
                sample_value = sample_series[column][0]

                # Determine data type based on sample value
                if isinstance(sample_value, int):
                    df = df.with_columns(df[column].cast(pl.Int64))
                elif isinstance(sample_value, float):
                    df = df.with_columns(df[column].cast(pl.Float64))
                elif isinstance(sample_value, str):
                    # Check for date format YYYY-MM-DD
                    try:
                        datetime.strptime(sample_value, "%Y-%m-%d")
                        df = df.with_columns(
                            pl.col(column)
                            .str.strptime(pl.Date, "%Y-%m-%d")
                            .alias(column)
                        )
                    except ValueError:
                        # Not a date, check if it could be a float in disguise
                        try:
                            float(sample_value)
                            df = df.with_columns(df[column].cast(pl.Float64))
                        except ValueError:
                            pass  # Leave as string if all checks fail

        return df

    @staticmethod
    def num_ascii_represented(value: Any) -> bool:
        # Check if the value is an integer or a float equivalent to an integer
        if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
            # Convert to integer in case it's a float
            value = int(value)
            # Check if the value is greater than 0 and less than 9
            return 0 <= value <= 9
        return False

    @staticmethod
    def change_column_type(df: pl.DataFrame) -> pl.DataFrame:
        """
        Adjusts the type of columns in a Polars DataFrame based on naming conventions:
        - Columns ending with 'P' or 'A' are cast to Float64.
        - Columns ending with 'D' or named 'date_decision' are cast to Date.
        """

        # define type transforms in a dictionary
        type_map = {"P": pl.Float64, "A": pl.Float64, "D": pl.Date}

        transform = []

        for col in df.columns:
            last_char = col.strip()[-1]

            if col == "date_decision":
                transform.append(pl.col(col).cast(pl.Date))
            elif last_char in type_map:
                transform.append(pl.col(col).cast(type_map[last_char]))

        if transform:
            df = df.with_columns(transform)

        return df

    def load_data_downloaded(base_loc: str, cat: str = "train"):
        # loading static cases
        df = pl.concat(
            [
                TableInfoManagers.change_column_type(
                    pl.read_csv(base_loc + f"csv_files/{cat}/{cat}_static_0_0.csv")
                ),
                TableInfoManagers.change_column_type(
                    pl.read_csv(base_loc + f"csv_files/{cat}/{cat}_static_0_1.csv")
                ),
            ],
            how="vertical_relaxed",
        ).join(
            TableInfoManagers.change_column_type(
                pl.read_csv(base_loc + f"csv_files/{cat}/{cat}_static_cb_0.csv")
            ),
            on="case_id",
            how="left",
        )

        df = df.join(
            TableInfoManagers.change_column_type(
                pl.read_csv(base_loc + f"csv_files/{cat}/{cat}_base.csv")
            ),
            on="case_id",
            how="inner",
        )

        print(f"------> Size of {cat} dataset: {df.shape} <------")
        return df


class DataTracker:
    def __init__(self) -> None:
        self.__data = deque()

    def insert(self, value: Any) -> None:
        self.__data.append(value)

    def get_front(self) -> Any:
        return self.__data.popleft() if self.__data else None

    def get_array(self) -> List[Any]:
        return list(self.__data)

    @property
    def size(self) -> int:
        return len(self.__data)

    @property
    def empty(self) -> bool:
        return not self.__data


class MetaData:
    def __init__(self) -> None:
        self.trackers = {
            "number": DataTracker(),
            "amount": DataTracker(),
            "date": DataTracker(),
        }
        self.text_id = -1

    def __getitem__(self, key: str):
        return self.trackers.get(key.lower())

    def insert(self, data_name: str, value: Any):
        data_tracker = self[data_name]
        if data_tracker is None:
            raise ValueError(f"Invalid data name: {data_name}")
        data_tracker.insert(value)

    def get_front(self, data_name: str) -> Any:
        data_tracker = self[data_name]
        if data_tracker is None:
            raise ValueError(f"Invalid data name: {data_name}")
        return data_tracker.get_front()

    def get_array(self, data_name: str) -> List[Any]:
        data_tracker = self[data_name]
        if data_tracker is None:
            raise ValueError(f"Invalid data name: {data_name}")
        return data_tracker.get_array()

    def empty(self) -> bool:
        return all(tracker.is_empty() for tracker in self.trackers.values())

    def __str__(self) -> str:
        out = f"Text_id: {self.text_id}\n"
        out += "\n".join(
            f"{name.capitalize()} Queue Length: {tracker.size}"
            for name, tracker in self.trackers.items()
        )
        return out


class BatchMeta:
    """Batch information for running a session"""

    def __init__(self) -> None:
        self.texts: List[str] = []
        self.tracked_meta: List[MetaData] = []
        self.target: List[Dict[str, int]] = []
        self.__unique_data: Dict[str, Set[str]] = {
            "date": set(),
            "number": set(),
            "amount": set(),
        }

        self.curr_text_id = 0

    def add_data(self, data_type: str, val: str) -> None:
        if data_type not in self.__unique_data:
            raise ValueError(f"Invalid data type: {data_type}")
        self.__unique_data[data_type].add(val)

    def get_data(self, data_type: str) -> List[str]:
        if data_type not in self.__unique_data:
            raise ValueError(f"Invalid data type: {data_type}")
        return list(self.__unique_data[data_type])

    def add_meta(self, md: MetaData) -> None:
        md.text_id = self.curr_text_id
        self.curr_text_id += 1
        self.tracked_meta.append(md)

    def get_array(self, data_type: str):
        if data_type not in self.__unique_data:
            raise ValueError(f"Invalid data type: {data_type}")
        return list(self.__unique_data[data_type])

    def __str__(self) -> str:
        return f"Length of texts: {len(self.texts)}\n" + "\n".join(
            f"Length of unique {key}: {len(value)}"
            for key, value in self.__unique_data.items()
        )


class SamplingStrategy(Enum):
    BASE = auto()
    OTHER = auto()


class TableHandler:
    """Handles indididual tables and prepares them for batching"""

    def __init__(
        self, data: pl.DataFrame, target_col: str = "target", unique_id: str = "case_id"
    ) -> None:
        self.data: pl.DataFrame = TableInfoManagers.infer_dtype_from_sample(data)
        self.target_col = target_col
        self.unique_id = unique_id
        self.strategy = SamplingStrategy.BASE
        self.samp_percentage: float = None

        # sets up certain parameters
        self._stratify_data_by_target()
        self._find_minority()

    @property
    def shape(self):
        return self.data.shape

    def _find_minority(self):
        min_val = min(self.base_ratio.values())
        for k, v in self.base_ratio.items():
            if min_val == v:
                self.minority_group = k
                break

    def _stratify_data_by_target(self):
        groups = self.data.group_by(self.target_col).agg(
            pl.col(self.unique_id).alias("indices")
        )
        groups = groups.to_dict(as_series=False)

        self.groups = {}
        for idx in groups[self.target_col]:
            self.groups[idx] = groups["indices"][idx]

        # initializing cursors
        self.group_cursors = {k: 0 for k in self.groups}

        # gets stratified data
        total_samples = float(sum(len(indices) for indices in self.groups.values()))

        self.base_ratio = {
            k: float(len(indices)) / total_samples for k, indices in self.groups.items()
        }

    @staticmethod
    def increment_values(base, step, max_value=0.5):
        """Generate a list of incremented values starting from 'base' by 'step' until 'max_value'."""
        results = []
        for current_value in np.arange(base, max_value, step):
            results.append(current_value)
        return results

    def choose_sampling_strat(
        self, precision: float = 1.0, recall: float = 1.0, force_other=False
    ):
        threshold = 0.5
        base = round(min(self.base_ratio.values()), 2)
        strategy = None

        if recall < threshold:
            step_options = np.linspace((0.5 + base) * 0.5, 0.5, num=8)
            strategy = SamplingStrategy.OTHER
        elif precision < threshold:
            step_options = np.linspace(base, (0.5 + base) * 0.5, num=8)
            strategy = SamplingStrategy.OTHER
        elif force_other:
            step_options = np.array([0.1, 0.2, 0.05, 0.07])
            strategy = SamplingStrategy.OTHER

        if strategy:
            self.strategy = strategy
            if step_options.size > 0:
                self.samp_percentage = np.random.choice(step_options)
            else:
                self.samp_percentage = None
        else:
            self.strategy = SamplingStrategy.BASE
            self.samp_percentage = None

    def reset(self, only_idxs: bool = False):
        if not only_idxs:
            for _, value in self.groups.items():
                random.shuffle(value)

        # reset cursor
        self.group_cursors = {k: 0 for k in self.groups}

    @staticmethod
    def handling_A_or_num(col_name: str, value: Any):
        data_type, tag_type, msg = "", "", ""

        if col_name[-1] == "A":
            data_type = "amount"
            tag_type = "<|AMOUNT|>"
        else:
            data_type = "number"
            tag_type = "<|NUM|>"

        if TableInfoManagers.num_ascii_represented(value):
            msg = f"{col_name} is {data_type} {int(value)}"
            return msg, None
        else:
            msg = f"{col_name} is {tag_type}"
            return msg, data_type

    @staticmethod
    def row_to_text(
        row_info: pl.DataFrame,
        agg_info: BatchMeta,
        col_types: List[Type],
        col_names: List[str],
        ignore: List[str] = [],
        output: List[str] = [],
    ) -> None:
        row_msg = []
        target_dict = {}
        td = MetaData()

        for value, dtype, name in zip(row_info, col_types, col_names):
            name = name.strip()
            if name in ignore:
                continue

            if name in output:
                target_dict[name] = float(value)
                continue

            msg = f"{name} is empty"
            if isinstance(dtype, (pl.Int64, pl.Float64)):
                if value is not None:
                    msg, data_type = TableHandler.handling_A_or_num(name, value)

                    if data_type:
                        # insert into trackers
                        td.insert(data_type, value)
                        agg_info.add_data(data_type, str(value))

            elif isinstance(dtype, pl.Date):
                if value is not None:
                    msg = f"{name} is <|DATE|>"

                    # insert into trackers
                    td.insert("date", str(value))
                    agg_info.add_data("date", str(value))

            elif isinstance(dtype, pl.String):
                if value is not None:
                    msg = f"{name} is {str(value)}"

            elif isinstance(dtype, pl.Boolean):
                if value is not None:
                    msg = f"{name} is {str(value)}"

            row_msg.append(msg)

        # update row text information

        agg_info.texts.append(", ".join(row_msg))
        agg_info.add_meta(td)
        agg_info.target.append(target_dict)

    def cursor_management(self, indices, batch_size, label):
        start_idx = self.group_cursors[label]
        end_idx = start_idx + batch_size

        if end_idx > len(indices):
            # wrap around
            end_idx %= len(indices)
            sampled = indices[start_idx:] + indices[:end_idx]
        else:
            sampled = indices[start_idx:end_idx]

        self.group_cursors[label] = end_idx % len(indices)

        return sampled

    def _stratified_sampling(self, batch_size):
        samps_per_class = {
            k: max(int(v * float(batch_size)), 1) for k, v in self.base_ratio.items()
        }

        sampled_idxs = []
        for label, idxs in self.groups.items():
            needed = self.cursor_management(idxs, samps_per_class[label], label)
            sampled_idxs.extend(needed)

        return sampled_idxs

    def _sample_random(self, batch_size):
        samps_per_class = {}

        total_majority_samples = int((1.0 - self.samp_percentage) * batch_size)
        total_minority_samples = batch_size - total_majority_samples

        majority_classes = [k for k in self.base_ratio if k != self.minority_group]

        samps_per_class[self.minority_group] = total_minority_samples

        majority_samples_each = total_majority_samples // len(majority_classes)
        remainder = total_majority_samples % len(majority_classes)

        for k in majority_classes:
            samps_per_class[k] = majority_samples_each

        for i in range(remainder):
            samps_per_class[majority_classes[i]] += 1

        sampled_idxs = []
        for label, idxs in self.groups.items():
            needed = self.cursor_management(idxs, samps_per_class[label], label)
            sampled_idxs.extend(needed)

        return sampled_idxs

    def get_meta_data(
        self,
        batch_size: int,
        ignore_list: List[str] = ["case_id"],
        output_list: List[str] = ["WEEK_NUM", "target"],
        verbose: bool = False,
    ):
        # gets stratified data
        indices = []

        if self.strategy == SamplingStrategy.BASE or self.samp_percentage is None:
            indices = self._stratified_sampling(batch_size)
        elif self.strategy == SamplingStrategy.OTHER:
            indices = self._sample_random(batch_size)
        else:
            raise ValueError("Nosampling strategy selected")

        random.shuffle(indices)
        batch_data = self.data.filter(pl.col(self.unique_id).is_in(indices))
        col_types = self.data.dtypes
        col_names = self.data.columns

        # batch
        agg_info: BatchMeta = BatchMeta()
        for row in batch_data.rows():
            self.row_to_text(
                row, agg_info, col_types, col_names, ignore_list, output_list
            )

        if verbose:
            print(f"Batch generated with {len(indices)} records.")

        return agg_info


class DataBatcher:
    def __init__(self) -> None:
        self.train = None
        self.valid = None
        self.test = None
        self.descriptive_Tags = {}
        self.cols = {}

    @staticmethod
    def calc_splits(
        total_value: int, train_test_split: float, validation_split: float = None
    ):
        num_train = int(total_value * train_test_split)
        remaining_samps = total_value - num_train

        if validation_split is None:
            return num_train, remaining_samps, 0

        valid_split = int(remaining_samps * validation_split)

        return num_train, valid_split, (remaining_samps - valid_split)

    def stratified_split(
        self,
        minor_class: pl.DataFrame,
        major_class: pl.DataFrame,
        train_test_split: float,
        validation_split: float = None,
    ):
        combined_data = pl.concat([minor_class, major_class], how="vertical_relaxed")
        combined_data = combined_data.sample(n=combined_data.height, shuffle=True)

        # get grouped data
        groups = combined_data.group_by("WEEK_NUM").agg(pl.col("case_id").alias("idxs"))
        groups = groups.to_dict(as_series=False)

        groups_dict = {}
        for idx in groups["WEEK_NUM"]:
            groups_dict[idx] = groups["idxs"][idx]

        # Initialize empty DataFrames for splits
        train_data, test_data, valid_data = None, None, None

        for indices in groups_dict.values():
            num_train, num_valid, num_test = self.calc_splits(
                len(indices), train_test_split, validation_split
            )

            week_data = combined_data.filter(pl.col("case_id").is_in(indices))

            week_train = week_data.slice(0, num_train)
            week_valid = week_data.slice(num_train, num_valid)
            week_test = (
                week_data.slice(num_train + num_valid, num_test)
                if validation_split is not None
                else pl.DataFrame()
            )

            train_data = (
                week_train
                if train_data is None
                else pl.concat([train_data, week_train], how="vertical_relaxed")
            )
            valid_data = (
                week_valid
                if valid_data is None
                else pl.concat([valid_data, week_valid], how="vertical_relaxed")
            )

            if validation_split is not None:
                test_data = (
                    week_test
                    if test_data is None
                    else pl.concat([test_data, week_test], how="vertical_relaxed")
                )

        if validation_split is None:
            return train_data, valid_data, None

        return train_data, valid_data, test_data

    def load_and_process(
        self,
        base_loc: str,
        minority_loc: str,
        majority_loc: str,
        training: bool = True,
        validation_split: float = -1.0,
        train_test_split: float = 0.8,
        seed: int = 1,
    ):
        minor_class = pl.read_csv(minority_loc)
        major_class = pl.read_csv(majority_loc)

        if training:
            train_data, valid_data, test_data = self.stratified_split(
                minor_class, major_class, train_test_split, validation_split
            )

            self.train = TableHandler(
                train_data.sample(n=train_data.height, shuffle=True)
            )
            self.valid = TableHandler(
                valid_data.sample(n=valid_data.height, shuffle=True)
            )

            if test_data is not None:
                self.test = TableHandler(
                    test_data.sample(n=test_data.height, shuffle=True)
                )
        else:
            test_data = pl.concat([minor_class, major_class], how="vertical_relaxed")
            self.test = TableHandler(test_data.sample(n=test_data.height, shuffle=True))

        # Extract descriptive information for each column
        feat_defs = pl.read_csv(base_loc + "feature_definitions.csv").to_dicts()
        col_defs = {row["Variable"]: row["Description"] for row in feat_defs}

        self.descriptive_tags = {
            col: col_defs.get(col, "No description available")
            for col in minor_class.columns
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
            for col in minor_class.columns
            if col.endswith(("P", "M", "D", "T", "L"))
        ]

        self.num_cols = [col for col in minor_class.columns if col.endswith("A")]
        self.col_types = minor_class.dtypes

        print(f"------> Size of categorical columns: {len(self.cat_cols)} <------")
        print(f"------> Size of numeric columns: {len(self.num_cols)} <------")

        # Dropping the 'MONTH' column from both train and test data
        if self.test is not None:
            self.test.data = self.test.data.drop(["MONTH"])
        if training:
            self.train.data = self.train.data.drop(["MONTH"])
            self.valid.data = self.valid.data.drop(["MONTH"])

    def get_meta_data(
        self,
        batch_size: int,
        data_type="train",
        ignore_list: List[str] = ["case_id"],
        output_list: List[str] = ["WEEK_NUM", "target"],
        verbose: bool = False,
    ):
        # self.processed_row_idx
        if data_type == "train":
            dtp = self.train
        elif data_type == "valid":
            dtp = self.valid
        else:
            dtp = self.test

        return dtp.get_meta_data(batch_size, ignore_list, output_list, verbose)

    @property
    def nones(self):
        return self.test is None and self.train is None and self.valid is None

    def reset(self, only_indexes: bool = True, training=False):
        if training:
            self.train.reset(False)
            self.valid.reset(only_indexes)

        if self.test is not None:
            self.test.reset(only_indexes)

    def save_state(self, location: str):
        if location.endswith(".pkl"):
            filepath = location
            directory = os.path.dirname(filepath)
        else:
            directory = os.path.join(location, "Batch_Info")
            filepath = os.path.join(directory, "batch_state.pkl")

        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(filepath, "wb") as file:
            pickle.dump(self.__dict__, file)

    def load_state(self, location: str):
        if location.endswith(".pkl"):
            filepath = location
        else:
            filepath = os.path.join(location, "Batch_Info", "batch_state.pkl")

        if os.path.exists(filepath):
            with open(filepath, "rb") as file:
                state = pickle.load(file)
                self.__dict__.update(state)
        else:
            print("No saved state found at the specified location.")
