from collections import Counter
import numpy as np
from typing import List, Pattern, Dict, Tuple
import regex as re
import math
import string
import pickle
import os


class BytePairEncodeAlgo:
    def __init__(
        self,
        pattern: Pattern[str] = re.compile(""),
        IQr_Mult: float = 1.5,
        IQR_Iter: int = 5,
        compression_ratio: float = 0.5,
    ):
        """Used to create mappings for words and builds a word vocabulary

        Args:
            pattern (Pattern[str]): regex pattern for matching strings
            IQr_Mult (float, optional):
                used for finding outliers in merged_pairs. Defaults to 1.5.
            IQR_Iter (int, optional):
                number of iterations to find outliers. Defaults to 5.
            compression_ratio (float, optional):
                how much the encoder compresses the text. Defaults to 0.5.
        """
        # public attributes
        self.match_pattern = (
            pattern if pattern.pattern != "" else self.__default_pattern()
        )
        self.IQr_Mult = IQr_Mult
        self.IQR_Iter = IQR_Iter
        self.storage_ratio = 1 - compression_ratio

        # private attributes
        self.__curr_tokenized = []  # Holds the current tokenized text
        self.__byte_pair_mapping = {}  # Maps byte-pairs to unique IDS
        self.__vocab = Counter()  # Frequency counter for bytes
        self.__merged_pairs = Counter()  # Holder for word pairs
        self.__doc_merged_pairs_freq = Counter()  # Holder for word freq
        self.__next_id = 256  # "Utf-8"
        self.__doc_count = 0
        self.__tok_track = set()

    def __default_pattern(self):
        # grab all possible puctuations
        escaped_punctuation = "".join(
            "\\" + char if char in ".^$*+?{}[]\\|()" else char
            for char in string.punctuation
        )

        pattern = re.compile(
            r"""
            # Case-insensitive matching
            (?i)
            # Dates in YYYY-MM-DD format
            \d{4}-\d{2}-\d{2}|
            # Enhanced alphanumeric with specific formats
            \b\w+_\d+[A-Z]?\b|
            # Specific keywords
            \b(?:has|value|is|empty|true|false|active\scontract|closed\scontract)\b|
            # Uppercase identifiers including numbers and underscores, not preceded by whitespace
            \b[A-Z0-9_]+(?<!\s)\b|
            # Match words with optional trailing punctuation
            \b\p{L}+(?:'\p{L}+)*\b(?:[{}]+)?|
            # Match whole numbers with optional trailing punctuation
            \b\p{N}+\b(?:[{}]+)?|
            # Monetary values, including optional $
            \$?\d+(?:,\d{3})*(?:\.\d+)?(?:[KMBT])?\b|
            # Match any character not a space, letter, or number
            [^\s\p{L}\p{N}]+|
            # Spaces
            \s+(?!\S)|\s+|"""
            + rf"[{escaped_punctuation}]+|"
            + "",
            re.VERSION1 | re.VERBOSE | re.IGNORECASE,
        )

        return pattern

    def get_current_tokenized(self):
        return self.__curr_tokenized

    def get_vocabulary(self):
        return self.__vocab

    def get_vocab_length(self):
        # this length is used as information in the embedding phase.
        return len(self.__vocab)

    def encode_text(self, text: str, verbose: bool = False) -> List[int]:
        """Converts characters in the strings to byters and
            compresses them with known popular merged pairs

        Args:
            text (str): text to turn to numbers

        Returns:
            List[int]: Tokenized and encoded text
        """
        tok_list = self.__byte_tokenizer(text)

        if verbose:
            msg = f"Length of Tokens {len(tok_list)}"
            print(f"Pre Merge: \n{tok_list}\n{msg}\n{'=' * len(msg)}\n")

        tok_list = self.__merge_known_pairs(tok_list)
        if verbose:
            msg = f"Length of Tokens {len(tok_list)}"
            print(f"Post Merge: \n{tok_list}\n{msg}\n{'=' * len(msg)}\n")

        return tok_list

    def decode(self, token_list: List[int]) -> str:
        """
        Decodes a list of tokens into the original text.

        Args:
            token_list (List[int]): The list of tokens to decode.

        Returns:
            str: The decoded original text.
        """

        # __byte_pair_mapping stores [pair -> token] we need the reverse
        reverse_map = {v: k for k, v in self.__byte_pair_mapping.items()}

        # decode each token to original character
        byte_array = []
        for token in token_list:
            byte_array.extend(self.__token_decoder(token, reverse_map))

        return bytes(byte_array).decode("utf-8")

    def __token_decoder(
        self, token: int, reverse_bpm: Dict[int, Tuple[int, int]]
    ) -> List[int]:
        """Breaks down tokens to encoded range

        Args:
            token (int): current token to decode
            reverse_bpm (Dict[int,Tuple[int, int]]):
                dictionary that maps token -> merged pair

        Returns:
            List[int]: decoded list
        """
        if token <= 255:
            return [token]

        # The token gives the pair
        pair_rep = reverse_bpm.get(token)
        if not pair_rep:
            raise ValueError(
                f"Token {token} not found in reverse map.\
                Check byte_pair_mapping"
            )

        decoded = []
        # recursive call to decode till we get to where we can call utf-8
        for val in pair_rep:
            decoded.extend(self.__token_decoder(val, reverse_bpm))

        return decoded

    def __byte_tokenizer(self, text: str) -> List[int]:
        """Converts string to list of tokens

        Args:
            text (str): text to convert

        Returns:
            list[int]: List of tokens
        """
        line_matches = self.match_pattern.findall(text)
        tok_list = []
        # update current encoded line
        for word in line_matches:
            tok_list.extend(word.encode("utf-8"))

        return tok_list

    def __merge_known_pairs(self, tok_list: List[int]) -> List[int]:
        """Scans and swaps already found pairs

        Args:
            tok_list (list[int]): list of tokens.
        Returns:
            List[int]: New tokenized array.
        """
        if not self.__byte_pair_mapping:
            return tok_list

        while True:
            new_tok_list = []
            new_merge = False
            idx = 0

            while idx < len(tok_list):
                # Check if we're at a point where a pair can be checked
                if idx < len(tok_list) - 1:
                    pair = (tok_list[idx], tok_list[idx + 1])

                    if pair in self.__byte_pair_mapping:
                        new_tok_list.append(self.__byte_pair_mapping[pair])
                        idx += 2  # Skip the next index as it's been merged
                        new_merge = True
                        continue

                new_tok_list.append(tok_list[idx])
                idx += 1

            tok_list = new_tok_list

            # Exit the while loop if no new merges occurred in this pass
            if not new_merge:
                break

        return tok_list

    def __update_vocab(self, tok_list: List[int]):
        # at this stage tok_list has the latest mappings
        curr_pair_track = set()

        for idx in range(len(tok_list)):
            current_token = tok_list[idx]
            pair = (current_token,)
            # Update the count for the current token
            if pair not in self.__tok_track:
                curr_pair_track.add(pair)
                self.__vocab[current_token] += 1

            # Update the count for the pair (current and next token)
            if idx < len(tok_list) - 1:
                next_token = tok_list[idx + 1]
                token_pair = (current_token, next_token)

                if token_pair not in self.__tok_track:
                    # updating stuff properly
                    self.__merged_pairs[token_pair] += 1
                    if token_pair not in curr_pair_track:
                        self.__doc_merged_pairs_freq[token_pair] += 1

                    # Tracking document count
                    curr_pair_track.add(token_pair)

        # update __tok_track
        for pair in curr_pair_track:
            self.__tok_track.add(pair)

    def process(self, text: str, verbose: bool = False):
        """Process for compressing

        Args:
            text (str): text to compress
            verbose (bool, optional):
                prints out run analysis. Defaults to False.
        """
        self.__doc_count += 1

        tok_list = self.__byte_tokenizer(text)
        original_size = len(tok_list)

        new_list = self.__merge_known_pairs(tok_list)
        curr_size = len(new_list)

        self.__tok_track.clear()

        # repeat process till we hit the compression ratio
        while curr_size > self.storage_ratio * original_size:

            self.__update_vocab(new_list)

            self.__update_mappings()

            new_list = self.__merge_known_pairs(new_list)

            if curr_size == len(new_list):
                break

            curr_size = len(new_list)

        # complete the calculation
        if verbose and original_size != len(new_list):
            self.run_analysis(tok_list, new_list)

        # place holder if necessary
        self.__curr_tokenized = new_list

    def __vocab_tf_idf(self):
        tf_idf_track = {}
        total_pairs = len(self.__merged_pairs)
        for pair, count in self.__merged_pairs.items():
            freq = self.__doc_merged_pairs_freq[pair]

            idf = math.log10(self.__doc_count / freq)
            tf_idf_track[pair] = (count / total_pairs) * idf

        # use to trim the document
        return tf_idf_track

    def __update_mappings(self):
        # here is where we add new to the list here is where I update byte_pair
        if not self.__vocab:
            return

        tf_idf_dict = self.__vocab_tf_idf()
        freqs = np.array(list(tf_idf_dict.values()))

        res = self.iterative_filtering(
            freqs, multiplier=self.IQr_Mult, max_iter=self.IQR_Iter
        )

        # updating merged pair information
        if res[0] > 0 or res[1] > 0:
            self.__update_with_iqr(res, tf_idf_dict)
        # else:
        # self.__upper_per_update(np.quantile(freqs, 0.75), tf_idf_dict)

    def __update_with_iqr(
        self, bounds: Tuple[float, float], tf_idf_dict: Dict[Tuple[int, int], float]
    ):
        """Uses IQR value to filter:
            - irrelvant information if less than lower bound
            - pairs that should be included in the vocabulary

        Args:
            bounds (List[float, float]):
                lower and upper bounds for filtering process
            tf_idf_dict: Dict[Tuple[int, int], float]
                TF_IDF for each merged pair
        """
        # gets median of main vocabulary. This is used as a threshold
        freqs = np.array(list(self.__vocab.values()))
        min_threshold = np.quantile(freqs, 0.5)

        for key, tf_idf in tf_idf_dict.items():
            if bounds[1] > 0 and tf_idf > bounds[1]:
                # adding to vocab list
                if (
                    self.__vocab[key[0]] >= min_threshold
                    and self.__vocab[key[1]] >= min_threshold
                ):
                    self.__byte_pair_mapping[key] = self.__next_id
                    self.__vocab[self.__next_id] = 0
                    self.__next_id += 1
                    del self.__merged_pairs[key]

            elif bounds[0] > 0 and tf_idf <= bounds[0]:
                del self.__merged_pairs[key]

    def __upper_per_update(self, val: float, tf_idf_dict: Dict[Tuple[int, int], float]):
        """Gets all matched pairs with counts > val

        Args:
            val (float): Threshold for extracting
            tf_idf_dict: Dict[Tuple[int, int], float]
                TF_IDF for each merged pair
        """

        # gets median of main vocabulary. This is used as a threshold
        freqs = np.array(list(self.__vocab.values()))
        min_threshold = np.quantile(freqs, 0.5)

        for key, tf_idf in tf_idf_dict.items():
            if tf_idf > val:
                # adding to vocab list
                if (
                    self.__vocab[key[0]] >= min_threshold
                    and self.__vocab[key[1]] >= min_threshold
                ):
                    self.__byte_pair_mapping[key] = self.__next_id
                    self.__vocab[self.__next_id] = 0
                    self.__next_id += 1
                    del self.__merged_pairs[key]

    @staticmethod
    def iterative_filtering(data: list, multiplier=1.5, max_iter=10):
        # using IQR to filter data
        lower_bound = -1
        upper_bound = -1

        for _ in range(max_iter):
            Q1 = np.quantile(data, 0.25)
            Q3 = np.quantile(data, 0.75)
            IQR = Q3 - Q1

            lower_bound_q = Q1 - multiplier * IQR
            upper_bound_q = Q3 + multiplier * IQR

            # Filter data
            filtered_data = [x for x in data if lower_bound_q <= x <= upper_bound_q]

            if len(filtered_data) == len(data):
                lower_bound = lower_bound_q
                upper_bound = upper_bound_q
                break

            data = filtered_data

        return [lower_bound, upper_bound]

    def run_analysis(
        self, original_list: List[int], new_list: List[int], disp: bool = True
    ):
        original_unique = np.unique(np.array(original_list))
        new_unique = np.unique(np.array(new_list))
        compression_ratio = (
            len(new_list) / len(original_list)
            if len(original_list) != len(new_list)
            else 0
        )

        # Prepare the data to determine the width
        rows = [
            ("Length of original list", len(original_list)),
            ("Number of unique tokens in original", len(original_unique)),
            ("Length of compressed list", len(new_list)),
            ("Number of unique tokens in compressed", len(new_unique)),
            ("Final compression ratio", f"{compression_ratio:.3f} "),
        ]

        # Find the longest row label
        longest_label_length = max(len(label) for label, _ in rows)
        label_value_gap = 2  # Space between label and value

        # Calculate the width for the output
        value_width = max(len(str(value)) for _, value in rows)
        width = longest_label_length + value_width + label_value_gap

        top_line = "+" + "-" * (width + 2) + "+"

        # Start building the output
        output = top_line + "\n"
        header = "Analysis Results"
        output += f"| {header:^{width}} |\n"
        output += top_line + "\n"

        # Append data rows to the output string
        for label, value in rows:
            output += f"| {label:<{longest_label_length}}: {value:>{value_width}} |\n"

        output += top_line

        # Print the complete output
        if disp:
            print(output)
            return " "

        return output

    def save_state(self, location: str):
        directory = os.path.join(location, "BPE_Info")
        if not os.path.exists(directory):
            os.makedirs(directory)

        filepath = os.path.join(directory, "bpe_state.pkl")
        with open(filepath, "wb") as file:
            pickle.dump(self.__dict__, file)

    def load_state(self, location: str):
        filepath = os.path.join(location, "BPE_info", "bpe_state.pkl")
        if os.path.exists(filepath):
            with open(filepath, "rb") as file:
                state = pickle.load(file)
                self.__dict__.update(state)
        else:
            print("No saved state found at the specified location.")

    def test(self, txt: str = ""):
        if not txt:
            txt = "On 2024-03-13, the project 'GeoData_Analysis_2024' officially"
            txt += " kicked off.The team had previously discussed several key points,"
            txt += (
                " emphasizing the importance of accuracy and efficiency. As of today,"
            )
            txt += (
                " there have been 152 issues logged, with 47 marked as 'resolved' and"
            )
            txt += " the remaining awaiting review. Interestingly, the budget allocated for"
            txt += (
                " this phase is 3,450,000.2456, which is under it's $3.5M cap suggested"
            )
            txt += " in the initial proposal."

        out_msg = f"\nOrginal_text: \n{txt}\n"
        out_msg += "\nRunning script to encode\n"
        self.process(txt, False)

        out_msg += "\nEncoding the Text\n"
        orig_list = self.__byte_tokenizer(txt)
        msg = f"Length of Tokens {len(orig_list)}"
        out_msg += f"Pre Merge: \n{orig_list}\n{msg}\n{'=' * len(msg)}\n"

        tok_list = self.__merge_known_pairs(orig_list)
        msg = f"Length of Tokens {len(tok_list)}"
        out_msg += f"Post Merge: \n{tok_list}\n{msg}\n{'=' * len(msg)}\n"

        out_msg += f"\n{self.run_analysis(orig_list, tok_list, False)}\n"

        out_msg += "\nTesting decoder\n"
        final_string = self.decode(tok_list)

        out_msg += f"Decoded String: \n{final_string}\n============== \n"
        out_msg += f"Matched Original?: {txt==final_string}"

        msg = f"Length of vocabulary {len(self.__vocab)}"
        out_msg += f"\n{msg}\n{'=' * len(msg)}\n"

        return out_msg


def main():
    operation = BytePairEncodeAlgo(
        pattern=re.compile(""), IQr_Mult=1.5, IQR_Iter=5, compression_ratio=0.5
    )

    print(operation.test())

    print("\n")
    txt = "In the midst of a sprawling desert, a hidden oasis thrived,"
    txt += " sheltered by towering dunes. At the heart of this verdant "
    txt += "sanctuary lay an ancient tree, its roots entwined with the very "
    txt += "essence of the land. Legends whispered of its creation, born from "
    txt += "the tears of a celestial being, granting it the power to sustain life "
    txt += "in the barren expanse. Travelers, drawn by tales of its mystical allure, "
    txt += "ventured across the arid wilderness, hoping to uncover its secrets. As the sun "
    # dipped below the horizon, casting a golden glow, the oasis revealed its true splendor, becoming a beacon of hope and serenity in the vast, unyielding desert."
    print(operation.test(txt))

    txt = "In the midst of a sprawling desert, a hidden oasis thrived,"
    txt += " sheltered by towering dunes. At the heart of this verdant "
    txt += "sanctuary lay an ancient tree, its roots entwined with the very "
    txt += "essence of the land. Legends whispered of its creation, born from "
    txt += "the tears of a celestial being, granting it the power to sustain life "
    txt += "in the barren expanse. Travelers, drawn by tales of its mystical allure, "
    txt += "ventured across the arid wilderness, hoping to uncover its secrets. As the sun "
    txt += "dipped below the horizon, casting a golden glow, the oasis revealed its true splendor, becoming a beacon of hope and serenity in the vast, unyielding desert."
    print(operation.test(txt))

    random_text = """
        Hello World! 12345
        Special characters: ~!@#$%^&*()
        Python is great for data analysis.
        'Let's use single quotes within a string.'
        "Double quotes can also be used."
        New line starts here.
        The quick brown fox jumps over the lazy dog.
        123 + 456 = 579
        Escape sequences: \t (tab), \n (new line)
        End of string.
        """
    print(operation.test(random_text))

    sample_text = """
        Once upon a time, in a land far, far away, there lived a coder named Alex.
        42 is the answer to life, the universe, and everything.
        Favorite symbols: @, #, $, %, ^, &, and *.
        Alex said, "Learning Python is fun!"
        Use \\ to escape a backslash, \n for newline, and \t for tab.
        Math in strings: 8 * 6 = 48
        Random characters: α, β, γ, δ, ε, ζ, η, θ
        End of sample text.
        """
    print(operation.test(sample_text))


if __name__ == "__main__":
    main()
