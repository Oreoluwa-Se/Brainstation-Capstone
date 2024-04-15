from .load_and_batch import MetaData
from collections import Counter
from dataclasses import dataclass, field
from nltk.corpus import words
from tqdm import tqdm
from typing import List, Pattern, Dict, Tuple, Any
import math
import nltk
import numpy as np
import os
import pickle
import regex as re
import string

nltk.download("words")

common_words = set(words.words())


# use dataclass here
@dataclass
class BpeArgs:
    pattern: Pattern[str] = None  # matching pattern
    # for interquartile range outlier measure
    IQR_Mult: float = 1.5
    # for better outlier prediction
    IQR_Iter: int = 5
    # checks for opportunities to include rare words that bring context
    doc_thresh: int = 100
    # controls the potential final context window after compressing
    target_context: int = -1
    # controls the total vocab size
    max_vocab_size: int = -1
    # Where to store or load information post
    store_loc: str = ""
    # List of special tokens to be included
    adhoc_tokens: List[str] = field(default_factory=list)
    # Special words that aren't key tags but should be included
    adhoc_words: List[str] = field(default_factory=list)


class TokCollector:
    """
    A class to collect and manage tokens, supporting operations like
    frequency counting, threshold updating, and bounds calculation using IQR.
    """

    def __init__(
        self, IQR_Mult: float = 1.5, IQR_Iter: int = 5, add_freq: bool = False
    ):
        self.IQR_Mult = IQR_Mult
        self.IQR_Iter = IQR_Iter
        self.vocab = Counter()
        self.mapping = {}
        self.__vthresh_updated = True
        self.min_vocab_threshold = float("-inf")

        self.freq = Counter() if add_freq else None
        self.__fthresh_updated = True if add_freq else False
        self.min_freq_threshold = float("-inf") if add_freq else None

    def add(self, new_token: Any):
        self.__vthresh_updated = False
        self.vocab[new_token] += 1

    def sub(self, token: Any, val: int = 1):
        self.__vthresh_updated = False
        if token in self.vocab:
            self.vocab[token] -= val

    def update_freq(self, new_token: Any):
        if self.freq is not None:
            self.__fthresh_updated = False
            self.freq[new_token] += 1

    @property
    def empty_vocab(self) -> bool:
        return len(self.vocab) == 0

    @property
    def empty_freq(self) -> bool:
        return len(self.freq) == 0 if self.freq is not None else True

    @property
    def vocab_len(self) -> int:
        return len(self.vocab)

    @property
    def freq_len(self) -> int:
        return len(self.freq) if self.freq is not None else 0

    def get_vocab_bounds(self) -> Tuple[float, float]:
        # Uses IQR to find outliers
        return self.iterative_filtering(
            np.array(list(self.vocab.values())), is_vocab=True
        )

    def update_vocab_thresh(self) -> None:
        if not self.__vthresh_updated:
            sorted_data = np.sort(np.array(list(self.vocab.values())))
            self.min_vocab_threshold = self.median_ext(sorted_data)
            self.__vthresh_updated = True

    def get_freq_bounds(self) -> Tuple[float, float]:
        if self.freq is not None:
            return self.iterative_filtering(
                np.array(list(self.freq.values())), is_vocab=False
            )
        return [-1, -1]

    def update_freq_thresh(self) -> None:
        if self.freq is not None and not self.__fthresh_updated:
            sorted_data = np.sort(np.array(list(self.freq.values())))
            self.min_freq_threshold = self.median_ext(sorted_data)
            self.__fthresh_updated = True

    def median_ext(self, sorted_data: np.array):
        n = len(sorted_data)
        return (
            (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
            if n % 2 == 0
            else sorted_data[n // 2]
        )

    def iterative_filtering(
        self, data: np.array, is_vocab: bool = True
    ) -> Tuple[float, float]:

        sorted_data = np.sort(data)
        # check if any discriminative numbers
        if sorted_data[0] == sorted_data[-1]:
            return [-1, -1]

        # update the minimum threshold
        if is_vocab:
            self.min_vocab_threshold = self.median_ext(sorted_data)
            self.__vthresh_updated = True
        elif self.freq is not None:
            self.min_freq_threshold = self.median_ext(sorted_data)
            self.__fthresh_updated = True

        for _ in range(self.IQR_Iter):
            n = len(sorted_data)

            Q1, Q3 = np.percentile(sorted_data, [25, 75])

            IQR = Q3 - Q1
            lower_bound_q = Q1 - self.IQR_Mult * IQR
            upper_bound_q = Q3 + self.IQR_Mult * IQR

            filtered_data = sorted_data[
                (sorted_data >= lower_bound_q) & (sorted_data <= upper_bound_q)
            ]

            if len(filtered_data) == n:
                break

            sorted_data = filtered_data

        return lower_bound_q, upper_bound_q

    def update_mappings(
        self, item: Tuple[Tuple[int, int], int], token_id: int, freq_count: int = 0
    ):
        self.mapping[item[0]] = token_id
        self.vocab[token_id] = item[1]

        if self.freq is not None:
            self.freq[item[0]] = freq_count

    def remove_presence(self, key: Tuple[int, int], alt=None):
        # handles removal of key and prunes the list as-well
        if key in self.vocab:
            del self.vocab[key]

        # Remove the current key we inserted previously
        if self.freq and key in self.freq:
            del self.freq[key]

        c_vocab = self.vocab

        for c_pair in list(c_vocab.keys()):
            if c_pair[0] == key[1] or c_pair[1] == key[0]:
                del self.vocab[c_pair]
                del self.freq[c_pair]

                if alt:
                    del alt[c_pair]

    def tf_idf_dict(self, doc_size: int, total_words: int) -> Dict[Any, float]:
        if self.freq:
            tf_idf_track = {}
            for pair, count in self.freq.items():
                if count > 0:
                    idf = math.log10(doc_size / count)
                    tf_idf = (count / total_words) * idf
                    if tf_idf > 0:
                        tf_idf_track[pair] = tf_idf
            return tf_idf_track
        return {}

    def clear(self):
        self.vocab.clear()
        self.mapping.clear()

        if self.freq is not None:
            self.freq.clear()


#######################################################################
class Encoder:
    # Uses BPE to encode text from a polars table
    __base_chars = 255
    __next_id = __base_chars

    def __init__(self, args: BpeArgs = None) -> None:
        if args is None:
            print("Empty instantiation. Ensure to load from pickle file location")
            return

        """Types of tokens"""
        # special tokens [token: count]
        self.__special = TokCollector(args.IQR_Mult, args.IQR_Iter, add_freq=True)
        self.__init_special(args.adhoc_tokens)
        self.__update_special_toks(args.adhoc_words)

        # base single ascii characters [token: count]
        self.__base = TokCollector(args.IQR_Mult, args.IQR_Iter, add_freq=True)
        # base paired characters [token: count]
        self.__paired = TokCollector(args.IQR_Mult, args.IQR_Iter, add_freq=True)
        self.__init_paired()

        # for commmmon words
        self.__common = TokCollector(args.IQR_Mult, args.IQR_Iter, add_freq=True)

        self.__target_context = args.target_context
        self.__store_loc = args.store_loc
        self.__max_vocab_size = args.max_vocab_size
        # Trackers
        self.__doc_count = 0
        self.__total_words = 0
        self.__run_pair_track = set()

        self.__pattern = (
            args.pattern
            if args.pattern is not None
            else self.__default_pattern(self.__special.mapping)
        )

        self.__loaded = False

    def __update_common_vocab(self, word):
        if word not in self.__common.mapping:
            self.__common.mapping[word] = self.__next_id
            self.__common.vocab[self.__next_id] = 0
            self.__next_id += 1

    def __update_special_toks(self, special_list: List[str]) -> None:
        if len(special_list) == 0:
            return

        for spec_tok in special_list:
            # create [char -> tokens]
            if spec_tok not in self.__special.mapping:
                self.__special.mapping[spec_tok] = self.__next_id
                self.__special.vocab[self.__next_id] = 0
                self.__next_id += 1

    def add_special_words(self, words: List[str]):
        self.__update_special_toks(words)

    def get_meta_tokens(self):

        return {
            key: self.__special.mapping[key]
            for key in ["<|AMOUNT|>", "<|NUM|>", "<|DATE|>"]
        }

    def __init_special(self, toks: List[str]):
        base = ["<|AMOUNT|>", "<|NUM|>", "<|DATE|>", "<|SOW|>", "<|EOW|>", "<|PAD|>"]

        # ammend new list to base
        if len(toks) > 0:
            collect = set(base)
            [collect.add(f"<|{val}|>") for val in toks]
            base = list(collect)

        # updates special mapping
        self.__update_special_toks(base)

    @staticmethod
    def __default_pattern(key_tags: Dict[str, int]):
        """Automatically inclueds special tokens into matched patterns

        Args:
            special_tokens (List[str]): word patterns to include
        """
        special_word_pattern = "|".join(
            re.escape(word) for word in list(key_tags.keys())
        )
        base_pattern = r'"<\|[^|]*\|>"|\b\w+\b(?:[!?.,;:]?)|[\w]+[_][\w]+|(?<!\s)[A-Z0-9_]+|\s+|[^a-zA-Z0-9\s]'

        pattern = f"{special_word_pattern}|{base_pattern}"
        return re.compile(pattern, re.IGNORECASE)

    def __init_paired(self) -> None:
        # all possible begining of word and end of word characters
        characters = string.ascii_letters + string.digits

        for char in characters:
            idx = list(char.encode("utf-8"))[0]
            mapped_char = (self.__special.mapping["<|SOW|>"], idx)

            self.__paired.mapping[mapped_char] = self.__next_id
            self.__paired.vocab[self.__next_id] = 0
            self.__next_id += 1

            mapped_char = (idx, self.__special.mapping["<|EOW|>"])
            self.__paired.mapping[mapped_char] = self.__next_id
            self.__paired.vocab[self.__next_id] = 0
            self.__next_id += 1

    def update_context_length(self, context_length: int):
        self.__target_context = context_length

    def run_analysis(
        self, original_list: List[int], new_list: List[int], disp: bool = True
    ):
        original_unique = np.unique(np.array(original_list))
        new_unique = np.unique(np.array(new_list))
        compression_ratio = (
            1 - (len(new_list) / len(original_list))
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
            ("Length of vocabulary", self.vocab_size),
        ]

        if not self.__loaded:
            rows += [
                ("Length of vocabulary", self.vocab_size),
                ("Total documents seen so far", self.__doc_count),
                ("Total words seen so far", self.__total_words),
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

    @property
    def vocab_size(self):
        # returns the total length
        # __base_chars represent base Ascii characters
        return (
            self.__base_chars
            + self.__paired.vocab_len
            + self.__special.vocab_len
            + self.__common.vocab_len
        )

    def word_to_code(self, text: str):
        if text in self.__special.mapping:
            return [self.__special.mapping[text]]

        if text in self.__common.mapping:
            return [self.__common.mapping[text]]

        return list(text.encode("utf-8"))

    def word_to_tokens(self, text: str, verbose: bool = False) -> List[int]:
        # these we encode simply
        skip_pattern = re.compile(
            rf"[{re.escape(string.punctuation)}$\*\+?\{{\}}\[\]\\|()]"
        )
        tokens, verbose_str = [], []
        for word in self.__pattern.findall(text):
            if word.isspace():
                continue

            if (
                len(list(word)) == 1
                or word in self.__special.mapping
                or skip_pattern.match(word) is not None
            ):
                tokens.extend(self.word_to_code(word))

                if verbose:
                    verbose_str.append(word)
                continue

            inner_split = word.split(" ")
            to_join, to_join_a = [], []

            for single_word in inner_split:
                # tokenize common words seperately
                if single_word.lower() in common_words:
                    self.__update_common_vocab(single_word)
                    to_join.append(self.__common.mapping[single_word])
                    to_join_a.append(single_word)
                    continue

                # tokenize
                to_join.extend(self.word_to_code("<|SOW|>"))
                to_join.extend(self.word_to_code(single_word))
                to_join.extend(self.word_to_code("<|EOW|>"))

                # for verbose purposes
                to_join_a.extend(["<|SOW|>" + single_word + "<|EOW|>"])

            # extending tokens
            tokens.extend(to_join)
            if verbose:
                verbose_str.extend(to_join_a)

        if verbose:
            print(verbose_str)

        return tokens

    def __merge_known_pairs(self, tok_list: List[int]) -> List[int]:
        """Keeps iterating until we've reduced the list to the smallest based on paired vocab

        Args:
            tok_list (List[int]): New tokens to compress

        Returns:
            List[int]: compressed tokens
        """
        if self.__paired.empty_vocab:
            return tok_list

        # iterate till no more pairs
        while True:
            new_list = []
            changed = False
            idx = 0

            while idx < len(tok_list):
                if idx + 1 < len(tok_list):
                    pair = (tok_list[idx], tok_list[idx + 1])

                    token = self.__paired.mapping.get(pair, None)
                    if token is not None:
                        new_list.append(token)
                        changed = True
                        idx += 2
                    else:
                        new_list.append(tok_list[idx])
                        idx += 1

                else:
                    # append last if not paired
                    new_list.append(tok_list[idx])
                    idx += 1

            # break if no changes
            if not changed:
                break

            # update
            tok_list = new_list

        return tok_list

    def __update_indv_tok_freq(self, token: int):
        # update the token frequency
        if token <= 255:
            target_vocab = self.__base
        elif token in self.__special.vocab:
            target_vocab = self.__special
        elif token in self.__common.vocab:
            target_vocab = self.__common
        else:
            target_vocab = self.__paired

        target_vocab.add(token)

    def __update_doc_tok_freq(self, token: int):
        # update the document frequency
        if token <= 255:
            target_vocab = self.__base
        elif token in self.__special.vocab:
            target_vocab = self.__special
        elif token in self.__common.vocab:
            target_vocab = self.__common
        else:
            target_vocab = self.__paired

        target_vocab.update_freq(token)

    def find_most_frequent_pairs(self, lst):
        # Count the frequency of each adjacent pair, skipping those with elements in special_vocab
        tok_collector = TokCollector()

        for i in range(len(lst) - 1):
            if (
                lst[i] not in self.__special.vocab
                and lst[i + 1] not in self.__special.vocab
            ):
                tok_collector.add((lst[i], lst[i + 1]))

        if tok_collector.empty_vocab:
            return [], None

        # Here we use TokCollector's functionality to get bounds and decide on outliers
        _, upper_bound = tok_collector.get_vocab_bounds()

        if upper_bound <= 1.0:
            return [], None

        max_freq = max(tok_collector.vocab.values())

        # Check if max_freq is an outlier
        if max_freq > upper_bound:
            most_frequent_pairs = [
                pair for pair, count in tok_collector.vocab.items() if count == max_freq
            ]
        else:
            most_frequent_pairs = []

        return most_frequent_pairs, max_freq

    def __update_vocab(self, tok_list: List[int]) -> None:
        """
        Updates individual frequency and number of documents token appears
        Also starts process for forming new pairs

        Args:
            tok_list (List[int]): List of tokens to be updated
        """
        loop_track = set()

        for idx in range(len(tok_list)):
            curr_tok, s_pair = tok_list[idx], (tok_list[idx],)

            if s_pair not in self.__run_pair_track:
                self.__update_indv_tok_freq(curr_tok)

                if s_pair not in loop_track:
                    self.__update_doc_tok_freq(curr_tok)

                loop_track.add(s_pair)

        # Update run pair track to be used in next loop
        self.__run_pair_track.update(loop_track)

    def __update_info(self, new_info: List[Tuple[int, int]], count: int = 1):
        for key in new_info:
            self.__paired.update_mappings((key, count), self.__next_id, 0)
            self.__next_id += 1

    def get_count(self, token: int) -> int:
        # Return count of token directly for base vocab
        if token <= 255:
            return self.__base.vocab.get(token, 0)

        # Iterate through other vocabularies to find the token count
        for vocab in [self.__special.vocab, self.__paired.vocab, self.__common.vocab]:
            count = vocab.get(token)
            if count is not None:
                return count

        return 0

    def __bpe_update_scheme(self, opts: List[Tuple[int, int]], max_count):
        # Calculate root_counts and filter by max_count in one step, then sort
        root_counts = [
            (pair, self.get_count(pair[0]) + self.get_count(pair[1])) for pair in opts
        ]
        sorted_pairs = sorted(root_counts, key=lambda x: x[1], reverse=True)

        # Filter by unique counts
        unique_pairs = []
        unique_toks = set()
        for pair, _ in sorted_pairs:
            if pair[0] not in unique_toks and pair[1] not in unique_toks:
                unique_pairs.append(pair)
                unique_toks.update(pair)

        # Now we update the information real-time
        self.__update_info(unique_pairs, max_count)

    def get_token(self, pair: Tuple[int, int]) -> int:
        if len(pair) == 1:
            return pair[0]
        else:
            tok = self.__paired.mapping.get(pair, None)
            if tok is not None:
                return tok

        return -1

    def compute_score(self, tok: int):
        # given a token we compute the tf-idf and likelihood
        indv_freq, doc_freq = -1.0, -1.0

        if tok <= 255:
            indv_freq = self.__base.vocab.get(tok, -1.0)
            doc_freq = self.__base.freq.get(tok, -1.0)

        elif tok in self.__special.vocab:
            indv_freq = self.__special.vocab.get(tok, -1.0)
            doc_freq = self.__special.freq.get(tok, -1.0)

        else:
            indv_freq = self.__paired.vocab.get(tok, -1.0)
            doc_freq = self.__paired.freq.get(tok, -1.0)

        if indv_freq == -1.0 or doc_freq == -1.0:
            return [indv_freq, doc_freq]

        prob = indv_freq / self.__total_words

        idf = math.log10(self.__doc_count / doc_freq)
        tf_idf = prob * idf

        return [prob, tf_idf]

    def process(self, text: str, encode_only: bool = False, verbose: bool = False):
        tok_list = self.word_to_tokens(text, False)
        new_list = self.__merge_known_pairs(tok_list)

        if encode_only:
            if verbose:
                self.run_analysis(tok_list, new_list)

            return new_list

        curr_size = len(new_list)

        self.__doc_count += 1
        self.__total_words += curr_size
        # These help track results per run
        self.__run_pair_track.clear()

        while True:
            # if merged version already fits the context window
            if self.__target_context > 0 and curr_size <= self.__target_context:
                break

            self.__update_vocab(new_list)
            opts, max_count = self.find_most_frequent_pairs(new_list)
            if len(opts) == 0 or max_count == 1:
                break

            # print(opts)
            self.__bpe_update_scheme(opts, max_count)
            new_list = self.__merge_known_pairs(new_list)

            # no further improvements can be made
            if curr_size == len(new_list):
                break

            curr_size = len(new_list)

        if verbose:
            self.run_analysis(tok_list, new_list)

    def encode_text(self, text: str, verbose: bool = False):
        encoded = self.process(text, encode_only=True, verbose=verbose)

        if len(encoded) < self.__target_context:
            num_add = self.__target_context - len(encoded)
            for _ in range(0, num_add):
                encoded.append(self.__special.mapping["<|PAD|>"])

        return encoded

    def save_state(self, location: str):
        if location.endswith(".pkl"):
            filepath = location
            directory = os.path.dirname(filepath)
        else:
            directory = os.path.join(location, "BPE_Info")
            filepath = os.path.join(directory, "bpe_state.pkl")

        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(filepath, "wb") as file:
            pickle.dump(self.__dict__, file)

    def load_state(self, location: str):
        if location.endswith(".pkl"):
            filepath = location
        else:
            filepath = os.path.join(location, "BPE_Info", "bpe_state.pkl")

        if os.path.exists(filepath):
            with open(filepath, "rb") as file:
                state = pickle.load(file)
                self.__dict__.update(state)

            self.__loaded = True
        else:
            print("No saved state found at the specified location.")

    def compress_files(
        self,
        metas: List[MetaData],
        save_or_update_every: int = 20,
    ):
        total_lines = len(metas)
        prev_per = 0

        with tqdm(total=total_lines, desc="Overall Progress") as pbar:
            for meta in metas:
                pbar.update(1)

                # greedy processing
                self.process(meta.text, verbose=False)

                curr_per = int(100 * (pbar.n / total_lines))
                if curr_per % save_or_update_every == 0 and curr_per != prev_per:
                    print(f"Length of vocab: {self.vocab_size}")
                    self.save_state(self.__store_loc)
                    prev_per = curr_per

                # Check if maximum vocabulary size is reached
                if (
                    self.__max_vocab_size != -1
                    and self.vocab_size >= self.__max_vocab_size
                ):
                    print(f"Maximum Vocabulary Size: {self.vocab_size} reached.")
                    break

            # Save state after each file is processed
            self.save_state(self.__store_loc)

            # Final save state at the end of processing
            self.save_state(self.__store_loc)
            pbar.close()
