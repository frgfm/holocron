# Copyright (C) 2019-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Alphabet encoding and lexicon-free CTC transcript decoding."""

import math
from collections.abc import Iterable

import numpy as np
import torch
from torch import Tensor

__all__ = ["CTCCodec", "prefix_beam_decode"]


class CTCCodec:
    """Map an ordered alphabet to CTC labels, reserving index zero for blank.

    Args:
        alphabet: unique printable Unicode characters; ordinary space is allowed

    Raises:
        ValueError: if the alphabet is empty, duplicated, or contains unsupported whitespace
    """

    def __init__(self, alphabet: str) -> None:
        if not alphabet or len(set(alphabet)) != len(alphabet):
            raise ValueError("alphabet must be nonempty with unique characters")
        if any(not char.isprintable() or (char.isspace() and char != " ") for char in alphabet):
            raise ValueError("only printable characters and ordinary spaces are supported")
        if not alphabet.replace(" ", ""):
            raise ValueError("alphabet must contain a visible character")
        self.alphabet = alphabet
        self.indices = {char: index + 1 for index, char in enumerate(alphabet)}

    def encode(self, text: str) -> Tensor:
        """Encode text without inserting CTC blanks.

        Args:
            text: characters drawn from this codec's alphabet

        Returns:
            one-dimensional CPU int64 tensor of target labels

        Raises:
            ValueError: if a character is absent from the alphabet
        """
        if any(char not in self.indices for char in text):
            raise ValueError("text contains a character outside the alphabet")
        return torch.tensor([self.indices[char] for char in text], dtype=torch.long)

    def decode(self, indices: Iterable[int]) -> str:
        """Collapse a greedy CTC alignment, preserving repeats separated by blanks.

        Args:
            indices: consecutive frame labels, truncated to the true sequence length

        Returns:
            decoded transcript, including predicted spaces and punctuation

        Raises:
            ValueError: if an index is outside the blank/alphabet label range
        """
        result = []
        previous = 0
        for index in indices:
            if not 0 <= index <= len(self.alphabet):
                raise ValueError("CTC indices must lie between zero and the alphabet size")
            if index and index != previous:
                result.append(self.alphabet[index - 1])
            previous = index
        return "".join(result)


def _logadd(left: float, right: float) -> float:
    high, low = max(left, right), min(left, right)
    return high if low == -math.inf else high + math.log1p(math.exp(low - high))


def prefix_beam_decode(log_probabilities: np.ndarray, codec: CTCCodec, beam_width: int = 5, token_topk: int = 8) -> str:
    """Approximate the most probable transcript, respecting blanks and repeated letters.

    Args:
        log_probabilities: frame log probabilities of shape (T, alphabet size + 1)
        codec: alphabet whose blank index is zero
        beam_width: number of transcript prefixes retained per frame
        token_topk: maximum nonblank frame labels considered; blank is always included

    Returns:
        the best prefix after summing blank/nonblank alignment probabilities

    Raises:
        ValueError: if beam width, token count, or probability shape is invalid
    """
    if beam_width <= 0 or token_topk <= 0:
        raise ValueError("beam_width and token_topk must be positive")
    if log_probabilities.ndim != 2 or log_probabilities.shape[1] != len(codec.alphabet) + 1:
        raise ValueError("log_probabilities must have shape (T, alphabet size + 1)")
    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, -math.inf)}
    for frame in log_probabilities:
        top = np.argsort(frame[1:])[-token_topk:] + 1
        candidates: dict[tuple[int, ...], tuple[float, float]] = {}
        for prefix, (blank, nonblank) in beams.items():
            total = _logadd(blank, nonblank)
            old_blank, old_nonblank = candidates.get(prefix, (-math.inf, -math.inf))
            candidates[prefix] = (_logadd(old_blank, total + float(frame[0])), old_nonblank)
            for raw_token in top:
                token = int(raw_token)
                score = float(frame[token])
                repeated = bool(prefix) and prefix[-1] == token
                if repeated:
                    old_blank, old_nonblank = candidates[prefix]
                    candidates[prefix] = (old_blank, _logadd(old_nonblank, nonblank + score))
                extended = (*prefix, token)
                old_blank, old_nonblank = candidates.get(extended, (-math.inf, -math.inf))
                candidates[extended] = (old_blank, _logadd(old_nonblank, (blank if repeated else total) + score))
        beams = dict(sorted(candidates.items(), key=lambda pair: _logadd(*pair[1]), reverse=True)[:beam_width])
    prefix = max(beams, key=lambda prefix: _logadd(*beams[prefix]))
    return "".join(codec.alphabet[index - 1] for index in prefix)
