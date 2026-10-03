# Copyright (C) 2019-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Lexicon-free prefix beam search that sums alternative CTC alignments."""

import math

import numpy as np


def _logadd(left, right):
    high, low = max(left, right), min(left, right)
    return high if low == -math.inf else high + math.log1p(math.exp(low - high))


def prefix_beam_decode(log_probabilities, codec, beam_width=5, token_topk=8):
    """Approximate the most probable transcript, respecting blanks and repeated letters.

    Returns:
        the best prefix after summing blank/nonblank alignment probabilities

    Raises:
        ValueError: if beam width or token count is not positive
    """
    if beam_width <= 0 or token_topk <= 0:
        raise ValueError("beam_width and token_topk must be positive")
    beams = {(): (0.0, -math.inf)}
    for frame in log_probabilities:
        top = np.argsort(frame[1:])[-token_topk:] + 1
        candidates = {}
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
