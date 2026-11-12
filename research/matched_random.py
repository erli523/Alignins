"""Layer- and size-matched null sets; independent of all training RNGs."""
import numpy as np


def jaccard(left, right):
    intersection = np.intersect1d(left, right, assume_unique=True).size
    union = left.size + right.size - intersection
    return intersection / union if union else float('nan')


class LayerMatchedRandom:
    def __init__(self, layout, repeats=30, seed=20261111):
        if repeats < 2 or seed < 0:
            raise ValueError('random repeats must be >=2 and seed must be nonnegative')
        self.layout = layout
        self.repeats = repeats
        self.seed = seed
        self.previous = {}

    def sample(self, client_id, round_number, raw, promoted):
        """Uniform subsets, without replacement, from each layer minus raw Top-K.

        Sampling compressed eligible ranks avoids constructing million-entry pools.
        Each replicate and global round has its own reproducible RNG stream.
        """
        result = np.empty((self.repeats, promoted.size), dtype=np.int32)
        generators = [np.random.default_rng(np.random.SeedSequence(
            [self.seed, int(client_id), int(round_number), repeat]))
            for repeat in range(self.repeats)]
        cursor = 0
        for layer in self.layout:
            start = layer['analysis_start']
            end = start + layer['numel']
            p_lo, p_hi = np.searchsorted(promoted, [start, end])
            count = int(p_hi - p_lo)
            if not count:
                continue
            r_lo, r_hi = np.searchsorted(raw, [start, end])
            blocked = raw[r_lo:r_hi].astype(np.int64) - start
            available = layer['numel'] - blocked.size
            if count > available:
                raise ValueError('Promoted count exceeds eligible layer population')
            # rank + number of excluded coordinates before that eligible rank.
            excluded_ranks = blocked - np.arange(blocked.size)
            for repeat, rng in enumerate(generators):
                ranks = rng.choice(available, size=count, replace=False)
                coordinates = ranks + np.searchsorted(excluded_ranks, ranks, side='right')
                result[repeat, cursor:cursor + count] = np.sort(coordinates) + start
            cursor += count
        if cursor != promoted.size:
            raise ValueError('Coordinate/layout mismatch in matched random sampling')
        return result

    def compare(self, client_id, round_number, raw, promoted, observed_jaccard):
        current = self.sample(client_id, round_number, raw, promoted)
        previous = self.previous.get(client_id)
        values = np.full(self.repeats, np.nan)
        if previous is not None:
            values = np.asarray([jaccard(current[r], previous[r])
                                 for r in range(self.repeats)])
        self.previous[client_id] = current
        valid = values[np.isfinite(values)]
        if valid.size:
            average = float(valid.mean())
            summary = dict(random_temporal_jaccard_mean=average,
                           random_temporal_jaccard_std=float(valid.std(ddof=1)),
                           random_temporal_jaccard_q025=float(np.quantile(valid, .025)),
                           random_temporal_jaccard_q975=float(np.quantile(valid, .975)),
                           temporal_jaccard_excess=float(observed_jaccard - average))
        else:
            summary = {key: float('nan') for key in [
                'random_temporal_jaccard_mean', 'random_temporal_jaccard_std',
                'random_temporal_jaccard_q025', 'random_temporal_jaccard_q975',
                'temporal_jaccard_excess']}
        return summary, values
