# Real-time post-selection strategy for sliding window decoding

For details of sliding window decoding, see @docs/sliding_window_decoding.md

## Post-selection strategy

Suppose that we're running soft window decoding with parameters `(W, F)` for a code with `T` rounds.

- `metric(window_start, window_end)`: committed cluster llr norm frac with a given norm order and `eval_windows=(window_start, window_end)`
- `c`: cutoff for post-selection
- `metric_windows`: number of windows for metric evaluation
- While running soft window decoding, compute `metric(window - metric_windows + 1, window_end)` in real time, for each `window >= metric_windows - 1`
- When the metric exceeds `c`, terminate the simulation and conclude that the decoding is aborted. If the decoding succeeds with the metric not exceeding `c` untile the end, the decoding is accepted.
- The logical error rate is evaluated only with accepted samples.

## Analysis goal

Analyze how the "effective average number of trials" and logical error rate vary depending on `c`. Here, the effective average number of trials can be calculated from multiple sample results as the summation of `(window_exceeding_cutoff + 1) * F / T` for samples divided by the number of accepted samples.