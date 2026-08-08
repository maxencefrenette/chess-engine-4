# Paired-opening Elo confidence intervals

## Goal

Replace the tournament report's independent Bernoulli working-model uncertainty
with intervals that respect draws and mirrored opening pairs, without changing
the adaptive Swiss schedule or launching additional games.

This is a retained-data methodology analysis. It has no W&B run, training
`EG_flops`, or promotion decision.

## Retention audit

Before this change, the adaptive lc0 runner used sequential mirrored openings.
Lc0's fake-PGN results file retained only separate white/black aggregate W/D/L
counts. Its captured stdout contained a `gameready` record with `gameid`, player-1
color, and result for every game, but `parse_match_result` discarded those records.
Consequently, all three committed tournament JSON files retain matchup W/D/L but
cannot reconstruct which two outcomes shared an opening.

Fastchess already used `-repeat`, requested `penta=true`, and retained a full PGN
on the Modal volume. Its Python result exposed only stdout and remote paths, not
structured paired evidence. The retained 200-game searched match from
`2026-07-11.02-dense-1e22-t74-eval` was downloaded read-only from that volume and
does retain the required game pairs.

Future adaptive reports now store both:

- `pair_scores`: ordered half-point scores `0..4` for each mirrored opening pair;
- `pentanomial`: counts for pair scores `0, 0.5, 1, 1.5, 2`.

Fastchess results return the same two fields parsed from same-round reversed-color
PGN games. Legacy tournament JSON remains resumable and is explicitly treated as
unpaired.

## Statistical model

For engine ratings `r`, the expected score for engine `i` against `j` is

```text
p(i beats j) = logistic(log(10) / 400 * (r_i - r_j)).
```

Wins, draws, and losses contribute scores `1`, `0.5`, and `0`. The point estimate
is a penalized Bradley-Terry score quasi-MLE, constrained so ratings sum to zero. A weak
centered `Normal(0, 10000 Elo)` penalty only makes separated or otherwise
degenerate data finite; its largest point-estimate change on retained tournaments
is `0.12 Elo`.

The 95% interval is the two-sided Wald interval
`r_i +/- 1.959964 * sqrt(V_ii)`. `V` is a CR1 sandwich covariance. When ordered
pair scores are present, the two color-reversed games form one cluster, and all
results using the same indexed sequential opening across matchups are also in
that cluster. When only pentanomial counts are present, each pair is a cluster
but cross-match opening identity is unavailable. For genuinely unpaired or
legacy aggregate data, each W/D/L game contributes an independent score cluster.
The CR1 multiplier is `G / (G - k)`, where `G` is the number of clusters and `k`
is the number of free ratings.

Reported intervals are marginal intervals for each sum-to-zero rating (rating
relative to the field mean), not pairwise-difference intervals. A pairwise Elo
difference must use its covariance contrast; in a two-engine match its half-width
is twice either centered rating's half-width.

## Retained-data results at matched game count

The old estimator and new fallback were evaluated on every committed tournament
without changing its games. Width ratios below are new/old per-engine 95% CI
half-widths.

| Retained tournament | Games | Engines | Old mean half-width | New mean half-width | Mean ratio | Per-engine ratio range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-07-12.04 policy round robin | 1,792 | 8 | 55.91 | 40.26 | 0.719 | 0.667-0.800 |
| 2026-08-06.05 dense policy | 1,280 | 10 | 96.07 | 68.87 | 0.711 | 0.635-0.794 |
| 2026-08-07.01 dense/MoE policy | 1,792 | 14 | 76.07 | 51.38 | 0.669 | 0.604-0.745 |

These historical reductions come from using the observed trinomial score
variance rather than pretending every draw is a Bernoulli trial. They are the
documented unpaired fallback, not a retrospective pentanomial claim.

The retained searched match has candidate W/D/L `0/6/194` and pentanomial
`[94, 6, 0, 0, 0]` over 100 opening pairs. At the same 200 games:

| Estimator | Centered candidate Elo | 95% CI half-width | Ratio to old |
| --- | ---: | ---: | ---: |
| Old working-binomial | -363.47 | 99.04 | 1.000 |
| New unpaired robust fallback | -363.45 | 69.65 | 0.703 |
| New paired-opening robust | -363.45 | 68.74 | 0.694 |

The paired treatment is `1.3%` narrower than the new unpaired fallback here and
`30.6%` narrower than the old interval. This particular near-sweep has little
within-pair ambiguity; larger paired gains are expected only when the two colors'
outcomes are materially correlated. No new validation tournament was required or
launched because this retained 100-pair searched PGN supplies real paired data.

## Assumptions and limitations

- Wald/sandwich intervals are asymptotic. Coverage can be poor with few opening
  clusters, near separation, or a singular empirical score distribution.
- A fully deterministic set of identical pair scores has zero empirical sandwich
  variance. The finite point estimate does not turn that zero-width interval into
  evidence about unseen openings; reports must disclose the opening sample and
  cluster count.
- Legacy aggregate W/D/L cannot recover color-pair covariance or shared opening
  identity. The fallback labels that limitation rather than guessing a pairing.
- Ordered pair scores identify sequential opening indices within this runner.
  They do not store the opening FEN itself, so evidence should not be combined
  across runs that use different books or ordering modes without additional data.
- Adaptive scheduling changes which engine contrasts are observed, but not the
  estimator or resumability. The sandwich covariance is conditional on the
  realized schedule.

The detailed machine-readable comparison is in `results.json`.
