# Requirements Document

## Introduction

This spec documents the single-run, end-to-end Yard-Sale simulation with taxation and universal redistribution, as specified in `metodologia.txt` (chapter "Metodologia"). It covers one execution of one configuration (one regime, one tax rate, one seed). It records what was implemented and verified.

Out of scope for this spec (deferred):
- Multiple independent realizations, shared populations across regimes, and separate random streams per realization.
- The experimental grid of nine configurations, pilot runs (T vs 2T), transient selection and analysis windows.
- Quintile mobility, Lorenz curves, and statistical analysis across realizations.
- The full test suite of the "Verificação numérica" section.
- Performance optimization.
- Persisting the snapshot history to disk.

## Glossary

- **Agent**: an individual identified by an index `i`, with wealth `w_i(t) >= 0` and a fixed risk-aversion parameter `beta_i`.
- **Monte Carlo step (MCS)**: `N/2` sequential trade attempts. Time `t` is measured in MCS.
- **Trade value `a_e`**: gross amount exchanged in interaction `e` under the fair rule.
- **Cash `C(t)`**: the fund that holds collected taxes until redistribution.
- **W0**: total initial wealth, `N * w0`.
- **Regime**: one of `none` (no tax), `wealth` (tax on accumulated wealth) or `transaction` (tax on trades).

## Requirements

### Requirement 1: Artificial society and initial conditions

**User Story:** As a researcher, I want every run to start from an egalitarian population with heterogeneous, fixed risk aversion, so that inequality emerges only from the dynamics.

#### Acceptance Criteria

1. WHEN a simulation starts THEN the system SHALL create `n_agents` agents (default `N = 10^4`), each with wealth `w_i(0) = initial_wealth` (default `1.0`).
2. WHEN a simulation starts THEN the system SHALL draw each `beta_i` independently from `U[0, 1)` using a `np.random.default_rng(seed)` generator.
3. WHILE a run is in progress THE system SHALL keep each `beta_i` constant and preserve agent identity (array index).
4. WHEN a simulation starts THEN the system SHALL initialize the cash `C(0) = 0`.
5. THE system SHALL produce identical results for identical configuration and seed.

### Requirement 2: Fair trade rule

**User Story:** As a researcher, I want trades to follow the fair rule from the methodology, so that results are comparable with the referenced literature.

#### Acceptance Criteria

1. WHEN a trade attempt occurs THEN the system SHALL select two distinct agents uniformly at random from the whole population.
2. WHEN two agents `i` and `j` are selected THEN the system SHALL compute `a_e = min{(1 - beta_i) w_i, (1 - beta_j) w_j}` from the state immediately before the interaction.
3. WHEN the winner is decided THEN the system SHALL pick `i` or `j` with probability 1/2 each, independently of wealth and risk aversion.
4. WHEN no transaction tax applies THEN the system SHALL add `a_e` to the winner and subtract `a_e` from the loser.
5. THE system SHALL complete each update before selecting the next pair.
6. IF `a_e = 0` THEN the system SHALL still count the attempt in the time scale, with no transfer.
7. THE system SHALL never produce negative individual wealth.

### Requirement 3: Monte Carlo time scale

**User Story:** As a researcher, I want time expressed in Monte Carlo steps, so that runs of different sizes are comparable.

#### Acceptance Criteria

1. WHEN a Monte Carlo step runs THEN the system SHALL perform `n_agents // 2` sequential trade attempts.
2. THE system SHALL run `n_steps` Monte Carlo steps (default `300`).

### Requirement 4: Tax regimes

**User Story:** As a researcher, I want to choose between no tax, a wealth tax and a transaction tax, so that I can compare their effects.

#### Acceptance Criteria

1. THE configuration SHALL expose `regime` in {`none`, `wealth`, `transaction`} (default `none`) and `tax_rate` in `[0, 1]` (default `0.05`).
2. IF `regime` is not one of the allowed values OR `tax_rate` is outside `[0, 1]` THEN the system SHALL raise `ValueError` when the configuration is created.
3. WHEN `regime` is `none` THEN the system SHALL apply no tax, regardless of `tax_rate`.
4. WHEN `regime` is `transaction` THEN for each trade the system SHALL update `w_winner += (1 - tau_X) a_e`, `w_loser -= a_e` and `C += tau_X a_e`, with `a_e` computed before the update.
5. WHEN `regime` is `wealth` THEN at the end of each Monte Carlo step, after all trades, the system SHALL charge every agent `T_i = tau_W * w_i` simultaneously from the same pre-tax state and add the total to the cash.
6. WHEN `regime` is `wealth` THEN trades SHALL occur without any tax deduction.

### Requirement 5: Universal redistribution

**User Story:** As a researcher, I want all collected tax to be returned equally every step, so that the fiscal cycle closes each step.

#### Acceptance Criteria

1. WHEN a Monte Carlo step ends (after the wealth tax, if any) THEN the system SHALL give every agent `b = C / N` and set `C = 0`.
2. THE system SHALL apply redistribution in every regime (with `C = 0` in `none` it is a no-op).
3. THE system SHALL not apply administrative costs, retention or targeted redistribution.

### Requirement 6: Conservation and numerical diagnostics

**User Story:** As a researcher, I want conservation and non-negativity monitored, so that I can trust the results.

#### Acceptance Criteria

1. THE system SHALL compute the relative conservation error `eps_W = |sum_i w_i + C - W0| / W0` before and after redistribution in every step, and report its maximum for the run.
2. THE system SHALL track and report the minimum individual wealth observed and the maximum cash balance.
3. THE system SHALL not apply automatic corrections to balances.

### Requirement 7: Indicators

**User Story:** As a researcher, I want per-step indicators recorded after redistribution, so that I can follow the dynamics over time.

#### Acceptance Criteria

1. WHEN a Monte Carlo step ends THEN after redistribution the system SHALL record:
   - the Gini coefficient `G(t)` using the sorted-wealth formula;
   - the share of the richest agent `max_i w_i / sum_i w_i`;
   - the gross traded volume `v(t) = (sum of a_e in the step) / W0`;
   - the step revenue `q(t) = A(t) / W0`, where `A(t)` is the cash immediately before redistribution.
2. WHEN the run ends THEN the system SHALL report final summary metrics: total, mean, median, min and max wealth, Gini and richest-agent share.
3. THE system SHALL store wealth snapshots at step 0, every `snapshot_interval` steps (default `10`) and at the final step, in memory only.
4. IF `snapshot_interval < 1` THEN the system SHALL raise `ValueError`.

### Requirement 8: Execution timing

**User Story:** As a researcher, I want to know how long a run takes, so that I can plan the larger experiments.

#### Acceptance Criteria

1. THE system SHALL measure the wall-clock duration of a run with `time.perf_counter()`, from initialization to the final metrics, excluding figure generation.
2. THE system SHALL return the duration in seconds as part of the run result.
3. WHEN `main.py` runs THEN the system SHALL print the total duration as `<seconds> s (<m> min <s> s)`.

### Requirement 9: Output

**User Story:** As a researcher, I want a printed summary and saved figures, so that runs can be inspected without a display (e.g. in a container).

#### Acceptance Criteria

1. WHEN `main.py` runs THEN the system SHALL print the regime, tax rate, population size, number of steps, summary metrics, diagnostics, number of snapshots and duration.
2. WHEN `main.py` runs THEN the system SHALL save to `results_dir` (default `Results`) a wealth histogram (log y-axis) and a per-step time-series figure (Gini, richest share, volume, revenue).
3. THE figure file names SHALL include the regime and, for taxed regimes, the tax rate (e.g. `series_temporais_wealth_tau0.05.png`).
4. THE system SHALL open plot windows only when `show_plots` is `True` (default `False`).
