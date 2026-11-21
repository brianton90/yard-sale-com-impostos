# Implementation Plan

- [x] 1. Add regime and tax rate to the configuration
  - Add `REGIMES`, `regime` and `tax_rate` to `config.py`, with validation in `__post_init__`.
  - Add derived properties `wealth_tax_rate` and `transaction_tax_rate`.
  - _Requirements: 4.1, 4.2, 4.3_

- [x] 2. Add the cash to the simulation state
  - Add `cash: float = 0.0` to `SimulationState` in `simulation/model.py`.
  - _Requirements: 1.4_

- [x] 3. Implement the fair trade rule and the transaction tax
  - Change `compute_stake(state, i, j)` to `min{(1 - beta_i) w_i, (1 - beta_j) w_j}`; remove `compute_poorer_agent`.
  - Add `transaction_tax_rate` to `apply_trade` and `yard_sale_step`; route `tau_X * a_e` to the cash.
  - Remove `poorer` from `TradeResult`.
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 4.4, 4.6_

- [x] 4. Implement wealth tax and redistribution
  - Create `simulation/taxes.py` with `apply_wealth_tax` (simultaneous, vectorized) and `redistribute_cash`.
  - _Requirements: 4.5, 5.1, 5.3_

- [x] 5. Add indicators and diagnostics
  - Add `compute_max_share` and `compute_conservation_error` to `simulation/metrics.py`.
  - Add `max_share` and the diagnostics fields to `SimulationMetrics`; add `SimulationSeries`.
  - _Requirements: 6.1, 6.2, 7.1, 7.2_

- [x] 6. Orchestrate the step sequence and time the run
  - In `simulation/engine.py`, run `N/2` trades, the wealth tax, the conservation check, redistribution, then record indicators after redistribution.
  - Keep snapshots at step 0, every `snapshot_interval` steps and at the final step.
  - Measure the duration with `time.perf_counter()` and return a `SimulationResult`.
  - _Requirements: 3.1, 3.2, 5.2, 6.1, 6.3, 7.1, 7.3, 7.4, 8.1, 8.2_

- [x] 7. Print the summary and save the figures
  - Add `plot_time_series` to `visualization/plots.py`.
  - Update `main.py` to print the regime, diagnostics and duration, and save both figures with the regime tag in the file name.
  - _Requirements: 8.3, 9.1, 9.2, 9.3, 9.4_

- [x] 8. Update the unit test for the new stake rule
  - Update `tests/test_rules.py` to call `compute_stake(state, i, j)` and assert the minimum of the available amounts.
  - _Requirements: 2.2_

- [x] 9. Verify the implementation with a smoke run
  - Run all three regimes (`N = 1000`, 20 steps, `tau = 0.2`) and check conservation, non-negativity, zero-rate equivalence and the `tau_W = 1` case.
  - Run `main.py` with the defaults and record the duration (61.71 s).
  - _Requirements: 1.5, 2.7, 6.1, 6.2, 8.3_

- [ ] 10. Run the pytest suite in a working environment
  - Install the dependencies (`pip install -r requirements.txt`, e.g. in a venv inside WSL).
  - Run `pytest` and confirm `tests/test_rules.py` and `tests/test_metrics.py` pass.
  - _Requirements: 2.2, 7.1_
