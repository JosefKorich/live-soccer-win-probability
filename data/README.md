# Data

Raw third-party datasets are intentionally not committed to this repository.

## Historical Brazilian match results

The baseline analysis was calibrated on a historical Brazilian soccer results dataset containing 372 Corinthians Série A matches in the filtered sample. The repository stores only the aggregate MLE rates used to reproduce the stochastic-model results:

- Corinthians home: 0.016547 goals/minute (1.49 per 90)
- Opponent at Corinthians home: 0.008961 goals/minute (0.81 per 90)
- Corinthians away: 0.010872 goals/minute (0.98 per 90)
- Opponent at Corinthians away: 0.012724 goals/minute (1.15 per 90)

## StatsBomb Open Data

The event-aware extension uses StatsBomb Open Data through `statsbombpy`. The package can fetch public competition, match, event, and xG data at runtime.

Source: https://github.com/statsbomb/open-data

The repository does not redistribute StatsBomb's underlying data.
