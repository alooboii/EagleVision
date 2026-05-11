# Configs

Reusable YAML configuration files live here.

## Folder Layout

- `data/`: dataset locations and loading assumptions.
- `model/`: model and adapter settings.
- `train/`: training runtime profiles.
- `eval/`: evaluation runtime profiles.

## Conventions

- Keep committed configs generic enough to run outside one person's machine.
- Use relative paths or documented environment/runtime paths where possible.
- Put Kaggle-specific configs here only when they are intended to be reproducible from Kaggle.
- Do not hard-code credentials, local usernames, or private dataset paths.

## Runtime Profiles

The repo keeps both full and smaller profiles:

- `phase1.yaml`: default Phase 1 training profile.
- `phase1_hourly.yaml`: lower-cost training profile for quicker iteration.
- `phase1_kaggle_complete.yaml`: Kaggle-oriented full Phase 1 profile.
- `default.yaml`: default evaluation profile.
- `hourly.yaml`: lower-cost evaluation profile.
- `phase1_kaggle_complete.yaml`: Kaggle-oriented evaluation profile.
