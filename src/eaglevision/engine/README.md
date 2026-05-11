# Engine

Training and evaluation orchestration lives here.

## Files

- `trainer.py`: main training loop.
- `evaluator.py`: evaluation loop.
- `checkpointing.py`: save/load checkpoint helpers.
- `logging.py`: metric logging utilities.

Keep model math and dataset logic in their own modules. The engine should compose those pieces into repeatable workflows.
