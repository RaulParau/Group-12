# Group 12 - 2026 IIVP Hindi Digits Classification Challenge

## Requirements
-  `uv` - [[uv-installation](https://docs.astral.sh/uv/getting-started/installation/) ]

To sync with venv and configuration (with all the dependencies):
```bash
uv sync
```
(If contributing) to add packages:
```bash
uv add <package_name>
```
To execute code (automatically uses venv):
```bash
uv run main.py
```

## Getting Started

### Model Training 
run
```bash
uv run src/train.py
```
Note that the function `train()` takes `verbose=False` by default.

By setting `verbose=True` the function will also print each epoch evaluation statistics

After running the function the best epoch of the model will be saved as a `.pth` file under the `models/` directory

### Model Testing
run
```bash
uv run src/test.py
```
This will test the saved model on an unseen test set (15% of the overall dataset)

> Note this isn't the final test that is submitted to Kaggle (it's only a test subset from the train dataset)

