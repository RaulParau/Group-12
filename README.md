# Group 12 - 2026 IIVP Hindi Digits Classification Challenge

## Collaborators
| Name   | Username | StudentID |
|----------|----------|-------------|
| Omer Nidam    | OmericoN   | i6384394  |
| Raul Parau | RaulParau   | i6399837    |
| Julian Kinkel | juliankinkel  | i6406013    |

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

### Parameter Configuration
Navigate to the `src/config.py` file.
There you will be able to adjust the hyperparameters and CNN model selection
The current configuration:
| Hyperparameter     | Value      | Description                          |
|--------------------|------------|--------------------------------------|
| model              | resnet     | Model architecture                   |
| optimizer          | AdamW      | Optimization algorithm               |
| lr                 | 3e-4       | OneCycle maximum learning rate       |
| weight_decay       | 1e-4       | L2 regularization strength           |
| epochs             | 15         | Number of training epochs            |
| label_smoothing    | 0.05       | Label smoothing factor               |
| scheduler          | onecycle   | Learning rate scheduler              |
| pct_start          | 0.3        | Fraction of cycle spent increasing LR|
| div_factor         | 25.0       | Initial LR division factor           |
| final_div_factor   | 1e4        | Final LR division factor             |

### Model Training 
run
```bash
uv run src/train.py
```
Note that the function `train()` takes `verbose=False` by default.

By setting `verbose=True` the function will also print each epoch evaluation statistics (including loss & accuracy plots)

After running the function the best epoch of the model will be saved as a `.pth` file under the `models/` directory

### Generating Kaggle Predictions
run
```bash
uv run src/kaggle_inference.py
```
This will produce a csv file comprising image ID, Category under `outputs/submission.csv`

