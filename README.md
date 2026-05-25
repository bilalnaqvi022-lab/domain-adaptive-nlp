# NLP Project Scaffold
This project provides a modular NLP starter setup using `transformers`, `torch`, `datasets`, `scikit-learn`, `pandas`, and `matplotlib`.
## Project structure
- `data/`: dataset loading and data utilities
- `models/`: model/tokenizer setup
- `training/`: training logic
- `evaluation/`: evaluation metrics and plotting
- `outputs/`: trained model and evaluation outputs
- `main.py`: end-to-end training + evaluation entrypoint
## Quick start (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```
## Notes
- Default pipeline uses a small IMDB subset for faster iteration.
- First run will download model and dataset artifacts.
