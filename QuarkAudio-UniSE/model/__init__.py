"""UniSE model package.

The preview runtime imports the lightweight codec and LLM submodules directly.
Importing the training-oriented ``Model`` here would unnecessarily require
PyTorch Lightning and the dataset stack during inference.
"""

__all__ = []
