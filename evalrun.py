# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Compatibility wrapper.

`run.py` is the canonical runner for both training and evaluation.
Use eval configs (`only_eval: true`) with this entrypoint to preserve existing scripts.
"""

from run import main


if __name__ == "__main__":
    main()
