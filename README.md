# Shot Boundary Detection

This repository contains and implementation of the model and training scheme
proposed in:

https://arxiv.org/pdf/1705.08214.pdf

This was completed, among other tasks, within the timeframe of a one day technical interview for an ML position.
Since I do not have independent use for it, I have opted not to complete the unfinished implementation
details e.g. the training script.

# Requirements

The main requirements to use this repository are:

1.  python3
2.  pytorch
3.  pytest (to run unit tests)

 To construct an environment containing the specific versions of these
 dependencies used to test the code, one should use conda or pip to ingest
``requirements.yml`` or ``requirements.txt`` respectively. For example, with
conda one can run:

``conda env create -f requirements.yml``

# Usage

Currently, the functionality of this repository is restricted to using the
RFSBD model defined in ``src/models/model.py``. For example, to run inference
on a random test video:

```
>>> import torch
>>> random_video = torch.rand((1, 3, 30, 128, 128))
>>> from src.models.model import RFSBD
>>> model = RFSBD()
>>> model(random_video)
```

The training script is under construction, and at this time is not finished.

# Testing

To run the unit test suite for this repository, cd into the outermost directory
and run:

``>>> pytest``

Currently, there is precisely one unit test which checks that the forward pass
of RFSBD runs on a single test video.
