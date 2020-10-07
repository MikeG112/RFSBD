# Shot Boundary Detection

This repository contains and implementation of the model and training scheme
proposed in:

https://arxiv.org/pdf/1705.08214.pdf


# Requirements

The main requirements to use this repository are:

1.  python3
2.  pytorch
3.  pytest (to run unit tests)

 To construct an environment containing the specific versions of these
 dependencies used to tes the code, one should use conda or pip to ingest ``requirements.yml`` or ``requirements.txt`` respectively.

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

# Testing

To run the unit test suite for this repository, cd into the outermost directory
and run:

``>>> pytest``

Currently, there is precisely one unit test which checks that the forward pass
of RFSBD runs on a single test video.
