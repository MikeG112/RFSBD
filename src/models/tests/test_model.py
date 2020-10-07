# External imports
import torch

# Internal imports
from src.models.model import RFSBD

def test_RFSBD_forwards():

    # shape: (batch, in_channels, num_frames, frame height, frame width)
    random_video = torch.rand((1, 3, 30, 128, 128))

    model = RFSBD()

    assert model(random_video) is not None, "Model doesn't compute!"

    return None
