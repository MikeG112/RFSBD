# TODO: Add data import from pipeline, a training wrapper
# around the train step and evaluation step, and some logging and visualization
# of resulting model performance.

# External imports
import torch
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split

# Internal imports
from src.utils.data import ArrayDataset
from src.utils.train import train
from src.models.model import RFSBD

# Define some fake training data
random_videos = torch.rand((101, 3, 30, 128, 128))
random_labels = torch.randint(low=0, high=2, size=(101,30,1))

# Split off val set
train_videos, val_videos, train_labels, val_labels = train_test_split(
                                                             random_videos,
                                                             random_labels,
                                                             train_size = .8)

train_data = ArrayDataset(train_videos, train_labels)
val_data = ArrayDataset(val_videos, val_labels)


# Set parameters and run
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class_weights = [.5, .5]
class_weights = torch.FloatTensor(class_weights).to(device)
loss_fn = CrossEntropyLoss(weight=class_weights)
epochs = 20
model = RFSBD().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.88)

train(model, train_data, val_data, optimizer, loss_fn,
          epochs, device=device)
