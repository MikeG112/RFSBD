import torch
import torch.nn as nn
import torch.nn.functional as F

class RFSBD(nn.Module):
    '''
    Module implementing the Shot Boundary Detection model proposed in
    https://arxiv.org/pdf/1705.08214.pdf
    '''

    def __init__(self, in_channels=3):
        super(RFSBD, self).__init__()
        self.conv_layer1 = nn.Conv3d(in_channels, 16,
                                     kernel_size=(11,5,5),
                                     padding=(5,2,2))
        self.conv_layer2 = nn.Conv3d(16, 32,
                                     kernel_size=(11,5,5),
                                     padding=(5,2,2))

        self.conv_layer3 = nn.Conv3d(32, 64,
                                     kernel_size=(11,5,5),
                                     padding=(5,2,2))

        self.conv_layer4 = nn.Conv3d(64, 32,
                                     kernel_size=(11,5,5),
                                     padding=(5,2,2))

        self.conv_layer5 = nn.Conv3d(32, 16,
                                     kernel_size=(11,5,5),
                                     padding=(5,2,2))

        self.conv_layer6 = nn.Conv3d(16, 2,
                                     kernel_size=(11,5,5),
                                     padding=(5,2,2))

        self.max_pool = nn.MaxPool3d(kernel_size=(1,2,2))

        self.max_pool_4 = nn.MaxPool3d(kernel_size=(1,4,4))

        self.avg_pool = nn.AvgPool3d(kernel_size=(1,2,2))

        self.softmax = nn.Softmax(dim=1)


    def forward(self, video):
        '''
        Parameters
        ----------
        video: torch.tensor
            Tensor of shape (batch_size, in_channels, num_frames,
                             frame_width=128, frame_height=128)
            encoding a video.
        Returns
        -------
        out: torch.tensor
            Tensor of shape (batch_size, num_frames, 2), where the last
            dimension encodes the probability that a given frame belongs
            to the same shot a the previous frame, or belongs to a different
            shot than the previous frame, respectively.
        '''
        # TODO: downsample input videos with larger frames to compressed size
        y = F.relu(self.conv_layer1(video)) # channels increased to 16
        y = self.max_pool(y) # spatial dims decreased by factor of 2
        y = F.relu(self.conv_layer2(y))
        y = self.max_pool(y)
        y = F.relu(self.conv_layer3(y))
        y = self.avg_pool(y)
        y = F.relu(self.conv_layer4(y))
        y = self.max_pool(y)
        y = F.relu(self.conv_layer5(y))
        y = self.max_pool(y)
        y = F.relu(self.conv_layer6(y))
        y = self.max_pool_4(y)
        # Output shape = (batch_size, num_frames, 2)
        output = self.softmax(y).squeeze(-1).squeeze(-1).permute(0,2,1)
        return output
