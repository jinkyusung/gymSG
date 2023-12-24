import torch
import torch.nn as nn
import numpy as np

class ClassificationNetworkColors(torch.nn.Module):
    def __init__(self):

        super().__init__()
        # setting device on GPU if available, else CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.classes = [[-1., 0., 0.],  # left
                        [-1., 0.5, 0.], # left and accelerate
                        [-1., 0., 0.8], # left and brake
                        [1., 0., 0.],   # right
                        [1., 0.5, 0.],  # right and accelerate
                        [1., 0., 0.8],  # right and brake
                        [0., 0., 0.],   # no input
                        [0., 0.5, 0.],  # accelerate
                        [0., 0., 0.8]]  # brake

        """
        D : Network Implementation

        Implementation of the network layers. 
        The image size of the input observations is 96x96 pixels.

        Using torch.nn.Sequential(), implement each convolution layers and Linear layers
        """

        # convolution layers
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 8, 2),  # 45 x 45 x 64 tensor
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 4, 2),  # 21 x 21 x 32 tensor
            nn.LeakyReLU(),
        )

        # Linear layers (output size : 9)
        self.fc = nn.Sequential(
            nn.Linear(21*21*32, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 9),
        )

    def forward(self, observation):
        """
        D : Network Implementation

        The forward pass of the network. 
        Returns the prediction for the given input observation.
        observation:   torch.Tensor of size (batch_size, 96, 96, 3)
        return         torch.Tensor of size (batch_size, C)

        """
        batch_size = observation.shape[0]
        x = observation.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x

    def actions_to_classes(self, actions):
        """
        C : Conversion from action to classes

        For a given set of actions map every action to its corresponding
        action-class representation. Every action is represented by a 1-dim vector 
        with the entry corresponding to the class number.
        actions:        python list of N torch.Tensors of size 3
        return          python list of N torch.Tensors of size 1
        """
        return [torch.Tensor([self.classes.index(list(v))]).long() for v in actions]

    def scores_to_action(self, scores):
        """
        C : Selection of action from scores

        Maps the scores predicted by the network to an action-class and returns
        the corresponding action [steer, gas, brake].
                        C = number of classes
        scores:         python list of torch.Tensors of size C
        return          (float, float, float)
        """
        return self.classes[np.argmax(scores[0].detach().numpy())]
