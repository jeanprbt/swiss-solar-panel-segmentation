from torch import nn

class UNet(nn.Module):
    def __init__(self, nb_layers: int, kernel_size: int):
        """
        Create a U-Net model with the given number of layers and kernel size.
        
        Args:
            nb_layers (int): Number of layers in the encoder, either 2 or 3.
            kernel_size (int): Kernel size of the convolutional layers.
        """
        super().__init__()
        encoder = []
        for i in range(nb_layers):
            in_channels, out_channels = 3 if i == 0 else 2 ** (5 + i), 2 ** (6 + i)
            padding = kernel_size // 2
            encoder.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
            )
            encoder += [nn.ReLU(), nn.MaxPool2d(2)]
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        for i in range(nb_layers - 1):
            in_channels, out_channels = 2 ** (5 + nb_layers - i), 2 ** (4 + nb_layers - i)
            decoder.append(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            )
            decoder.append(nn.ReLU())
            
        decoder.append(nn.ConvTranspose2d(2**6, 1, kernel_size=2, stride=2))
        decoder.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x