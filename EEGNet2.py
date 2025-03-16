
class EEGNet(nn.Module):
    def __init__(self ,eeg_channels, num_temporal_filters, num_spacial_channels, num_classes):
        super().__init__()
        self.num_temporal_filters = num_temporal_filters
        self.num_spacial_channels = num_spacial_channels
        self.eeg_channels = eeg_channels
        self.num_classes = num_classes

        self.parallel_networks = nn.ModuleList([nn.Sequential(
            nn.Conv2d(1, 1, (1, 4* (2 ** i)), padding='same'),
            self._spacial_filter(4 * (2 ** i))
        ) for i in range(num_temporal_filters)])

        self.conv_block1 = self._conv_block(num_temporal_filters * num_spacial_channels, 64, (1, 64), (1, 4))
        self.conv_block2 = self._conv_block(64, 16, (1, 64), (1, 2))

        self.linear1 = nn.Linear(10_000, 64)
        self.relu = nn.ReLU()

        self.linear2 = nn.Linear(64, 1)
        self.linear3 = nn.Linear(1, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def _conv_block(self, in_features, out_features, kernel, pool=False, padding='same'):
        return nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel, padding=padding),
            nn.ELU(0.5),
            nn.AvgPool2d(pool) if pool else nn.AvgPool2d(1),
            nn.Dropout(0.2)
        )

    def _spacial_filter(self, kernel_width):
        return nn.Sequential(
            self._conv_block(1, 64, (self.eeg_channels, 1), padding='valid'),
            self._conv_block(64, self.num_spacial_channels, (1, kernel_width), pool=(1, 2))
        )

    def forward(self, x, is_train=True):
        x = torch.concat([parallel_filter(x) for parallel_filter in self.parallel_networks], dim=1)
        # print(x.shape)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        # print(x.shape)

        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        x = self.relu(x)

        x = self.linear2(x)
        x = self.relu(x)

        x = self.linear3(x)
        if not is_train:
            x = self.softmax(x)
            return x
        x = nn.functional.log_softmax(x, dim=1)
        return x