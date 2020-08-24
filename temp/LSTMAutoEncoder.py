import torch
import torch.nn as nn


class LSTMAutoEncoder(nn.Module):
    
    __config = {
        'num_features': 79,
        'l1_hid_dim': 128,
        'l2_hid_dim': 32,
        'l3_hid_dim': 32,
        'l4_hid_dim': 128,
    }
    
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = self.config
        self.config = config
        kwargs = dict(batch_first=True, bidirectional=True)
        # input shape is (bsz, window, num_features)
        self.L1 = nn.LSTM(self.num_features, self.l1_hid_dim, **kwargs)
        self.L2 = nn.LSTM(self.l1_hid_dim*2, self.l2_hid_dim, **kwargs)
        self.L3 = nn.LSTM(self.l2_hid_dim*2, self.l3_hid_dim, **kwargs)
        self.L4 = nn.LSTM(self.l3_hid_dim*2, self.l4_hid_dim, **kwargs)
        self.fc = nn.Linear(self.l4_hid_dim*2, self.num_features)
        
    def forward(self, x):
        window = x.size(1)
        l1 = torch.relu(self.L1(x)[0])
        l2 = torch.relu(self.L2(l1)[0])
        l2 = l2[:, -1, :].unsqueeze(dim=1).repeat(1, window, 1)
        l3 = torch.relu(self.L3(l2)[0])
        l4 = torch.relu(self.L4(l3)[0])
        output = self.fc(l4)
        return output
    
    @property
    def config(self):
        return self.__config
    
    @config.setter
    def config(self, config):
        if not isinstance(config, dict):
            raise AttributeError("config must be ``dict``.")
        for key, val in config.items():
            if key in self.config.keys():
                setattr(self, key, val)
                self.config[key] = val
