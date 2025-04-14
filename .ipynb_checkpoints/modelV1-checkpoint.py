import torch
import torch.nn as nn

from modularModels1 import BlockMaker, modularNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_1 = modularNN(10,
                   blocks=[BlockMaker(28,[128,128],64,
                                      [nn.ReLU(),nn.ReLU(),nn.LeakyReLU()]),
                           BlockMaker(64,[64],32,
                                      [(nn.ELU(),nn.Dropout(0.2)),nn.Sigmoid()]),
                           ],
                   blockSize=(28,32)).to(device)

# model_2 = modularNN(10,
#                    blocks=[BlockMaker(28,[64,64],32,
#                                       [nn.Sigmoid(),(nn.Sigmoid(),nn.Dropout(0.3)),nn.Sigmoid()]),
#                            BlockMaker(32,[32],16,
#                                       [(nn.Tanh(),nn.Dropout(0.3)),nn.Sigmoid()]),
#                            ],
#                    blockSize=(28,16))


