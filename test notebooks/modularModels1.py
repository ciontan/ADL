import torch
import torch.nn as nn


"""
List of torch.nn functions for a basic linear Neural Netowrk

torch.nn.Linear

torch.nn.Dropout

Normalizers:
    torch.nn.BatchNorm1d
    torch.nn.LayerNorm

Activation Fns:
    torch.nn.ReLU
    torch.nn.LeakyReLU
    torch.nn.Sigmoid
    torch.nn.Tanh
    torch.nn.Softmax
    torch.nn.Softplus
    torch.nn.PReLU
    torch.nn.ELU
"""
class BasicModel(nn.Module):
    def __init__(self,inFeatures,hiddens,classes,activations:list):
        super().__init__()
        if isinstance(activations,list):
            assert len(activations) <= len(hiddens)
        linearList = []
        start = inFeatures
        for size in hiddens:
            linearList.append(nn.Linear(start,size))
            start = size

        layerList = []
        for idx in range(len(linearList)):
            layerList.append(linearList[idx])
            if isinstance(activations,list):
                if idx < len(activations):
                    layerList.append(activations[idx])
                else:
                    layerList.append(activations[-1])
            else:
                layerList.append(activations)
        layerList.append(nn.Linear(size,classes))
        # layerList.append(nn.Sigmoid())
        self.block = nn.Sequential(
            *layerList,
        )
        
    def forward(self, x):
        return self.block(x)
    
class newModel1(nn.Module):
    """
    ALOHA
    """
    def __init__(self, inputs:int, hidden:list = None, outputs:int = 2,
                 blocks:list = None, blockSize:tuple = None,
                 activation = nn.ReLU(), dropout:float = None,
                 ):
        """
        Params:
        - inputs: Number of input features
        - hidden: List of hidden layer sizes
        - outputs: Number of Output classes
        - blocks: List of premade Blocks
        - blockSize: tuple pair of input size for starting block AND output size for ending block
        - activation: Activation function used between each layer
        - dropout: dropout ratio after each layer(not implements if using Blocks)
        """
        super().__init__()

        if blocks:
            self.layers = nn.Sequential(
                nn.Linear(inputs,blockSize[0]),
                activation,
                *blocks,
                nn.Linear(blockSize[1],outputs),
            )

        elif hidden:

            layer_list = []
            inp = inputs
            for size in hidden:
                layer_list.append(nn.Linear(inp,size))
                layer_list.append(activation)
                if dropout: layer_list.append(nn.Dropout(dropout))
                inp = size
            layer_list.append(nn.Linear(inp,outputs))
            self.layers = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.layers(x)
        

class Block_1Layer(nn.Module):
    """
    A 2 layer Block, with optional dropout"""
    def __init__(self, input:int, output:int,
                 activation = nn.ReLU(),
                 dropout:float = None):
        """
        Params:
        - input: Number of input features
        - outputs: Number of Output classes
        - activation: nn.ReLU, LeakyReLU, nn.Sigmoid, nn.Tanh, nn.Softmax, nn.Softplus, nn.PReLU, nn.ELU
        - dropout: dropout ratio
        """
        super().__init__()
        layer_List = []
        layer_List.append(nn.Linear(input,output))
        layer_List.append(activation)
        if dropout: 
            assert 1 > dropout and dropout > 0 , "Dropout should be between 0 and 1"
            layer_List.append(nn.Dropout(dropout))
 
        self.layers =  nn.Sequential(
            *layer_List
        )
    
    def forward(self, x):
        return self.layers(x)

class Block_2Layer(nn.Module):
    """
    A 2 layer Block, with optional dropout between the layers"""
    def __init__(self, in_features:int, hidden:int, out_features:int,
                 activation = nn.ReLU(),
                 dropout:float = None):
        super().__init__()
        layer_List = []
        layer_List.append(nn.Linear(in_features,hidden))
        layer_List.append(activation)
        if dropout: 
            assert 1 > dropout and dropout > 0 , "Dropout should be between 0 and 1"
            layer_List.append(nn.Dropout(dropout))
        layer_List.append(nn.Linear(hidden,out_features))
        layer_List.append(activation)
            
        self.layers =  nn.Sequential(
            *layer_List
        )
    
    def forward(self, x):
        return self.layers(x)
    
class modularNN(nn.Module):
    """
    Use with BlockMaker
    """
    def __init__(self, classes:int, blocks:list = None, blockSize:tuple = None,
                 ):
        """
        Params:
        - blocks: List of premade Blocks
        - blockSize: tuple pair of input size for starting block AND output size for ending block
        """
        super().__init__()

        if blocks and blockSize:
            self.layers = nn.Sequential(
                *blocks,
                nn.Linear(blockSize[1],classes),
            )
 
    def forward(self, x):
        return self.layers(x)
        

class BlockMaker(nn.Module):
    """
    Create a Sequential Block with your own choice of Activation Functions!
    """
    def __init__(self, in_features:int, hidden_neurons:list|tuple, out_features:int,
                 otherModules:list|tuple,
                 ):
        """
        Params:
        - sizes: (list) layer sizes
        - otherModules: activations or dropouts to add after each layer. put them in a tuple to add more than 1 after a layer
        """
        assert len(hidden_neurons)+1 == len(otherModules), "num of Layers and Modules do not match."

        super().__init__()
        sizes = [ *hidden_neurons, out_features]
        linear_List = []
        start = in_features
        for size in sizes:
            linear_List.append(nn.Linear(start,size))
            start = size

        layer_List = []
        try:
            for idx in range(len(linear_List)):
                layer_List.append(linear_List[idx])
                if isinstance(otherModules[idx],tuple):
                    for module in otherModules[idx]:
                        layer_List.append(module)
                else:
                    layer_List.append(otherModules[idx])
        except IndexError:
            print("num of Layers and Modules do not match!")
        except:
            print("Unknown Error")

        self.layers =  nn.Sequential(
            *layer_List
            )
            
    def forward(self, x):
        return self.layers(x)

