import torch
import torch.nn as nn

class Board:
    def __init__(self):
        self.board = torch.zeros(9, dtype=torch.float32)
        self.sum = 0
        self.game = []

    def printBoard(self):
        for i in range(3):
            print(self.board[i*3:(i+1)*3])

    def _play(self, id, val):
        if self.board[val].item() == 0:
            self.sum += 1
            self.board[val] = id
            self.game.append((id, val))
            return True
        return False

    def _checkWin(self, id, val):
        b = self.board  # shorthand for readability
        match val:
            case 0:
                if (b[0] == b[1]).item() and (b[1] == b[2]).item() and (b[0] == id).item():
                    return True
                if (b[0] == b[3]).item() and (b[3] == b[6]).item() and (b[0] == id).item():
                    return True
                if (b[0] == b[4]).item() and (b[4] == b[8]).item() and (b[0] == id).item():
                    return True
            case 1:
                if (b[0] == b[1]).item() and (b[1] == b[2]).item() and (b[0] == id).item():
                    return True
                if (b[1] == b[4]).item() and (b[4] == b[7]).item() and (b[1] == id).item():
                    return True
            case 2:
                if (b[0] == b[1]).item() and (b[1] == b[2]).item() and (b[0] == id).item():
                    return True
                if (b[2] == b[5]).item() and (b[5] == b[8]).item() and (b[2] == id).item():
                    return True
                if (b[2] == b[4]).item() and (b[4] == b[6]).item() and (b[2] == id).item():
                    return True
            case 3:
                if (b[3] == b[4]).item() and (b[4] == b[5]).item() and (b[3] == id).item():
                    return True
                if (b[0] == b[3]).item() and (b[3] == b[6]).item() and (b[0] == id).item():
                    return True
            case 4:
                if (b[3] == b[4]).item() and (b[4] == b[5]).item() and (b[3] == id).item():
                    return True
                if (b[1] == b[4]).item() and (b[4] == b[7]).item() and (b[1] == id).item():
                    return True
                if (b[0] == b[4]).item() and (b[4] == b[8]).item() and (b[0] == id).item():
                    return True
                if (b[2] == b[4]).item() and (b[4] == b[6]).item() and (b[2] == id).item():
                    return True
            case 5:
                if (b[3] == b[4]).item() and (b[4] == b[5]).item() and (b[3] == id).item():
                    return True
                if (b[2] == b[5]).item() and (b[5] == b[8]).item() and (b[2] == id).item():
                    return True
            case 6:
                if (b[6] == b[7]).item() and (b[7] == b[8]).item() and (b[6] == id).item():
                    return True
                if (b[0] == b[3]).item() and (b[3] == b[6]).item() and (b[0] == id).item():
                    return True
                if (b[2] == b[4]).item() and (b[4] == b[6]).item() and (b[2] == id).item():
                    return True
            case 7:
                if (b[6] == b[7]).item() and (b[7] == b[8]).item() and (b[6] == id).item():
                    return True
                if (b[1] == b[4]).item() and (b[4] == b[7]).item() and (b[1] == id).item():
                    return True
            case 8:
                if (b[6] == b[7]).item() and (b[7] == b[8]).item() and (b[6] == id).item():
                    return True
                if (b[2] == b[5]).item() and (b[5] == b[8]).item() and (b[2] == id).item():
                    return True
                if (b[0] == b[4]).item() and (b[4] == b[8]).item() and (b[0] == id).item():
                    return True
        return False

    def play(self, id, val):
        if self._play(id, val):
            if self._checkWin(id,val):
                return True,"win"
            if self.sum == 9:
                return True,"draw"
            return False,"none"
        return True,"invalid"

    def clear(self):
        self.board = torch.zeros(9, dtype=torch.float32)
        self.sum = 0
        self.game = []

    @property
    def board_state(self):
        return self.board

    @property
    def game_state(self):
        return self.game

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(2*input_size, hidden_sizes[0])
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.l4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.l5 = nn.Linear(hidden_sizes[3], num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        xin = torch.stack((self.relu(x), self.relu(-x)), dim=2).flatten(start_dim=1)
        logits = torch.zeros_like(x)  # Create a tensor of all 0s
        logits = logits.masked_fill(x != 0, float('-inf'))
        out = self.l1(xin)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.l4(out)
        out = self.relu(out)
        out = self.l5(out)
        out = out + logits
        out = self.softmax(out)
        return out

def load_model(model_path='model.pth', input_size=9, hidden_sizes=[256, 512, 64, 32], num_classes=9):
    """
    Load a trained model from a .pth file.
    
    Args:
        model_path (str): Path to the model file. Default is 'model.pth'
        input_size (int): Input size for the neural network. Default is 9
        hidden_sizes (list): List of hidden layer sizes. Default is [256, 512, 64, 32]
        num_classes (int): Number of output classes. Default is 9
    
    Returns:
        NeuralNet: Loaded model in evaluation mode
    """
    model = NeuralNet(input_size, hidden_sizes, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model
