import torch
from utils.datasets_utils import get_mnist_loaders, get_cifar_loaders
from models.fc_models import FullyConnectedModel
from models.cnn_models import SimpleCNN, CNNwithResidual, RegularizedResidualCNN, CNN_1x1_3x3
from utils.visualization_utils import get_experiment_data


# 2.1 - Влияние размера ядра свертки
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, test_loader = get_mnist_loaders(batch_size=64)

# - 3x3 ядра
cnn3x3 = SimpleCNN(input_channels=1, num_classes=10).to(device)

# - 5x5 ядра
cnn5x5 = SimpleCNN(input_channels=1, num_classes=10).to(device)

# - 7x7 ядра
cnn7x7 = SimpleCNN(input_channels=1, num_classes=10).to(device)

# - Комбинация разных размеров (1x1 + 3x3)
cnn1x13x3 = CNN_1x1_3x3(input_channels=1, num_classes=10).to(device)

#get_experiment_data(cnn1x13x3, train_loader, test_loader, device, "CNN7x7")


# 2.2 - Влияние глубины CNN
# - Неглубокая CNN (2 conv слоя)
cnn2conv = SimpleCNN(input_channels=1, num_classes=10).to(device)

# - Средняя CNN (4 conv слоя)
cnn4conv = SimpleCNN(input_channels=1, num_classes=10).to(device)

# - Глубокая CNN (6+ conv слоев)
cnn6conv = SimpleCNN(input_channels=1, num_classes=10).to(device)

# - CNN с Residual связями
resiudalCNN = CNNwithResidual(input_channels=1, num_classes=10).to(device)

get_experiment_data(cnn6conv, train_loader, test_loader, device, "CNN with 2 conv")
