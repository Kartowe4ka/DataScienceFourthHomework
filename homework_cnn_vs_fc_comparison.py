import torch
from utils.datasets_utils import get_mnist_loaders, get_cifar_loaders
from models.fc_models import FullyConnectedModel
from models.cnn_models import SimpleCNN, CNNwithResidual, RegularizedResidualCNN
from utils.visualization_utils import get_experiment_data


# 1.1 - Сравнение на MNIST
# Сравните производительность на MNIST:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, test_loader = get_cifar_loaders(batch_size=64)

# - Полносвязная сеть (3-4 слоя)
configMNIST = {
        "input_size": 784,
        "num_classes": 10,
        "layers": [
            {"type": "linear", "size": 1024},
            {"type": "relu"},
            {"type": "linear", "size": 512},
            {"type": "relu"},
            {"type": "linear", "size": 256},
            {"type": "relu"},
            {"type": "linear", "size": 128},
            {"type": "relu"}
        ]
    }
fcnMNIST = FullyConnectedModel(**configMNIST).to(device)

# - Простая CNN (2-3 conv слоя)
simple_cnnMNIST = SimpleCNN(input_channels=1, num_classes=10).to(device)

# - CNN с Residual Block
residual_cnnMNIST = CNNwithResidual(input_channels=1, num_classes=10).to(device)

# Для каждого варианта:
# - Обучите модель с одинаковыми гиперпараметрами
# - Сравните точность на train и test множествах
# - Измерьте время обучения и инференса
# - Визуализируйте кривые обучения
# - Проанализируйте количество параметров
get_experiment_data(residual_cnnMNIST, train_loader, test_loader, device, 'ResidualCNN')


# 1.2 - Сравнение на CIFAR-10
# Сравните производительность на CIFAR-10:
# - Полносвязная сеть (глубокая)
configCIFAR = {
    "input_size": 3072,
    "num_classes": 10,
    "layers": [
        {"type": "linear", "size": 2048},
        {"type": "relu"},
        {"type": "linear", "size": 1024},
        {"type": "relu"},
        {"type": "linear", "size": 512},
        {"type": "relu"},
        {"type": "linear", "size": 256},
        {"type": "relu"}
    ]
}
fcnCIFAR = FullyConnectedModel(**configCIFAR).to(device)

# - CNN с Residual блоками
residual_cnnCIFAR = CNNwithResidual(input_channels=3, num_classes=10).to(device)

# - CNN с регуляризацией и Residual блоками
reg_residual_cnnCIFAR = RegularizedResidualCNN(input_channels=3, num_classes=10, dropout_rate=0.3).to(device)

# Для каждого варианта:
# - Обучите модель с одинаковыми гиперпараметрами
# - Сравните точность и время обучения
# - Проанализируйте переобучение
# - Визуализируйте confusion matrix
# - Исследуйте градиенты (gradient flow)
get_experiment_data(reg_residual_cnnCIFAR, train_loader, test_loader, device, "RegResidual")



