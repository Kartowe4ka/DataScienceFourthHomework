import time
from utils.training_utils import train_model
import torch
import matplotlib.pyplot as plt

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 4))

    ax1.plot(history['train_losses'], label="Train Loss")
    ax1.plot(history['test_losses'], label="Test Loss")
    ax1.legend()

    ax2.plot(history['train_accs'], label="Train Acc")
    ax2.plot(history['test_accs'], label="Test Acc")
    ax2.legend()

    plt.tight_layout()
    plt.show()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

def compare_models(fc_history, cnn_history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(fc_history['test_accs'], label="FC Network", marker='o')
    ax1.plot(cnn_history['test_accs'], label="CNN", marker='s')
    ax1.set_title('Test Accuracy Comparison')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(fc_history['test_losses'], label="FC Network", marker='o')
    ax2.plot(cnn_history['test_losses'], label="CNN", marker='s')
    ax2.set_title('Test Losses Comparison')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def get_experiment_data(model, train_loader, test_loader, device, name):
    # Обучение
    start_train = time.time()
    history = train_model(model, train_loader, test_loader, epochs=10, device=str(device))
    end_train = time.time()
    train = end_train - start_train

    # Инференс
    model.eval()
    start_infer = time.time()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
    end_infer = time.time()
    infer = end_infer - start_infer

    # Подсчёт параметров
    params = count_parameters(model)

    print(f"{name}: Кол-во параметров - {params}\nВремя обучения - {train:.2f} секунд \nВремя инференса - {infer:.2f} секунд")
    plot_training_history(history)






