# Задание 1.1 - Сравнение на MNIST

Для выполнения задания я использовал 3 вида моделей: полносвязную сеть, обычную CNN и CNN с Residual Block. Для визуализации и извлечения информации использовал функцию `get_experiment_data`, которая запускала обучение модели, а также находила кол-во параметров, время обучения, время инференса, а также создавала кривые обучения.

## Итоговый результат в таблице:

| Модель      | TrainLoss | TrainAcc | TestLoss | TestAcc | Время обучения (с) | Время инференса (с) | Кол-во параметров |
|-------------|-----------|----------|----------|---------|--------------------|---------------------|-------------------|
| FCN         |  0.0498   |  0.9858  |  0.0725  | 0.9802  |         57         |        0.97         |      1,494,154    |
| SimpleCNN   |  0.0269   |  0.9913  |  0.0295  | 0.9911  |         56         |        0.90         |      421,642      |
| ResidualCNN |  0.0220   |  0.9928  |  0.0312  | 0.9908  |         73         |        0.98         |      160,906      |

## Графики:
FCN модель:
![FCN модель](https://github.com/Kartowe4ka/DataScienceFourthHomework/blob/main/1.1(FCN).png)
SimpleCNN модель
![FCN модель](https://github.com/Kartowe4ka/DataScienceFourthHomework/blob/main/1.1(SimpleCNN).png)
ResidualCNN модель
![FCN модель](https://github.com/Kartowe4ka/DataScienceFourthHomework/blob/main/1.1(ResidualCNN).png)
