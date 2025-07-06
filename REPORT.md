# Задание 1.1 - Сравнение на MNIST

Для выполнения задания я использовал 3 вида моделей: полносвязную сеть, обычную CNN и CNN с Residual Block. Для визуализации и извлечения информации использовал функцию `get_experiment_data`, которая запускала обучение модели, а также находила кол-во параметров, время обучения, время инференса, а также создавала кривые обучения.

## Итоговый результат в таблице (данные представлены для 5 эпох):

| Модель      | TrainLoss | TrainAcc | TestLoss | TestAcc | Время обучения (с) | Время инференса (с) | Кол-во параметров |
|-------------|-----------|----------|----------|---------|--------------------|---------------------|-------------------|
| FCN         |  0.0498   |  0.9858  |  0.0725  | 0.9802  |         57         |        0.97         |      1,494,154    |
| SimpleCNN   |  0.0269   |  0.9913  |  0.0295  | 0.9911  |         56         |        0.90         |      421,642      |
| ResidualCNN |  0.0220   |  0.9928  |  0.0312  | 0.9908  |         73         |        0.98         |      160,906      |

## Графики:
FCN модель
![FCN модель](https://github.com/Kartowe4ka/DataScienceFourthHomework/blob/main/plots/1.1(FCN).png)
SimpleCNN модель
![FCN модель](https://github.com/Kartowe4ka/DataScienceFourthHomework/blob/main/plots/1.1(SimpleCNN).png)
ResidualCNN модель
![FCN модель](https://github.com/Kartowe4ka/DataScienceFourthHomework/blob/main/plots/1.1(ResidualCNN).png)

___

# Задание 1.2 - Сравнение на CIFAR-10

Для выполнения задания я использовал 3 вида моделей: полносвязную сеть, CNN с Residual Block и CNN с Residual Block и регуляризацией. Для визуализации и извлечения информации использовал функцию `get_experiment_data`, которая запускала обучение модели, а также находила кол-во параметров, время обучения, время инференса, а также создавала кривые обучения.

## Итоговый результат в таблице (данные представлены для 10 эпох):

| Модель         | TrainLoss | TrainAcc | TestLoss | TestAcc | Время обучения (с) | Время инференса (с) | Кол-во параметров |
|----------------|-----------|----------|----------|---------|--------------------|---------------------|-------------------|
| FCN            |  0.9160   |  0.6726  |  1.4659  | 0.5265  |         162        |        2.69         |     9,050,378     |
| ResidualCNN    |  0.2703   |  0.9069  |  0.5510  | 0.8249  |         134        |        1.27         |       161,482     |
| RegResidualCNN |  0.7458   |  0.7374  |  0.6596  | 0.7719  |         173        |        1.43         |       416,202     |

## Графики:
FCN модель
![FCN модель](https://github.com/Kartowe4ka/DataScienceFourthHomework/blob/main/plots/2.1(FCN).png)
ResidualCNN модель
![ResidualCNN модель](https://github.com/Kartowe4ka/DataScienceFourthHomework/blob/main/plots/2.1(ResidualCNN).png)
RegResidualCNN модель
![RegresidualCNN модель](https://github.com/Kartowe4ka/DataScienceFourthHomework/blob/main/plots/2.1(RegResidualCNN).png)

По графикам видно, что первые две модели подвергаются переобучению, так как кривые для тестовой выборки со временем начинают идти в противоположном направлении

___

# Задание 2.1 - Влияние размера ядра свертки

Для ядер 3x3, 5x5 и 7x7 я использовал одинаковый класс, внутри которой просто менял kernel_size и padding отдельно для каждого теста. Для комбинации разных размеров я создал отдельный класс CNN_1x1_3x3. Для визуализации и извлечения информации использовал функцию `get_experiment_data`, которая запускала обучение модели, а также находила кол-во параметров, время обучения, время инференса, а также создавала кривые обучения.

## Итоговый результат в таблице (данные представлены для 10 эпох):

| Модель      | TrainLoss | TrainAcc | TestLoss | TestAcc | Время обучения (с) | Время инференса (с) | Кол-во параметров |
|-------------|-----------|----------|----------|---------|--------------------|---------------------|-------------------|
| CNN 3x3     |  0.0128   |  0.9958  |  0.0335  | 0.9911  |         123        |        0.92         |       421,642     |
| CNN 5x5     |  0.0125   |  0.9960  |  0.0326  | 0.9920  |         99         |        1.09         |       454,922     |
| CNN 7x7     |  0.0132   |  0.9958  |  0.0349  | 0.9911  |         106        |        1.09         |       504,842     |
| CNN 1x1 3x3 |  0.0141   |  0.9951  |  0.0290  | 0.9931  |         103        |        0.99         |       427,050     |

## Графики:
CNN 3x3 модель
![FCN модель](https://github.com/Kartowe4ka/DataScienceFourthHomework/blob/main/plots/2.1(CNN3x3).png)
CNN 5x5 модель
![ResidualCNN модель](https://github.com/Kartowe4ka/DataScienceFourthHomework/blob/main/plots/2.1(CNN5x5).png)
CNN 7x7 модель
![RegresidualCNN модель](https://github.com/Kartowe4ka/DataScienceFourthHomework/blob/main/plots/2.1(CNN7x7).png)
CNN 1x1 + 3x3 модель
![RegresidualCNN модель](https://github.com/Kartowe4ka/DataScienceFourthHomework/blob/main/plots/2.1(CNN1x13x3).png)

___

# Задание 2.2 - Влияние глубины CNN

Для первых трех моделей я использовал один класс, в котором изменял кол-во conv слоев для каждого отдельного тестирования. Для последней модели использовал класс CNNwithResidual. Для визуализации и извлечения информации использовал функцию `get_experiment_data`, которая запускала обучение модели, а также находила кол-во параметров, время обучения, время инференса, а также создавала кривые обучения.

## Итоговый результат в таблице (данные представлены для 10 эпох):
| Модель      | TrainLoss | TrainAcc | TestLoss | TestAcc | Время обучения (с) | Время инференса (с) | Кол-во параметров |
|-------------|-----------|----------|----------|---------|--------------------|---------------------|-------------------|
| CNN 2 conv  |  0.0107   |  0.9963  |  0.0304  | 0.9924  |         172        |        2.12         |       421,642     |
| CNN 4 conv  |  0.0141   |  0.9953  |  0.0259  | 0.9927  |         223        |        1.92         |       260,746     |
| CNN 6 conv  |  0.0184   |  0.9941  |  0.0188  | 0.9941  |         641        |        3.19         |      1,135,754    |
| ResidualCNN |  0.0126   |  0.9958  |  0.0175  | 0.9944  |         161        |        1.13         |       160,906     |

## Графики
CNN 2 conv
![FCN модель](https://github.com/Kartowe4ka/DataScienceFourthHomework/blob/main/2.1(CNN3x3).png)
CNN 4 conv
![ResidualCNN модель](https://github.com/Kartowe4ka/DataScienceFourthHomework/blob/main/2.1(CNN5x5).png)
CNN 6 conv
![RegresidualCNN модель](https://github.com/Kartowe4ka/DataScienceFourthHomework/blob/main/2.1(CNN7x7).png)
ResidualCNN
![RegresidualCNN модель](https://github.com/Kartowe4ka/DataScienceFourthHomework/blob/main/2.1(CNN1x13x3).png)




