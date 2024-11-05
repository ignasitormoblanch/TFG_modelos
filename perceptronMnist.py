import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset



ds = load_dataset("ylecun/mnist")
#.view(-1, 28*28) / 255.0 es para normalizarlo ya que las dimensiones de entrada en el DataLoader y
# en la capa de entrada del modelo aún no coinciden. El modelo espera una entrada con 784 características
# (28x28), pero está recibiendo datos con dimensiones (896, 28)
conjuntotrainimages=torch.LongTensor(np.array(ds['train'][:50000]['image'])).view(-1, 28*28) / 255.0
train_labels = torch.LongTensor(ds['train'][:50000]['label'])

conjuntovalimages=torch.LongTensor(np.array(ds['train'][50000:]['image'])).view(-1, 28*28) / 255.0
conjuntoval_labels = torch.LongTensor(ds['train'][50000:]['label'])

conjuntotest=torch.LongTensor(np.array(ds['test'][:10000]['image'])).view(-1, 28*28) / 255.0
conjuntotest_labels = torch.LongTensor(ds['test'][:10000]['label'])


train_dataset = TensorDataset(conjuntotrainimages, train_labels)
val_dataset = TensorDataset(conjuntovalimages, conjuntoval_labels)
test_dataset = TensorDataset(conjuntotest, conjuntotest_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)



# Definimos el modelo de Perceptrón Multicapa (MLP)
#nn.Module es la clase base en PyTorch para todos los modelos de redes neuronales
class Perceptron(nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        # Capa de entrada (784 pq las img son de 28 pixeles 28*28=784 -> 128)
        self.fc1 = nn.Linear(784, 128)
        # Primera capa intermedia (128 -> 64)
        self.fc2 = nn.Linear(128, 64)
        # Segunda capa intermedia (64 -> 32)
        self.fc3 = nn.Linear(64, 32)
        # Capa de salida (32 -> 10)
        self.fc4 = nn.Linear(32, 10)

    def forward(self, x):
        #aqiui vemos como los datos pasan a traves de las capas
        x = F.relu(self.fc1(x))   # Capa de entrada con ReLU
        x = F.relu(self.fc2(x))   # Primera capa intermedia con ReLU
        x = F.relu(self.fc3(x))   # Segunda capa intermedia con ReLU
        #hacer un softmax para hacer la funcion de probabilidades pero croossentropy
        #en pytorch ya lo hace asi q no hace falta
        x = self.fc4(x)           # Capa de salida sin activación (para aplicar CrossEntropyLoss)
        return x

# Inicializamos el modelo, la función de pérdida y el optimizador
#crea el modelo
model = Perceptron()


#criterion define la función de pérdida para el modelo, en este caso CrossEntropyLoss
criterion = nn.CrossEntropyLoss()
#El optimizador se encarga de actualizar los parámetros de la red para reducir la pérdida.
# Aquí estamos usando SGD (Stochastic Gradient Descent) con una tasa de aprendizaje (lr) de 0.01
#esto es el backpropagation
optimizer = optim.SGD(model.parameters(), lr=0.01)

#vamos a entrenar 1 epoca



listalossval=[]
listalosstrain=[]
listaaccuracytrain=[]
listaaccuracyval=[]

n_epoch=1000

for epoch in range(n_epoch):
    sum=0
    sum2=0
    train_correct_train = 0
    total_train = 0
    train_correct_val=0
    total_val=0
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Propagación hacia adelante
        outputs = model(images)
        loss = criterion(outputs, labels)
        sum+=loss.item()
        _, predicted = torch.max(outputs, 1)  # Obtiene el índice de la predicción más alta
        # Suma los aciertos
        train_correct_train += (predicted == labels).sum().item()  # el item es pq el sum te devuelve Tensor(25) y con el item pasa a ser 25

        total_train += labels.size(0)  # Cuenta el total de ejemplos procesados

        #la backpropagation hay q hacerla en cada bach o cuando acaba la primera época?
        # Backpropagación y optimización
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    for batch_idx2, (images2, labels2) in enumerate(val_loader):
        # Propagación hacia adelante
        outputs2 = model(images2)
        loss2 = criterion(outputs2, labels2)
        sum2+=loss2.item()
        _, predicted2 = torch.max(outputs2, 1)
        train_correct_val += (predicted2 == labels2).sum().item()
        total_val += labels2.size(0)


    print(f'La media de pérdida en la epoca {epoch} de train ha sido de {(sum/len(train_loader)):.4f}')
    print(f'La media de pérdida en la epoca {epoch} de validación ha sido de {(sum2/len(val_loader)):.4f}')
    train_accuracy = 100 * train_correct_train / total_train
    val_accuracy = 100 * train_correct_val / total_val

    if(epoch%5==0):

        listalosstrain.append(sum / len(train_loader))
        listalossval.append(sum2 / len(val_loader))
        listaaccuracytrain.append(train_accuracy)
        listaaccuracyval.append(val_accuracy)

train_correct_test=0
total_test=0
for batch_idx3, (images3, labels3) in enumerate(test_loader):
    # Propagación hacia adelante
    outputs3 = model(images3)
    _, predicted3 = torch.max(outputs3, 1)
    train_correct_test += (predicted3 == labels3).sum().item()
    total_test += labels3.size(0)

test_accuracy = 100 * train_correct_test / total_test
print(f'El accuracy de esto es: {test_accuracy}')

# Crear la figura con dos subplots, uno para Loss y otro para Accuracy
plt.figure(figsize=(12, 10))

# Subplot 1: Loss de entrenamiento y validación
plt.subplot(1, 2, 1)
plt.plot(listalosstrain, marker='o', linestyle='-', color='b', label='Train Loss')
plt.plot(listalossval, marker='o', linestyle='-', color='r', label='Validation Loss')
plt.title('Loss durante el Entrenamiento y Validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)

# Subplot 2: Accuracy de entrenamiento y validación
plt.subplot(1, 2, 2)
plt.plot(listaaccuracytrain, marker='o', linestyle='-', color='b', label='Train Accuracy')
plt.plot(listaaccuracyval, marker='o', linestyle='-', color='r', label='Validation Accuracy')
plt.title('Accuracy durante el Entrenamiento y Validación')
plt.xlabel('Épocas')
plt.ylabel('Exactitud')
plt.legend()
plt.grid(True)

# Mostrar ambos gráficos en la misma figura
plt.tight_layout()  # Para ajustar el espacio entre subplots
plt.show()


plt.figure(figsize=(12, 10))

# Subplot 1: Loss de entrenamiento y validación
plt.subplot(2, 1, 1)
plt.plot(listalosstrain, marker='o', linestyle='-', color='b', label='Train Loss')
plt.plot(listalossval, marker='o', linestyle='-', color='r', label='Validation Loss')
plt.title('Loss durante el Entrenamiento y Validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)

# Subplot 2: Accuracy de entrenamiento y validación
plt.subplot(2, 1, 2)
plt.plot(listaaccuracytrain, marker='o', linestyle='-', color='b', label='Train Accuracy')
plt.plot(listaaccuracyval, marker='o', linestyle='-', color='r', label='Validation Accuracy')
plt.title('Accuracy durante el Entrenamiento y Validación')
plt.xlabel('Épocas')
plt.ylabel('Exactitud')
plt.legend()
plt.grid(True)

# Mostrar ambos gráficos en la misma figura
plt.tight_layout()  # Para ajustar el espacio entre subplots
plt.show()