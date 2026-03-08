# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
The problem statement involves building and training a Convolutional Neural Network (CNN) to classify images of fashion items. Specifically, it uses the Fashion-MNIST dataset, which is a common benchmark for image classification tasks.

Dataset Details: Name: Fashion-MNIST Type: Grayscale images of clothing articles. Image Size: Each image is 28x28 pixels. Classes: There are 10 distinct classes of fashion items (e.g., T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot). Training Set Size: 60,000 images. Test Set Size: 10,000 images.

## DESIGN STEPS
STEP 1:
Data Loading and Preprocessing: The Fashion-MNIST dataset is loaded, images are transformed (converted to tensors and normalized), and DataLoaders are created for efficient batch processing.

STEP 2:
CNN Model Definition: A CNNClassifier class is defined, outlining the architecture of the convolutional neural network, including convolutional layers, pooling layers, and fully connected layers.

STEP 3:
Model Setup: An instance of the CNNClassifier model is created, the CrossEntropyLoss is chosen as the loss function, and the Adam optimizer is configured.

STEP 4:
Model Training: The train_model function is executed to train the CNN model on the training dataset over multiple epochs, optimizing the model's parameters to minimize the loss.

STEP 5:
Model Evaluation and Prediction: The test_model function evaluates the trained model's performance on the unseen test dataset, providing metrics like accuracy, a confusion matrix, and a classification report. Additionally, the predict_image function demonstrates how to use the trained model to make predictions on individual images.

## PROGRAM

### Name:Rithika R

### Register Number:212224240136

```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding = 1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding = 1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1 = nn.Linear(128*3*3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64,10)







    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool((torch.relu(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

```

### OUTPUT

## Training Loss per Epoch

<img width="449" height="215" alt="image" src="https://github.com/user-attachments/assets/414a9b19-6888-4438-a37d-c92d72dc9956" />

## Confusion Matrix

<img width="709" height="608" alt="image" src="https://github.com/user-attachments/assets/ca92985a-af00-4c71-a877-fce2e739daca" />


## Classification Report
<img width="674" height="404" alt="image" src="https://github.com/user-attachments/assets/111004ee-4a5c-454d-916c-eddf60cc77ca" />


### New Sample Data Prediction
<img width="389" height="432" alt="image" src="https://github.com/user-attachments/assets/8fe1650a-bb29-4960-b3cd-4c1c55fa4225" />

## RESULT
The CNN model was succesfully created
