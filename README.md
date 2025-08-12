## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

<img width="763" height="556" alt="image" src="https://github.com/user-attachments/assets/7d2e0585-3d4e-4533-9345-58916e0ae274" />


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:
### Register Number:
```python
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,8)
        self.fc2=nn.Linear(8,10)
        self.fc3=nn.Linear(10,1)
        self.history={'loss':[]}

  def forward(self,x):
        x=torch.relu(self.fc1(x))
        x=self.fc2(x)
        x=self.fc3(x)
        return x


# Initialize the Model, Loss Function, and Optimizer
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(ai_brain.parameters(), lr=0.001)


def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = ai_brain(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')


```
## Dataset Information


<img width="655" height="339" alt="image" src="https://github.com/user-attachments/assets/34da43b5-0983-4da4-bc5c-af2ab5560e7e" />


## OUTPUT

### Training Loss Vs Iteration Plot

<img width="696" height="505" alt="image" src="https://github.com/user-attachments/assets/230befc6-c9c9-4d22-a872-17572c7ac146" />



### New Sample Data Prediction

<img width="511" height="44" alt="image" src="https://github.com/user-attachments/assets/13e4b00d-940f-47a4-b68a-0b081406b576" />


## RESULT

Include your result here
