# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
<img width="615" height="491" alt="nn" src="https://github.com/user-attachments/assets/01fc80b1-cbd4-4dc4-bbb5-9069425198fe" />

## DESIGN STEPS

### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

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

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: ALLEN JOVETH P

### Register Number: 212223240007

```python

class NeuralNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(1,8)
    self.fc2 = nn.Linear(8,10)
    self.fc3 = nn.Linear(10,1)
    self.relu = nn.ReLU() # Added ReLU activation
    self.history = {'loss':[]}

  def forward(self,x):
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.fc3(x)
    return x
    
# initailze the model
lig = NeuralNet()
criterion = nn.MSELoss()
optimzer = optim.RMSprop(lig.parameters(),lr=0.0001)

def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
      optimizer.zero_grad() # Corrected typo from optimzer to optimizer
      loss = criterion(ai_brain(X_train),y_train)
      loss.backward()

      optimizer.step()
      lig.history['loss'].append(loss.item())


      if epoch % 200 == 0:
          print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
```

### Dataset Information
<img width="661" height="118" alt="image" src="https://github.com/user-attachments/assets/698e6808-6f68-4fbb-8752-48a0bea217c0" />

### OUTPUT

### Training Loss Vs Iteration Plot
<img width="674" height="507" alt="image" src="https://github.com/user-attachments/assets/2c93b940-c456-449e-b009-b127a2150c7e" />




### New Sample Data Prediction
<img width="282" height="30" alt="image" src="https://github.com/user-attachments/assets/1faec863-8df0-44c8-a841-fa59467e3281" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
