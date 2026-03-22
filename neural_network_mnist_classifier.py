import sys
import csv
import numpy as np
# fix seed 
np.random.seed(42)

class Fully_Connected_Layer:
    def __init__(self, learning_rate):
        self.InputDim = 784
        self.HiddenDim = 128
        self.OutputDim = 10
        self.learning_rate = learning_rate
        
        '''Weight Initialization'''
    
        self.W1 = np.random.randn(self.InputDim, self.HiddenDim) / np.sqrt(self.InputDim)
        self.b1 = np.zeros((1, self.HiddenDim))
        
        self.W2 = np.random.randn(self.HiddenDim, self.OutputDim) / np.sqrt(self.HiddenDim)
        self.b2 = np.zeros((1, self.OutputDim))
        
    def sigmoid(self, z):
        # prevent overflow 
        z = np.clip(z, -100, 100)
        return 1 / (1 + np.exp(-z))
    
    def Forward(self, Input):
        # Layer 1
        self.z1 = np.dot(Input, self.W1) + self.b1
        self.h = self.sigmoid(self.z1)
        
        # Layer 2
        self.z2 = np.dot(self.h, self.W2) + self.b2
        self.o = self.sigmoid(self.z2)
        return self.o
    
    def Backward(self, Input, Label, Output):
        m = Input.shape[0]
        # --- Gradient Calculation ---
        # 1. Output Layer Error
        error_output = Output - Label
        delta_out = error_output * (Output * (1 - Output))
        
        dW2 = np.dot(self.h.T, delta_out) / m
        db2 = np.sum(delta_out, axis=0, keepdims=True) / m
        
        # 2. Hidden Layer Error
        error_hidden = np.dot(delta_out, self.W2.T)
        delta_hidden = error_hidden * (self.h * (1 - self.h))
        
        dW1 = np.dot(Input.T, delta_hidden) / m
        db1 = np.sum(delta_hidden, axis=0, keepdims=True) / m
        
        # Update
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
    
    def Train(self, Input, Label):
        Output = self.Forward(Input)
        self.Backward(Input, Label, Output)

def load_data(file_path):
    data = []
    labels = []
    
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            try:
                vals = np.array([float(x) for x in row])
                
                # make label
                labels.append(int(vals[-1]))
                
                # make data
                data.append(vals[:-1]) 
                
            except ValueError:
                continue
                
    return np.array(data), np.array(labels)

def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]

def get_accuracy(network, x, y_true):
    pred = network.Forward(x)
    pred_labels = np.argmax(pred, axis=1)
    return np.mean(pred_labels == y_true)

def main():
    if len(sys.argv) < 3:
        print("Usage: python hw3_2.py training.csv testing.csv")
        sys.exit(1)

    train_path = sys.argv[1]
    test_path = sys.argv[2]
    
    # 1. data load 
    train_x, train_y = load_data(train_path)
    test_x, test_y = load_data(test_path)
    
    # One-hot encoding
    train_y_enc = one_hot_encode(train_y)
    
    # 2. hyperparameter settings
    learning_rate = 0.2
    iterations = 5000
    batch_size = 64
    
    # 3. model initialize 
    nn = Fully_Connected_Layer(learning_rate)
    
    # 4. Training
    num_train = train_x.shape[0]
    for i in range(iterations):
        # Mini-batch
        idx = np.random.choice(num_train, batch_size)
        x_batch = train_x[idx]
        y_batch = train_y_enc[idx]
        
        nn.Train(x_batch, y_batch)
        
    # print output 
    train_acc = get_accuracy(nn, train_x, train_y)
    test_acc = get_accuracy(nn, test_x, test_y)
    
    print(f"{train_acc:.3f}")
    print(f"{test_acc:.3f}")
    print(iterations)
    print(learning_rate)

if __name__ == "__main__":
    main()
