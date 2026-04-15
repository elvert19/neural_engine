import neural_engine
import numpy as np

def test_xor():
    print("--- Setting up XOR Test ---")
    
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ], dtype=np.float64)

    y = np.array([
        [0.0],
        [1.0],
        [1.0],
        [0.0]
    ], dtype=np.float64)

    model = neural_engine.Sequential()
    
    model.add_dense(2, 4) 
    model.add_tanh()      
    
    model.add_dense(4, 1) 
    model.add_sigmoid()   

    model.set_optimizer("adam", 0.1)
    model.set_loss("mse")           

    print("\n--- Starting Training ---")
    epochs = 1000
    loss_history = model.train(X, y, epochs, 4)

    print(f"\nFinal Loss: {loss_history[-1][0]:.6f}")

    print("\n--- Predictions ---")
    predictions = model.predict(X)
    
    for input_val, pred, target in zip(X, predictions, y):
        print(f"Input: {input_val} | Target: {target} | Prediction: {pred[0]:.4f}")

if __name__ == "__main__":
    test_xor()