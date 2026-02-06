import math

# Activation functions
def relu(x):
    return max(0, x)

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# Fixed weights (as per your PDF)
W1 = 0.8   # w1 to w9 all same = 0.8
W2 = 1.2   # w10 to w13
W3 = -2.1  # w14 to w17

def forward_pass(x1, x2, x3):
    print(f"\nInput: x1={x1}, x2={x2}, x3={x3}")

    # Layer 1 (3 neurons)
    a1 = relu(x1*W1 + x2*W1 + x3*W1)
    a2 = relu(x1*W1 + x2*W1 + x3*W1)
    a3 = relu(x1*W1 + x2*W1 + x3*W1)

    print("a1 =", a1)
    print("a2 =", a2)
    print("a3 =", a3)

    # Layer 2 (2 neurons)
    a4 = relu(a1*W2 + a2*W2 + a3*W3)
    a5 = relu(a1*W2 + a2*W2 + a3*W3)

    print("a4 =", a4)
    print("a5 =", a5)

    # Output layer
    z = a4*W3 + a5*W3
    y = sigmoid(z)

    print("z =", z)
    print("Sigmoid Output =", y)

    # Final class (threshold 0.5)
    y_hat = 1 if y >= 0.5 else 0
    print("Final Output (ŷ) =", y_hat)

# Test cases from PDF
forward_pass(1, 2, 3)
forward_pass(-4, -1, 3)
forward_pass(4, 5, 6)