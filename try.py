from torchvision import datasets

# Load the MNIST dataset
mnist = datasets.MNIST(train=True, root='./data', download=True)

print(mnist)