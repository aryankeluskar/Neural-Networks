import numpy as np

# Frequency data extracted from the image
wins = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
frequency = np.array([0, 0, 2, 0, 3, 2, 3, 6, 2, 1, 6, 3, 2, 2, 0, 0, 0])

# make an array of the frequency data
data = []
for i in range(len(wins)):
    for j in range(frequency[i]):
        data.append(wins[i])

print(data)
print(len(data))

print(np.sum(data))
print(np.mean(data))
print(np.std(data))

# # Calculate the standard deviation
# std = np.std(frequency)
# print(std)

# # Calculate the mean
# mean = np.mean(frequency)
# print(mean)