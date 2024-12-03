from layers.dense import Dense

inputs = [1, 0, 2, 1]

layer1 = Dense(len(inputs), 5)
print(layer1.forward(inputs))