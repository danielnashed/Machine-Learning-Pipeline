import pickle

# Assuming you have a pickled file named 'data.pickle'
with open('KNN.pickle', 'rb') as f:
    data_test = pickle.load(f)

print(data_test)