import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the data from the pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = data_dict['data']
labels = data_dict['labels']

# Debugging: Inspect the data structure
print(f"Total number of samples: {len(data)}")
lengths = [len(sample) for sample in data]
print(f"Unique lengths of samples: {set(lengths)}")

# OPTION 1: Remove Inconsistent Samples
# Find the most common length
most_common_length = max(set(lengths), key=lengths.count)

# Remove samples that do not match the most common length
filtered_data = [sample for sample, length in zip(data, lengths) if length == most_common_length]
filtered_labels = [label for label, length in zip(labels, lengths) if length == most_common_length]

# Convert to NumPy arrays
data = np.asarray(filtered_data)
labels = np.asarray(filtered_labels)

print(f"Data shape after filtering: {data.shape}")
print(f"Labels shape: {labels.shape}")

# OPTION 2: Pad Inconsistent Samples
# Uncomment the following code if you prefer padding the samples

# max_length = max(len(sample) for sample in data)
# padded_data = [sample + [0] * (max_length - len(sample)) for sample in data]
# data = np.asarray(padded_data)
# labels = np.asarray(labels)

# print(f"Data shape after padding: {data.shape}")
# print(f"Labels shape: {labels.shape}")

# Encode labels if they are not numeric
if labels.dtype == 'object':  # Check if labels are strings
    le = LabelEncoder()
    labels = le.fit_transform(labels)

# Ensure the labels array is 1D
labels = np.ravel(labels)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize and train the RandomForestClassifier
model = RandomForestClassifier()

# Debugging: Ensure x_train and y_train have the correct shapes
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")

model.fit(x_train, y_train)

# Make predictions on the test set
y_predict = model.predict(x_test)

# Calculate the accuracy of the model
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score * 100))

# Save the trained model to a file
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
