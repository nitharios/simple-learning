from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Load our data into two Python lists
with open("clickbait.txt") as f:
  lines = f.read().strip().split("\n")
  lines = [line.split("\t") for line in lines]
headlines, labels = zip(*lines)

# Break dataset into test and train sets
train_headlines = headlines[:8000]
test_headlines = headlines[8000:]

train_labels = labels[:8000]
test_labels = labels[8000:]

# Create a vectorizer and classifier
vectorizer = TfidfVectorizer()
svm = LinearSVC()

# Transform our text data into numerical vectors
train_vectors = vectorizer.fit_transform(train_headlines)
test_vectors = vectorizer.transform(test_headlines)

# Train the classifier and predict on test set
svm.fit(train_vectors, train_labels)

# Generate predictions based on test set
predictions = svm.predict(test_vectors)

# print(test_headlines[0:5])
# print(predictions[:5])

print(accuracy_score(test_labels, predictions))
