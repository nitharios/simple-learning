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
