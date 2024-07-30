import os
import itertools
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import extract
import categorize

# Initialize lists to store features and labels
features = {
    "baseline_angle": [],
    "top_margin": [],
    "letter_size": [],
    "line_spacing": [],
    "word_spacing": [],
    "pen_pressure": [],
    "slant_angle": []
}

page_ids = []

# Load label data from file
def load_labels(file_path, features, traits, page_ids):
    if not os.path.isfile(file_path):
        print("Error: No Label list found")
        return False

    with open(file_path, "r") as labels:
        for line in labels:
            content = line.split()
            features["baseline_angle"].append(float(content[0]))
            features["top_margin"].append(float(content[1]))
            features["letter_size"].append(float(content[2]))
            features["line_spacing"].append(float(content[3]))
            features["word_spacing"].append(float(content[4]))
            features["pen_pressure"].append(float(content[5]))
            features["slant_angle"].append(float(content[6]))
            
            for i in range(8):
                traits[f"t{i+1}"].append(float(content[7 + i]))
            
            page_ids.append(content[15])
    return True

# Combine features for trait prediction
def combine_features(features_list):
    return [list(pair) for pair in zip(*features_list)]

# Train and evaluate classifier
def train_and_evaluate(X, y, random_state):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=random_state)
    clf = SVC(kernel='rbf')
    clf.fit(X_train, y_train)
    accuracy = accuracy_score(clf.predict(X_test), y_test)
    return clf, accuracy

# Main function to load data, train classifiers, and predict traits
def main():
    traits = {f"t{i+1}": [] for i in range(8)}  # Move traits initialization here

    if not load_labels("label_list", features, traits, page_ids):
        return

    feature_pairs = [
        (features["baseline_angle"], features["slant_angle"]),
        (features["letter_size"], features["pen_pressure"]),
        (features["letter_size"], features["top_margin"]),
        (features["line_spacing"], features["word_spacing"]),
        (features["slant_angle"], features["top_margin"]),
        (features["letter_size"], features["line_spacing"]),
        (features["letter_size"], features["word_spacing"]),
        (features["line_spacing"], features["word_spacing"])
    ]

    random_states = [8, 16, 32, 64, 42, 52, 21, 73]
    classifiers = []

    for i, (X_features, y_traits) in enumerate(zip(feature_pairs, traits.values())):
        X = combine_features(X_features)
        clf, accuracy = train_and_evaluate(X, y_traits, random_states[i])
        classifiers.append(clf)
        print(f"Classifier {i+1} accuracy: {accuracy}")

    # Prediction loop
    while True:
        file_name = input("Enter file name to predict or x to exit: ")
        if file_name.lower() == 'x':
            break

        raw_features = extract.start(file_name)
        processed_features = {
            "baseline_angle": categorize.determine_baseline_angle(raw_features[0])[0],
            "top_margin": categorize.determine_top_margin(raw_features[1])[0],
            "letter_size": categorize.determine_letter_size(raw_features[2])[0],
            "line_spacing": categorize.determine_line_spacing(raw_features[3])[0],
            "word_spacing": categorize.determine_word_spacing(raw_features[4])[0],
            "pen_pressure": categorize.determine_pen_pressure(raw_features[5])[0],
            "slant_angle": categorize.determine_slant_angle(raw_features[6])[0]
        }

        trait_combinations = [
            [processed_features["baseline_angle"], processed_features["slant_angle"]],
            [processed_features["letter_size"], processed_features["pen_pressure"]],
            [processed_features["letter_size"], processed_features["top_margin"]],
            [processed_features["line_spacing"], processed_features["word_spacing"]],
            [processed_features["slant_angle"], processed_features["top_margin"]],
            [processed_features["letter_size"], processed_features["line_spacing"]],
            [processed_features["letter_size"], processed_features["word_spacing"]],
            [processed_features["line_spacing"], processed_features["word_spacing"]]
        ]

        trait_names = [
            "Emotional Stability", "Mental Energy or Will Power", "Modesty",
            "Personal Harmony and Flexibility", "Lack of Discipline",
            "Poor Concentration", "Non Communicativeness", "Social Isolation"
        ]

        results = []
        for i, (clf, traits) in enumerate(zip(classifiers, trait_combinations)):
            prediction = clf.predict([traits])[0]
            results.append((trait_names[i], prediction))

        print("\n--- Graphology Traits Prediction ---")
        for trait, prediction in results:
            print(f"{trait}: {'Positive' if prediction == 1 else 'Negative'}")
        print("------------------------------------\n")

if __name__ == "__main__":
    main()
