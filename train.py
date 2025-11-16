import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import subprocess
import os

mlflow.set_tracking_uri("http://136.112.45.174:5000")


experiment_name = "IRIS_poisoning_test_mlops_week_8"
artifact_location = "gs://mlops-course-clean-vista-473214-i6/mlflow-assets/iris-classifier-model"

experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    print(f"Creating new experiment '{experiment_name}'...")
    mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
    mlflow.set_experiment(experiment_name)
else:
    mlflow.set_experiment(experiment_name)

df = pd.read_csv("./data/iris.csv")
features = df.drop("species", axis=1)
labels = LabelEncoder().fit_transform(df["species"])
feat_tr, feat_val, lbl_tr, lbl_val = train_test_split(features, labels, test_size=0.3, random_state=21, stratify=labels)

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

for noise_level in [0.0, 0.05, 0.10, 0.50]:
    with mlflow.start_run(run_name=f"{'Clean' if noise_level == 0 else f'Poisioned_{int(noise_level*100)}pct'}"):
        corrupted_labels = lbl_tr.copy()
        num_corrupt = int(len(lbl_tr) * noise_level)
        
        if num_corrupt:
            corrupt_idx = np.random.choice(len(lbl_tr), num_corrupt, replace=False)
            for idx in corrupt_idx:
                corrupted_labels[idx] = np.random.choice([x for x in [0,1,2] if x != corrupted_labels[idx]])
        
        # Save training data with corrupted labels in data subfolder
        train_df = feat_tr.copy()
        train_df['species'] = corrupted_labels
        train_df.to_csv("data/train.csv", index=False)
        
        # Track with DVC and push to cloud
        subprocess.run(["dvc", "add", "data/train.csv"], check=True)
        subprocess.run(["dvc", "push"], check=True)
        subprocess.run(["git", "add", "data/train.csv.dvc", "data/.gitignore"])
        subprocess.run(["git", "commit", "-m", f"Add training data for noise_level={noise_level}"], check=False)
        
        clf = DecisionTreeClassifier(max_depth=3, random_state=1)
        clf.fit(feat_tr, corrupted_labels)
        
        mlflow.log_params({"classifier": "DecisionTree", "depth": 3, "noise_rate": noise_level})
        mlflow.log_metric("val_accuracy", accuracy_score(lbl_val, clf.predict(feat_val)))
        mlflow.sklearn.log_model(clf, "model", input_example=feat_tr.head(5))