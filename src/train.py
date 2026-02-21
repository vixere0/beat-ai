import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

from dataset import load_dataset

X, y, label_map = load_dataset(r"c:\Users\Abdullah Baran\OneDrive\Desktop\beat-ai\data")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
model.fit(X_train, y_train)



model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))

print("\nLabel map:", label_map)
joblib.dump(model, r"c:\Users\Abdullah Baran\OneDrive\Desktop\beat-ai\model\genre_classifier.pkl")
joblib.dump(label_map, r"c:\Users\Abdullah Baran\OneDrive\Desktop\beat-ai\model\label_map.pkl")
joblib.dump(scaler, r"c:\Users\Abdullah Baran\OneDrive\Desktop\beat-ai\model\scaler.pkl")
print("Model kaydedildi.")
