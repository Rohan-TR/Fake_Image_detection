import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

# Function for Error Level Analysis (ELA)
def ela(image_path, scale=10):
    original_image = cv2.imread(image_path)
    resized_image = cv2.resize(original_image, (scale, scale))
    cv2.imwrite("temp.jpg", resized_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    temp_image = cv2.imread("temp.jpg")
    ela_image = cv2.absdiff(resized_image, temp_image)
    os.remove("temp.jpg")
    return ela_image

# Function to load and preprocess data
def load_data(data_folder):
    data = []
    labels = []

    for image_file in os.listdir(data_folder):
        if image_file.endswith(".jpg"):
            image_path = os.path.join(data_folder, image_file)
            ela_img = ela(image_path)
            data.append(ela_img)
            labels.append("real" if "real" in image_file else "fake")

    le = LabelEncoder()
    labels = le.fit_transform(labels)
    labels = to_categorical(labels)

    return np.array(data), np.array(labels)

# Function to create CNN model
def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))  # 2 classes: real and fake
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Main code
data_folder = "path/to/your/data/folder"
data, labels = load_data(data_folder)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Create CNN model
input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
model = create_cnn_model(input_shape)

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
accuracy = model.evaluate(X_test, y_test)[1]
print(f"Model Accuracy: {accuracy}")
