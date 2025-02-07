import os
import cv2
import dlib
import numpy as np
import tensorflow as tf

# 🔹 Lade das vortrainierte Modell
model_path = "/Users/blaubaer/Projects/age_timeline/own_model/age_model.h5"  # Falls du .h5 hast, ändere das
# model = tf.keras.models.load_model(model_path)

# 🔹 Modell ohne Kompilierung laden
model = tf.keras.models.load_model(model_path, compile=False)
print("✅ Modell erfolgreich geladen!")

# 🔹 Manuell kompilieren
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
print("✅ Modell wurde neu kompiliert und ist einsatzbereit!")

# 🔹 Dlib Gesichtsdetektor laden
detector = dlib.get_frontal_face_detector()

# 🔹 Ordner definieren
image_folder = "/Users/blaubaer/Projects/age_timeline/own_model/analog_portraits/"
output_folder = "/Users/blaubaer/Projects/age_timeline/own_model/sorted_faces/"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def detect_and_predict_age(image_path):
    """ Erkennt Gesichter in einem Bild und sagt deren Alter voraus. """
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"⚠️ Fehler: Bild konnte nicht geladen werden: {image_path}")
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    face_data = []

    for i, face in enumerate(faces):
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_img = image[y:y+h, x:x+w]

        # Prüfen, ob das Gesicht gültig ist
        if face_img is None or face_img.size == 0:
            print(f"⚠️ Warnung: Gesicht konnte nicht extrahiert werden! Überspringe Bild: {image_path}")
            continue

        # Gesicht auf die Größe 64x64 skalieren
        face_resized = cv2.resize(face_img, (64, 64)) / 255.0  # Normalisieren
        face_array = np.expand_dims(face_resized, axis=0)

        # Alter vorhersagen
        predicted_age = model.predict(face_array)[0][0]
        face_data.append((predicted_age, face_img))

    return face_data

def process_images():
    """ Durchsucht alle Bilder im Eingangsordner, extrahiert Gesichter und speichert sie sortiert nach Alter. """
    all_faces = []

    for image_name in os.listdir(image_folder):
        if image_name.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_folder, image_name)
            detected_faces = detect_and_predict_age(image_path)
            all_faces.extend(detected_faces)

    # 🔹 Gesichter nach Alter sortieren
    all_faces.sort(key=lambda x: x[0])  

    # 🔹 Gespeicherte Gesichter nach Alter abspeichern
    for i, (age, face) in enumerate(all_faces):
        output_path = os.path.join(output_folder, f"{i+1:02d}_age{int(age)}.jpg")
        cv2.imwrite(output_path, face)
        print(f"✅ Gesicht gespeichert: {output_path} (Alter: {int(age)})")

# 🔹 Anwendung starten
process_images()
print("🚀 Verarbeitung abgeschlossen! Alle Gesichter wurden nach Alter sortiert.")

