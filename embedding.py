import face_recognition
import os
import pickle

FULL_NAME = "Carlos Alejandro Galindo Islas"
STUDENT_ID = "UTM22030587"

PHOTOS_DIR = "IA Photos"
ENCODINGS_FILE = "encodings.pkl"

student_embeddings = []

for filename in os.listdir(PHOTOS_DIR):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(PHOTOS_DIR, filename)
        print(f"Procesando {image_path}...")

        image = face_recognition.load_image_file(image_path)

        face_encodings = face_recognition.face_encodings(image)

        if face_encodings:
            for encoding in face_encodings:
                student_embeddings.append({
                    "full_name": FULL_NAME,
                    "student_id": STUDENT_ID,
                    "encoding": encoding
                })
        else:
            print(f"⚠️  No se detectó ningún rostro en {filename}")

with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump(student_embeddings, f)

print(f"\n✅ Embeddings guardados en '{ENCODINGS_FILE}' con {len(student_embeddings)} entradas.")
