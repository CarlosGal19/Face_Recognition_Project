import face_recognition
import os
import pickle

# Cambia estos valores con tus datos
FULL_NAME = "Carlos Alejandro Galindo Islas"
STUDENT_ID = "UTM22030587"

# Ruta a la carpeta con fotos
PHOTOS_DIR = "IA Photos"
ENCODINGS_FILE = "encodings.pkl"

# Lista donde se almacenarán todos los embeddings
embeddings_list = []

# Recorre todos los archivos en la carpeta
for filename in os.listdir(PHOTOS_DIR):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(PHOTOS_DIR, filename)
        print(f"Procesando {image_path}...")

        # Carga la imagen
        image = face_recognition.load_image_file(image_path)

        # Detecta las caras y obtiene los embeddings
        face_encodings = face_recognition.face_encodings(image)

        if face_encodings:
            # Agrega todos los embeddings encontrados (por si hay más de una cara)
            embeddings_list.extend(face_encodings)
        else:
            print(f"⚠️  No se detectó ningún rostro en {filename}")

# Estructura final del estudiante
student_data = {
    "full_name": FULL_NAME,
    "student_id": STUDENT_ID,
    "encodings": embeddings_list
}

# Guardar en archivo .pkl
with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump(student_data, f)

print(f"\n✅ Embeddings guardados en '{ENCODINGS_FILE}' con {len(embeddings_list)} vectores.")
