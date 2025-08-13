import face_recognition
import pickle

# === 1. Cargar PKL con los datos del estudiante ===
PKL_FILE = "encodings.pkl"  # Cambia por la ruta de tu archivo

with open(PKL_FILE, "rb") as f:
    student_data = pickle.load(f)

known_encodings = student_data["encodings"]
student_name = student_data["full_name"]

# === 2. Lista de rutas de las fotos a analizar ===
image_paths = [
    "Test Photos/test_1.jpg",
    "Test Photos/test.jpg"
]


# === 3. Procesar cada foto ===
for image_path in image_paths:
    try:
        # Cargar imagen y convertir a RGB
        image = face_recognition.load_image_file(image_path)

        # Obtener encodings de los rostros en la imagen
        image_encodings = face_recognition.face_encodings(image)

        if not image_encodings:
            print(f"[{image_path}] ❌ No se detectó ningún rostro.")
            continue

        detected = False
        for encoding in image_encodings:
            matches = face_recognition.compare_faces(known_encodings, encoding)
            if True in matches:
                print(f"[{image_path}] ✅ Persona detectada: {student_name}")
                detected = True
                break

        if not detected:
            print(f"[{image_path}] ❌ No coincide con {student_name}")

    except Exception as e:
        print(f"[ERROR] No se pudo procesar {image_path}: {e}")
