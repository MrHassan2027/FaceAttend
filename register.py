"""Encode faces from the /faces directory into encodings.pkl"""
import face_recognition
import pickle
from pathlib import Path

FACES_DIR = Path("faces")
OUTPUT = Path("encodings.pkl")

def register():
    known_encodings = []
    known_names = []

    image_paths = list(FACES_DIR.glob("*.jpg")) + list(FACES_DIR.glob("*.png"))
    if not image_paths:
        print("No images found in ./faces — add Name.jpg files")
        return

    for img_path in image_paths:
        name = img_path.stem
        print(f"  Encoding {name}...")
        img = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(img)
        if not encodings:
            print(f"  WARNING: No face found in {img_path.name}, skipping")
            continue
        known_encodings.append(encodings[0])
        known_names.append(name)

    with open(OUTPUT, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)

    print(f"\nRegistered {len(known_names)} faces → {OUTPUT}")
    for name in known_names:
        print(f"  - {name}")

if __name__ == "__main__":
    register()
