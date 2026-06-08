"""Run webcam attendance — detects and logs recognized faces to CSV"""
import face_recognition
import cv2
import pickle
import csv
import argparse
from pathlib import Path
from datetime import datetime

ENCODINGS_FILE = Path("encodings.pkl")
ATTENDANCE_DIR = Path("attendance")


def load_encodings():
    if not ENCODINGS_FILE.exists():
        raise FileNotFoundError("Run register.py first to encode faces")
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
    return data["encodings"], data["names"]


def get_attendance_file() -> Path:
    ATTENDANCE_DIR.mkdir(exist_ok=True)
    return ATTENDANCE_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.csv"


def mark_attendance(name: str, csv_path: Path, session_marked: set) -> bool:
    if name in session_marked:
        return False
    session_marked.add(name)
    timestamp = datetime.now().strftime("%H:%M:%S")
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["Name", "Time", "Date"])
        w.writerow([name, timestamp, datetime.now().strftime("%Y-%m-%d")])
    print(f"  Marked: {name} at {timestamp}")
    return True


def main(source: int | str = 0):
    known_encodings, known_names = load_encodings()
    csv_path = get_attendance_file()
    session_marked: set[str] = set()

    cap = cv2.VideoCapture(source)
    print(f"FaceAttend running — logging to {csv_path}. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        for encoding, location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
            name = "Unknown"
            if True in matches:
                distances = face_recognition.face_distance(known_encodings, encoding)
                best = int(distances.argmin())
                if matches[best]:
                    name = known_names[best]
                    mark_attendance(name, csv_path, session_marked)

            top, right, bottom, left = [v * 4 for v in location]
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("FaceAttend", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSession complete. {len(session_marked)} people marked present.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=0, help="Camera index or video file path")
    args = parser.parse_args()
    main(args.source)
