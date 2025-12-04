"""
camera_ui_emotion_age.py

Real-time webcam app that uses:

  - emotion_infer.py       (emotion model, model.torchscript.pt or model.pt)
  - test_image.py          (age model, best_resnet18_epoch100.pt)

For each detected face:
  - draws a bounding box
  - predicts an emotion
  - predicts the age bucket
  - overlays "Age: <bucket> | Emotion: <label>"

Controls:
  - Press 'q' or ESC to quit. or simply send a keyboard interrupt (this works for me)
"""

from __future__ import annotations

import sys
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image
import torch

import emotion_infer   
import test_image    


# 1. EMOTION MODEL
def init_emotion_model() -> Dict:
    """
    Load the emotion model once using emotion_infer's helpers.
      - model.torchscript.pt / model.pt 
    """
    model_path = emotion_infer._find_model_path(cli_path=None)
    backend = emotion_infer._infer_backend_from_path(model_path, backend=None)

    if backend == "tf":
        model, tf_mod = emotion_infer._load_tf_model(model_path)
        print(f"[Emotion] Loaded TF/Keras model from {model_path}")
        return {
            "backend": "tf",
            "model": model,
            "tf": tf_mod,
            "labels": emotion_infer.EMOTIONS_DEFAULT,
            "image_size": 48,
            "color": "grayscale",
        }
    else:
        model, torch_mod = emotion_infer._load_torchscript_model(model_path)
        print(f"[Emotion] Loaded TorchScript model from {model_path}")
        return {
            "backend": "torch",
            "model": model,
            "torch": torch_mod,
            "labels": emotion_infer.EMOTIONS_DEFAULT,
            "image_size": 48,
            "color": "grayscale",
        }


def predict_emotion_from_face(face_bgr: np.ndarray, state: Dict) -> Tuple[str, float]:
    """
    Classify a face crop using emotion_infer._tf_predict / _torch_predict.
    """
    if face_bgr.size == 0:
        return "unknown", 0.0

    size = state.get("image_size", 48)
    color = state.get("color", "grayscale")
    labels = state.get("labels", emotion_infer.EMOTIONS_DEFAULT)
    backend = state["backend"]

    if color == "grayscale":
        face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (size, size))
        arr = face_resized.astype("float32") / 255.0
    else:
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (size, size))
        arr = face_resized.astype("float32") / 255.0

    if backend == "tf":
        label, conf, _ = emotion_infer._tf_predict(
            state["tf"],
            state["model"],
            arr,
            channels_last=True,
            labels=labels,
        )
    else:
        label, conf, _ = emotion_infer._torch_predict(
            state["torch"],
            state["model"],
            arr,
            color=color,
            size=size,
            labels=labels,
        )

    return label, conf


# 2. AGE MODEL
def init_age_model():
    """
    Load the age model once via test_image.load_model().
    Expects best_resnet18_epoch100.pt next to test_image.py.
    """
    age_model = test_image.load_model()
    age_model.eval()
    print(f"[Age] Loaded age model from {test_image.MODEL_PATH}")
    return age_model


def predict_age_from_face(face_bgr: np.ndarray, age_model) -> Tuple[str, float]:
    """
    Predict age bucket using test_image's transform and AGE_CLASSES.
    """
    if face_bgr.size == 0:
        return "unknown", 0.0

    # BGR -> RGB -> PIL
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb)

    img_t = test_image.transform(face_pil).unsqueeze(0).to(test_image.DEVICE)

    with torch.no_grad():
        logits = age_model(img_t)
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(torch.argmax(probs).item())
        conf = float(probs[idx].item())

    age_label = test_image.AGE_CLASSES[idx]
    return age_label, conf

# 3. OpenCV Haar cascade
def init_face_detector():
    """
    like Face_Detection_CV
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError(f"Could not load Haar cascade at {cascade_path}")
    print("[Face] Haar cascade loaded")
    return face_cascade

# 4. MAIN LOOP
def run(camera_index: int = 0):
    face_detector = init_face_detector()
    emotion_state = init_emotion_model()
    age_model = init_age_model()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {camera_index}")
        sys.exit(1)

    print("[INFO] Webcam running. SEND A KEYBOARD INTERRUPT TO STOP FOR SOME REASON")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame")
            break

        # Mirror view
        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

        for (x, y, w, h) in faces:
            pad = int(0.1 * w)
            x0 = max(x - pad, 0)
            y0 = max(y - pad, 0)
            x1 = min(x + w + pad, frame.shape[1] - 1)
            y1 = min(y + h + pad, frame.shape[0] - 1)

            face_roi = frame[y0:y1, x0:x1]
            if face_roi.size == 0:
                continue

            try:
                emo_label, emo_conf = predict_emotion_from_face(face_roi, emotion_state)
            except Exception as e:
                emo_label, emo_conf = "err", 0.0
                print("[Emotion ERROR]", e)

            try:
                age_label, age_conf = predict_age_from_face(face_roi, age_model)
            except Exception as e:
                age_label, age_conf = "err", 0.0
                print("[Age ERROR]", e)

            # box
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

            # text label
            txt = (
                f"Age: {age_label} ({age_conf*100:.0f}%) | "
                f"Emotion: {emo_label} ({emo_conf*100:.0f}%)"
            )

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (tw, th), _ = cv2.getTextSize(txt, font, font_scale, thickness)

            text_y = y0 - 10 if y0 - 10 > 20 else y0 + 20
            cv2.rectangle(
                frame,
                (x0, text_y - th - 4),
                (x0 + tw + 4, text_y + 4),
                (0, 255, 0),
                thickness=-1,
            )
            cv2.putText(
                frame,
                txt,
                (x0 + 2, text_y),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA,
            )

        cv2.imshow("Real-time Age + Emotion", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # 'q' or ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cam_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run(camera_index=cam_idx)
