import tensorflow as tf
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import argparse
import h5py
import cv2


def workaround_old_model(model_path):
    f = h5py.File(model_path, mode="r+")
    model_config_string = f.attrs.get("model_config")

    if model_config_string and '"groups": 1,' in model_config_string:
        model_config_string = model_config_string.replace('"groups": 1,', '')
        f.attrs.modify('model_config', model_config_string)
        f.flush()

    f.close()


def predict_image(imagePath):
    np.set_printoptions(suppress=True)

    workaround_old_model("keras_model.h5")
    model = load_model("keras_model.h5", compile=False)

    class_names = open("labels.txt", "r").readlines()

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    image = Image.open(imagePath).convert("RGB")

    size = (224, 224)

    # Pillow compatibility fix
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS

    image = ImageOps.fit(image, size, resample)

    image_array = np.asarray(image)

    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data[0] = normalized_image_array

    prediction = model.predict(data)

    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    print("Class:", class_name[2:] if len(class_name) > 2 else class_name)
    print("Confidence Score:", confidence_score)


def camera_record():
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Camera not available")
        return

    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        'output.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        20.0,
        (frame_width, frame_height)
    )
#model laden
    workaround_old_model("keras_model.h5")
    model = load_model("keras_model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        out.write(frame)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        size = (224, 224)

        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS

        image = ImageOps.fit(image, size, resample)

        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        prediction = model.predict(data)

        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]

        label = class_name[2:] if len(class_name) > 2 else class_name

        print("Class:", label)
        print("Confidence Score:", confidence_score)
        text = f"{label} ({confidence_score:.2f})"

        cv2.putText(
            frame,
            text,
            (10, 30),  # Position (x, y)
            cv2.FONT_HERSHEY_SIMPLEX,
            1,  # Schriftgröße
            (255, 140, 0),  # Farbe (grün)
            2  # Dicke
        )
        cv2.imshow('Camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Classification (Teachable Machine)')

    parser.add_argument("-image", nargs='?')
    parser.add_argument("-camera", action='store_true')

    args = parser.parse_args()

    if args.image:
        print("processing", args.image)
        predict_image(args.image)
    else:
        if args.camera:
            camera_record()
        else:
            print("missing image argument")
            parser.print_help()

    print("TensorFlow version:", tf.__version__)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
