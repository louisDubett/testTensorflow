import tensorflow as tf
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import argparse
import h5py

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def workaround_old_model(kera_old_model_name):
    #see https://discuss.ai.google.dev/t/cannot-load-h5-model/42465/2
    #teachable machine provide an old version for keras that does not match this tensorflow version
    f = h5py.File(kera_old_model_name, mode="r+")
    model_config_string = f.attrs.get("model_config")
    if model_config_string.find('"groups": 1,') != -1:
        model_config_string = model_config_string.replace('"groups": 1,', '')
        f.attrs.modify('model_config', model_config_string)
        f.flush()
        model_config_string = f.attrs.get("model_config")
        assert model_config_string.find('"groups": 1,') == -1

    f.close()

def louis_ki(imagePath):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    workaround_old_model("keras_model.h5")
    model = load_model("keras_model.h5", compile=False)

    # Load the labels
    class_names = open("labels.txt", "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(imagePath).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='Image Classification from teachable machine')
    parser.add_argument("-image", nargs='?')
    try:
        args = parser.parse_args()
        if args.image is None:
            print("missing image argument")
            parser.print_help()
        else:
            print('processing '+ args.image)
            louis_ki(args.image)
    except (argparse.ArgumentError or argparse.ArgumentTypeError):
        print("failed to parse arguments")
        parser.print_help()
        raise

    print("TensorFlow version:", tf.__version__)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
