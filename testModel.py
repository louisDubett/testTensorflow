import tensorflow as tf
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import argparse
import h5py

update_old_teachable_model=False

def tensorflow_model_ki(imagePath,model_name ):
    #based on https://www.tensorflow.org/tutorials/keras/save_and_load?hl=fr
    new_model = tf.keras.models.load_model(model_name)
    # Check its architecture
    new_model.summary()
    # Evaluate the restored model

    img_height = 300
    img_width = 400
    img = tf.keras.utils.load_img(
        imagePath, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = new_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    # class_names=["daisy","dandelion","rose","sunflowers","tulips"]
    class_names = ["becher", "erlenmeyer", "kolben", "messzylinder", "pipette", "reagenzglas"]
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )


    #loss, acc = new_model.evaluate(imagePath, test_labels, verbose=2)
    #print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

    #print(new_model.predict(imagePath).shape)


def teachable_machine_ki(imagePath, model_name):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    #hack because old version of tensorflow used by teachable machine
    #https://stackoverflow.com/questions/78187204/trying-to-export-teachable-machine-model-but-returning-error
    if update_old_teachable_model:
        f = h5py.File(model_name, mode="r+")
        model_config_string = f.attrs.get("model_config")

        if model_config_string.find('"groups": 1,') != -1:
            model_config_string = model_config_string.replace('"groups": 1,', '')
        f.attrs.modify('model_config', model_config_string)
        f.flush()

        model_config_string = f.attrs.get("model_config")

        assert model_config_string.find('"groups": 1,') == -1
    #endof of hack

    model = load_model(model_name, compile=False)
    model.summary()

    # Load the labels
    #class_names = open("labels.txt", "r").readlines()
    #TODO remove hard coded
    class_names = ["Becher", "Erlenmeyer", "Kolben", "Messzylinder", "Pipette","Reagenzglas"]

    #size given by teachable machine
    img_height = 224
    img_width = 224

    img = tf.keras.utils.load_img(
        imagePath, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
###old


    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    #data = np.ndarray(shape=(1, img_height, img_width, 3), dtype=np.float32)

    # Replace this with the path to your image
    #image = Image.open(imagePath).convert("RGB")

    # resizing the image to be at least 180*180 and then cropping from the center

    #size = (img_height, img_width)
    #image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    #image_array = np.asarray(image)

    # Normalize the image
    #normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    #data[0] = normalized_image_array

    # Predicts the model
    #prediction = model.predict(data)
    #index = np.argmax(prediction)
    #class_name = class_names[index]
    #confidence_score = prediction[0][index]

    # Print prediction and confidence score
    #print("Class:", class_name[2:], end="")
    #print("Confidence Score:", confidence_score)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='Image Classification from teachable machine or tensor flow')
    #use for ex
    #               -model my_model-chimie.keras -image becher100_ml_1.png --tensorflow
    #or
    #               -model teachable_machine_model.h5 -image becher100_ml_1.png --teachable_machine
    #
    parser.add_argument("-image", nargs='?')
    parser.add_argument("-model", nargs='?')
    parser.add_argument('--teachable_machine', action='store_true', help='use teachable machine model')
    parser.add_argument('--tensorflow', action='store_true', help='use teachable machine model')
    try:
        args = parser.parse_args()
        if args.image is None or args.model is None:
            print("missing image argument")
            parser.print_help()
        else:
            print('processing '+ args.image+ ' with model '+ args.model)
            if args.teachable_machine:
                teachable_machine_ki(args.image, args.model)
            else:
                if args.tensorflow:
                    tensorflow_model_ki(args.image,args.model)

    except (argparse.ArgumentError or argparse.ArgumentTypeError):
        print("failed to parse arguments")
        parser.print_help()
        raise

    print("TensorFlow version:", tf.__version__)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
