import argparse
from typing import Any, Union

import cv2
from pathlib import Path

from cv2 import Mat
from numpy import dtype, floating, integer, ndarray


def capture(directory_name):

    # Create the directory
    image_dir = Path(directory_name)
    if (not image_dir.exists()):
        image_dir.mkdir()


    cam = cv2.VideoCapture(0)
    cv2.namedWindow("capture img")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "opencv_frame_{}.png".format(img_counter)
            image_file = image_dir.joinpath(img_name)
            #cv2.imwrite(image_file, frame)
            h, w = frame.shape[:2]
            #desired height is 320
            fx = 320/h
            fy = fx
            # Scale down by x%
            smallerImage = cv2.resize(frame, None,  fx=fx, fy=fy)
            cv2.imwrite(image_file, smallerImage)

            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()

cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image capture from webcam')
    parser.add_argument("-name")
    try:
        args = parser.parse_args()
        if args.name is None:
            print("missing image argument")
            parser.print_help()
        else:
            print("press ESC to exit and space to take image")
            capture(args.name)

    except (argparse.ArgumentError or argparse.ArgumentTypeError):
        print("failed to parse arguments")
        parser.print_help()

