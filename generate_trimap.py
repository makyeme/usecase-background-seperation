from pathlib import Path
from sys import argv

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from duts import DutsDataset

# load trained model
trained_model = tf.keras.models.load_model("saved_models/segmentation_model")

# update model with extra training checkpoints
trained_model.load_weights("saved_models/segmentation_updates/checkpoints")


def show_step(img, title=""):
    """Just to show an image with Matplotlib, to inspect steps."""
    plt.figure()
    plt.imshow(img)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def find_trimap(img_path, show_steps=False):
    """Generates a trimap from an image file.

    :param img_path: Path to img file, all OpenCV compatible types supported.
    :param show_steps: Whether to show the intermediate steps with MPL, for debugging purposes.
    """

    # load image
    image = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    if show_steps:
        show_step(image, "Original")

    # Resize to 224,224
    image_resized = cv2.resize(image, (224, 224))
    if show_steps:
        show_step(image_resized, "Resized")

    # make tensor
    tensor = tf.convert_to_tensor([image_resized.astype(np.float32) / 255.0])

    # first prediction
    prediction_1 = (
        tf.argmax(trained_model.predict(tensor), -1)[0].numpy().astype(np.float32)
    )
    if show_steps:
        show_step(prediction_1, "First trimap prediction")

    # Morphological cleaning
    kernel = np.ones((5, 5), np.uint8)
    prediction_1 = cv2.morphologyEx(prediction_1, cv2.MORPH_OPEN, kernel)
    if show_steps:
        show_step(prediction_1, "Morphologic opened")

    # Crop image to interesting part and start over
    ## Get part of matrix that isolates nonzero values
    nzy, nzx = np.nonzero(prediction_1)
    X = np.array([nzx.min(), nzx.max()]).astype(np.float32)
    Y = np.array([nzy.min(), nzy.max()]).astype(np.float32)
    ## Transform these to original image size
    X *= width / 224.0
    Y *= height / 224.0
    ## Calculate some numbers for cutting out a square
    square_length = np.min(
        [
            np.max([X[1] - X[0], Y[1] - Y[0]]) * 1.05,
            height,
            width,
        ]
    )  # 5% increase around subject
    x_padding = (square_length - (X[1] - X[0])) / 2
    y_padding = (square_length - (Y[1] - Y[0])) / 2
    ## Move points to upper-left and -lower-right corners
    X += x_padding * np.array([-1, 1])
    Y += y_padding * np.array([-1, 1])
    X = np.round(X).astype(np.int16)
    Y = np.round(Y).astype(np.int16)
    ## Check if crop falls outside image
    X = np.clip(X, 0, width)
    Y = np.clip(Y, 0, height)
    ## Finally, select this part of the image
    cropped_img = image[Y[0] : Y[1], X[0] : X[1]]
    cropped_shape = cropped_img.shape[:2]
    if show_steps:
        show_step(cropped_img, "Autocropped subject")

    # Predict on autocropped image
    tensor = tf.convert_to_tensor(
        [cv2.resize(cropped_img, (224, 224)).astype(np.float32) / 255.0]
    )
    prediction_2 = (
        tf.argmax(trained_model.predict(tensor), -1)[0].numpy().astype(np.float32)
    )
    if show_steps:
        show_step(prediction_2, "Second trimap prediction")

    # Morphological cleaning
    # kernel = np.ones((5, 5), np.uint8)
    prediction_2 = cv2.morphologyEx(prediction_2, cv2.MORPH_OPEN, kernel)
    prediction_2 = cv2.morphologyEx(prediction_2, cv2.MORPH_CLOSE, kernel)
    if show_steps:
        show_step(prediction_2, "Morphologically adj, 2")

    # Put it back in original image format
    ## scale back
    prediction_2 = np.round(
        cv2.resize(
            prediction_2,
            np.flip(cropped_shape),
            interpolation=cv2.INTER_CUBIC,
        )
    )
    ## Put it in an empty image on the correct position
    final = np.zeros(shape=image.shape[:2], dtype=np.uint8)
    final[Y[0] : Y[1], X[0] : X[1]] = prediction_2
    if show_steps:
        show_step(final, "Upscaled to original size")

    return final


def save_trimap(trimap_array, output_path):
    """Export a trimap image as PNG."""
    normalized = np.atleast_3d(np.round(trimap_array.astype(np.float32) * 255.0 / 2.0))
    cv2.imwrite(output_path, normalized)


if __name__ == "__main__":
    # EXAMPLE OF USAGE

    # select input image from command line arg
    example_path = Path(argv[1] if len(argv) > 1 else "test_image.jpg")
    print(f"Generating trimap for {example_path}.")

    # find trimap
    example = find_trimap(str(example_path), show_steps=False)

    # output file name: same as input with _trimap suffix
    out_path = example_path.parent / (example_path.stem + "_trimap.png")
    print(f"Saving trimap to {out_path}.")

    # save to PNG
    save_trimap(example, str(out_path))
