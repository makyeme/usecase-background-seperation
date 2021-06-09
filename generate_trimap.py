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


def show_step(img, title):
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

    # Scale back to original size
    trimap_upscaled = cv2.resize(
        prediction_1, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LANCZOS4
    )
    if show_steps:
        show_step(trimap_upscaled, "Final Trimap")

    # round back to {0,1,2}
    final = np.round(trimap_upscaled).astype(np.uint8)
    return final


def save_trimap(trimap_array, output_path):
    """Export a trimap image as PNG."""
    normalized = np.atleast_3d(np.round(trimap_array / 2 * 255))
    cv2.imwrite(output_path, normalized)


if __name__ == "__main__":
    example_path = Path(argv[1] if len(argv) > 1 else "test_dog.jpg")
    print(f"Generating trimap for {example_path}.")
    example = find_trimap(str(example_path), show_steps=False)
    out_path = example_path.parent / (example_path.stem + "_trimap.png")
    print(f"Saving trimap to {out_path}.")
    save_trimap(example, str(out_path))
