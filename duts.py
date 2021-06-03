from functools import lru_cache
import zipfile
from os import makedirs
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


class DutsDataset:
    download_link = "http://saliencydetection.net/duts/download/"

    def __init__(self, basepath: str):
        """Initiate the full DUTS dataset. SPecify path were to find/download/extract the dataset.

        :param basepath: Path to folder where the dataset is/will be stored.
        """

        # check local download
        self.basepath = Path(basepath)
        self.download_missing()

    def _reload_image_cache(self):
        # Internal list of all training images
        self.training_images = [
            DutsImage(name.stem, self.basepath / "DUTS-TE")
            for name in (self.basepath / "DUTS-TE").glob("**/*.jpg")
        ]
        # Internal list of all test images
        self.test_images = [
            DutsImage(name.stem, self.basepath / "DUTS-TR")
            for name in (self.basepath / "DUTS-TR").glob("**/*.jpg")
        ]

    def download_missing(self):
        """Checks whether dataset is present in basepath. If zip available, it extracts; if not, it suggests wget command."""
        # make sure folder exists
        if not self.basepath.exists():
            makedirs(self.basepath)

        for s in ["DUTS-TE", "DUTS-TR"]:
            if not (self.basepath / s).exists():
                print(f"{s} dataset not found in {self.basepath}.")
                if (self.basepath / s).with_suffix(".zip").exists():
                    # extract downloaded zip
                    print(f"Extracting {s} from zip")
                    with zipfile.ZipFile((self.basepath / s).with_suffix(".zip")) as zf:
                        zf.extractall(self.basepath)
                else:
                    print(
                        f"Zipfile not available, download with \n!wget -cP {self.basepath} {self.download_link}{s}.zip\nand run .download_missing() again."
                    )

        self._reload_image_cache()


class DutsImage:
    """Wrapper around single image of the DUTS dataset, with methods to show with matplotlib."""

    def __init__(self, name, basepath="./DUTS/DUTS-TR/"):
        """Create new DUTS image.

        :param name: filename of image (without extension)
        :param basepath: folder with 'DUTS-*-Images' folders
        """
        # set paths
        self.name = name
        basepath = Path(basepath)
        self.orig_path = (basepath / f"{basepath.name}-Image" / name).with_suffix(
            ".jpg"
        )
        self.mask_path = (basepath / f"{basepath.name}-Mask" / name).with_suffix(".png")

    @lru_cache(100)
    def get_image(self) -> np.ndarray:
        """Return numpy array of unaltered image."""
        return cv2.cvtColor(
            cv2.imread(str(self.orig_path.resolve()), cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB,
        )

    @lru_cache(100)
    def get_mask(self) -> np.ndarray:
        """Returns numpy array for unaltered mask"""
        return cv2.imread(str(self.mask_path.resolve()), cv2.IMREAD_GRAYSCALE) / 255

    @lru_cache(100)
    def generate_trimap(self) -> np.ndarray:
        """Returns trimap, generated from ground thruth."""
        mask_ = self.get_mask()  # load mask

        kernel_size = np.array(mask_.shape) // 40
        kernel = np.ones(kernel_size, np.uint8)

        dilation = cv2.dilate(mask_, kernel, iterations=1)
        erosion = cv2.erode(mask_, kernel, iterations=1)

        return (erosion + dilation) / 2

    def show_image(self, ax=None):
        if not ax:
            fig, ax = plt.subplots()

        ax.imshow(self.get_image())
        ax.set_axis_off()
        return ax

    def show_mask(self, ax=None):
        if not ax:
            fig, ax = plt.subplots()
        ax.imshow(self.get_mask())
        ax.set_axis_off()
        return ax

    def show_trimap(self, ax=None):
        if not ax:
            fig, ax = plt.subplots()
        ax.imshow(self.generate_trimap())
        ax.set_axis_off()
        return ax

    def show_both(self, ax=None, bg_color=[0, 0, 0]):
        """Show image with mask applied."""
        if not ax:
            fig, ax = plt.subplots()
        background = np.zeros(self.get_image().shape)
        for i in range(3):
            background[:, :, i] = bg_color[i]
        mask = np.expand_dims(
            self.get_mask(),
            axis=2,
        )
        mix = (1 - mask) * background + self.get_image() * mask
        ax.imshow(mix.astype(np.uint8))
        ax.set_axis_off()
        return ax

    def show_trimap_both(self, ax=None):
        """Show generated trimap on top of image."""
        if not ax:
            fig, ax = plt.subplots()

        mix = self.get_image().copy()
        mix[self.generate_trimap() == 0.5, :] = [255, 0, 0]
        mix[self.generate_trimap() > 0.5, :] = [255, 255, 0]
        ax.imshow(mix)
        ax.set_axis_off()
        return ax

    def show_all(self, *args, **kwargs):
        fig, axes = plt.subplots(ncols=4, figsize=(20, 8))
        self.show_image(axes[0])
        self.show_mask(axes[1])
        self.show_both(axes[2], *args, **kwargs)
        self.show_trimap_both(axes[3])

        axes[0].set_title("Original")
        axes[1].set_title("Target alpha")
        axes[2].set_title("Alpha applied")
        axes[3].set_title("Generated Trimap")

        fig.tight_layout()
        return fig, axes

    def __repr__(self) -> str:
        return f"<DutsImage ‘{self.name}’>"


if __name__ == "__main__":
    dataset = DutsDataset("../DUTS")
    dataset.training_images[0].show_all()
    plt.show()
