import zipfile
from os import makedirs
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


class DutsDataset:
    download_link = "http://saliencydetection.net/duts/download/"

    def __init__(self, basepath: str):
        # check local download
        self.basepath = Path(basepath)
        self.download_missing()

        self.training_images = [
            DutsImage(name.stem, self.basepath / "DUTS-TE")
            for name in (self.basepath / "DUTS-TE").glob("**/*.jpg")
        ]
        self.test_images = [
            DutsImage(name.stem, self.basepath / "DUTS-TR")
            for name in (self.basepath / "DUTS-TR").glob("**/*.jpg")
        ]

    def download_missing(self):
        if not self.basepath.exists():
            makedirs(self.basepath)

        for s in ["DUTS-TE", "DUTS-TR"]:
            if not (self.basepath / s).exists():
                print(f"{s} dataset not found in {self.basepath}.")
                if (self.basepath / s).with_suffix(".zip").exists():
                    # extract downloaded zip
                    print(f"Extracting {s} from zip")
                    with zipfile.ZipFile((self.basepath / s).with_suffix(".zip")) as zf:
                        zf.extractall(self.basepath / s)
                else:
                    print(
                        f"Zipfile not available, download with `!wget -cP {self.basepath} {self.download_link}{s}.zip`"
                    )


class DutsImage:
    def __init__(self, name, basepath="./DUTS/DUTS-TR/"):
        self.name = name
        basepath = Path(basepath)
        self.orig_path = (basepath / f"{basepath.name}-Image" / name).with_suffix(
            ".jpg"
        )
        self.mask_path = (basepath / f"{basepath.name}-Mask" / name).with_suffix(".png")

    def get_image(self) -> np.ndarray:
        return cv2.cvtColor(
            cv2.imread(str(self.orig_path.resolve()), cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB,
        )

    def get_mask(self) -> np.ndarray:
        return cv2.imread(str(self.mask_path.resolve()), cv2.IMREAD_GRAYSCALE) / 255

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

    def show_both(self, ax=None, bg_color=[0, 0, 0]):
        if not ax:
            fig, ax = plt.subplots()
        background = np.zeros(self.get_image().shape)
        for i in range(3):
            background[:, :, i] = bg_color[i]
        mask = np.expand_dims(self.get_mask(), axis=2)
        mix = (1 - mask) * background + self.get_image() * mask
        ax.imshow(mix.astype(np.uint8))
        ax.set_axis_off()
        return ax

    def show_all(self, *args, **kwargs):
        fig, axes = plt.subplots(ncols=3, figsize=(16, 8))
        self.show_image(axes[0])
        self.show_mask(axes[1])
        self.show_both(axes[2], *args, **kwargs)
        return fig, axes

    def __repr__(self) -> str:
        return f"<DutsImage ‘{self.name}’>"


if __name__ == "__main__":

    dataset = DutsDataset("../DUTS")
    print(dataset.training_images)
