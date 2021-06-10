from os import makedirs
from pathlib import Path

from generate_trimap import find_trimap, save_trimap

validation_dir = Path(
    "/home/bram/BeCode/projects/background-separation/DUTS/DUTS-TE/DUTS-TE-Image/"
)
out_dir = validation_dir / "generated_trimaps"
if not out_dir.exists():
    makedirs(out_dir)

for file in validation_dir.glob("*.jpg"):
    trimap = find_trimap(str(file))

    out_path = out_dir / (file.stem + "_trimap.png")
    save_trimap(trimap, output_path=str(out_path))
    print(f"Saved trimap of '{file.name}' → '{out_path.name}'")
