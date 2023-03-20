from argparse import ArgumentParser
from pathlib import Path
from typing import List

import numpy as np
import PIL
import tensorflow_datasets as tfds
from tqdm.auto import tqdm


def prepare_obj_data(n_classes: int) -> str:
    return (
        f"classes = {n_classes}\n"
        f"train = data/train.txt\n"
        f"valid = data/test.txt\n"
        f"names = data/obj.names\n"
        f"backup = data/\n"
    )


def prepare_obj_names(names: List[str]) -> str:
    return "\n".join(names)


def get_frame_with_annotations(example, label_key: str):
    num_objects = example["instances"][label_key].shape[0]
    y1x1y2x2 = example["instances"]["bboxes"]
    y1x1y2x2_frames = example["instances"]["bbox_frames"]
    labels = example["instances"][label_key]
    frames = example["video"]

    for t, frame in enumerate(frames):
        annotations = []
        for k in range(num_objects):
            if t in y1x1y2x2_frames[k]:
                idx = np.nonzero(y1x1y2x2_frames[k] == t)[0][0]
                y1, x1, y2, x2 = y1x1y2x2[k][idx]
                x = (x1 + x2) / 2
                y = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                annotations.append(f"{labels[k]} {x} {y} {w} {h}")
        yield frame, "\n".join(annotations)


def save_image_and_annotations(
    image: np.ndarray, annotations: str, directory: Path, idx: int, padding: int = 6
):
    image = PIL.Image.fromarray(image)
    image.save(directory / f"{idx:0{padding}}.jpg")
    with open(directory / f"{idx:0{padding}}.txt", "w") as f:
        f.write(annotations)
    return f"data/{'/'.join(directory.parts[1:])}/{idx:0{padding}}.jpg"


def save_dataset(dataset, dataset_info, directory: str):
    label_key = "shape_label"
    try:
        classes = dataset_info.features["instances"][label_key].names
    except KeyError:
        label_key = "category"
        classes = dataset_info.features["instances"][label_key].names
    n_classes = len(classes)
    directory_path = Path(directory)
    directory_path.mkdir(parents=True, exist_ok=True)
    with open(directory_path / "obj.data", "w") as f:
        f.write(prepare_obj_data(n_classes))
    with open(directory_path / "obj.names", "w") as f:
        f.write(prepare_obj_names(classes))
    subsets = {"train": "train", "test": "validation"}
    for subset in ["train", "test"]:
        subset_path = directory_path / subset
        subset_path.mkdir(parents=True, exist_ok=True)
        files = []
        ds = dataset[subsets[subset]]
        for seq_idx, seq in tqdm(
            enumerate(iter(tfds.as_numpy(ds))),
            total=ds.cardinality().numpy(),
            desc=f"Saving {subset} dataset",
            leave=False,
        ):
            seq_path = subset_path / f"{seq_idx:06}"
            seq_path.mkdir(parents=True, exist_ok=True)
            for t, (image, ann) in tqdm(
                enumerate(get_frame_with_annotations(seq, label_key)),
                desc="Image",
                leave=False,
                total=len(seq["video"]),
            ):
                filename = save_image_and_annotations(image, ann, seq_path, t)
                files.append(filename)

        with open(directory_path / f"{subset}.txt", "w") as f:
            f.write("\n".join(files))


def main():
    parser = ArgumentParser(
        description="Download a dataset from tfds and save it in YOLO format."
    )
    parser.add_argument("dataset", type=str)
    args = parser.parse_args()

    ds, ds_info = tfds.load(
        args.dataset, data_dir="gs://kubric-public/tfds", with_info=True
    )
    save_dataset(ds, ds_info, args.dataset)


if __name__ == "__main__":
    main()
