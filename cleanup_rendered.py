from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm


def clean_dir(path: Path):
    """Remove irrelevant files from single sequence directory."""
    for f in path.iterdir():
        if ".json" in f.name or "rgba" in f.name:
            continue
        f.unlink()


def clean_dataset(path: Path):
    """Remove irrelevant files from dataset directory."""
    for d in tqdm(path.iterdir()):
        if d.is_dir():
            clean_dir(d)


def main():
    parser = ArgumentParser(
        description="Cleanup a rendered dataset."
    )
    parser.add_argument("directory", type=str)
    args = parser.parse_args()

    path = Path(args.directory)
    clean_dataset(path)


if __name__ == "__main__":
    main()
