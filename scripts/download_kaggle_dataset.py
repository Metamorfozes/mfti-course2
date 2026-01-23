import argparse
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a Kaggle dataset.")
    parser.add_argument(
        "--dataset",
        default="hsankesara/flickr-image-dataset",
        help="Kaggle dataset identifier.",
    )
    parser.add_argument(
        "--out",
        default="artifacts/datasets/flickr30k",
        help="Output directory for the dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        args.dataset,
        "-p",
        args.out,
        "--unzip",
    ]
    subprocess.run(cmd, check=True)
    print(args.out)


if __name__ == "__main__":
    main()
