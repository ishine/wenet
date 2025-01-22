import os
import shutil

DATASET_DIR = "/root/DSR/wenet/kespeech/data/Subdialects"
PREFIX = "/root/DSR/KeSpeech/KeSpeech"


def make_abs_path():
    for set in os.listdir(DATASET_DIR):
        with open(os.path.join(DATASET_DIR, set, "wav.scp"), "r") as fin, open(
            os.path.join(DATASET_DIR, set, "wav.scp.abs"), "w"
        ) as fout:
            for line in fin.readlines():
                key, rel_path = line.strip().split(maxsplit=1)
                abs_path = os.path.join(PREFIX, rel_path)
                fout.write(f"{key} {abs_path}\n")
        shutil.copy(
            os.path.join(DATASET_DIR, set, "wav.scp"),
            os.path.join(DATASET_DIR, set, "wav.scp.bak"),
        )
        shutil.move(
            os.path.join(DATASET_DIR, set, "wav.scp.abs"),
            os.path.join(DATASET_DIR, set, "wav.scp"),
        )


def main():
    make_abs_path()


if __name__ == "__main__":
    main()
