import os
import logging
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

KESPEECH_ROOT = "/root/data/KeSpeech/KeSpeech"
OUTPUT_DIR = os.path.join(KESPEECH_ROOT, "Subdialects")

SUBDIALECTS = [
    "Mandarin",
    "Beijing",
    "Jiang-Huai",
    "Jiao-Liao",
    "Ji-Lu",
    "Lan-Yin",
    "Northeastern",
    "Southwestern",
    "Zhongyuan",
]


def create_dirs():
    for subdialect in SUBDIALECTS:
        subdialect_path = os.path.join(OUTPUT_DIR, subdialect)
        sub_dataset_dir = ["dev_phase1", "test_phase1", "train_phase1"]
        for target_dir in sub_dataset_dir:
            target_path = os.path.join(subdialect_path, target_dir)
            if os.path.exists(target_path):
                logging.info(f"path is already exist.({target_path})")
            else:
                os.makedirs(target_path)
                logging.info(f"path is created.({target_path})")


def split_subdialect():
    prefix = os.path.join(KESPEECH_ROOT, "Tasks", "SubdialectID")
    for sub_dataset_name in os.listdir(prefix):
        logging.info(f"Start processing {sub_dataset_name}")
        sub_dataset_path = os.path.join(prefix, sub_dataset_name)
        with open(
            os.path.join(sub_dataset_path, "utt2subdialect"), "r"
        ) as f_utt2subdialect, open(
            os.path.join(sub_dataset_path, "text"), "r"
        ) as f_text, open(
            os.path.join(sub_dataset_path, "wav.scp"), "r"
        ) as f_wav_scp:
            utt2subdialects = f_utt2subdialect.readlines()
            utt2texts = f_text.readlines()
            utt2paths = f_wav_scp.readlines()
            for utt2subdialect, utt2text, utt2path in tqdm(
                zip(utt2subdialects, utt2texts, utt2paths),
                total=len(utt2subdialects),
            ):
                # read original files
                (audio_id, dialect) = utt2subdialect[:-1].split(" ")
                (audio_id2, text) = utt2text[:-1].split(" ")
                (audio_id3, audio_path) = utt2path[:-1].split(" ")
                if not audio_id == audio_id2 == audio_id3:
                    logging.error(
                        f"Audio id dismatched.({audio_id}, {audio_id2}, {audio_id3})"
                    )

                # write target files
                dialect_subdir = os.path.join(OUTPUT_DIR, dialect, sub_dataset_name)
                text_path = os.path.join(dialect_subdir, "text")
                wav_scp_path = os.path.join(dialect_subdir, "wav.scp")
                with open(text_path, "a+") as f_target_text, open(
                    wav_scp_path, "a+"
                ) as f_target_wav_scp:
                    f_target_text.write(f"{audio_id}\t{text}\n")
                    f_target_wav_scp.write(
                        f"{audio_id}\t{os.path.join(KESPEECH_ROOT,audio_path)}\n"
                    )
        logging.info(f"Processing {sub_dataset_name} is completed.")


def main():
    create_dirs()
    split_subdialect()


if __name__ == "__main__":
    main()
