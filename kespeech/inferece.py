import os

import wenet
from tqdm import tqdm


model = wenet.load_model("chinese")

KESPEECH_ROOT = "/root/data/KeSpeech/KeSpeech"
LOG_DIR = "../logs"


def recognize():
    subdialect_path = os.path.join(KESPEECH_ROOT, "Subdialects")
    for subdialect in tqdm(os.listdir(subdialect_path)):
        if subdialect == "Mandarin":
            continue
        with open(
            os.path.join(subdialect_path, subdialect, "test_phase1", "wav.scp"), "r"
        ) as f_wav_scp, open(
            os.path.join(LOG_DIR, "recognition_result", f"{subdialect}.log"), "w+"
        ) as f_log:
            for line in tqdm(f_wav_scp.readlines(), desc=f"{subdialect}"):
                (audio, path) = line[:-1].split("\t")
                result = model.transcribe(path)
                f_log.write(f"{audio}\t{result['text']}\n")


if __name__ == "__main__":
    recognize()
