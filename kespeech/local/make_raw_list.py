#!/usr/bin/env python3

import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--segments", default=None, help="segments file")
    parser.add_argument("wav_file", help="wav file")
    parser.add_argument("text_file", help="text file")
    parser.add_argument("utt2subdialect_file", help="utt2subdialect file")
    parser.add_argument("output_file", help="output list file")
    args = parser.parse_args()

    wav_table = {}
    with open(args.wav_file, "r", encoding="utf8") as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            wav_table[arr[0]] = arr[1]

    if args.segments is not None:
        segments_table = {}
        with open(args.segments, "r", encoding="utf8") as fin:
            for line in fin:
                arr = line.strip().split()
                assert len(arr) == 4
                segments_table[arr[0]] = (arr[1], float(arr[2]), float(arr[3]))

    with open(args.text_file, "r", encoding="utf8") as fin, open(
        args.utt2subdialect_file, "r", encoding="utf8"
    ) as fdialect, open(args.output_file, "w", encoding="utf8") as fout:
        for line, dialect in zip(fin, fdialect):
            arr = line.strip().split(maxsplit=1)
            key = arr[0]
            txt = arr[1] if len(arr) > 1 else ""
            arr2 = dialect.strip().split(maxsplit=1)
            key2 = arr2[0]
            subdialect = arr2[1] if len(arr2) > 1 else ""
            assert key == key2
            if args.segments is None:
                assert key in wav_table
                wav = wav_table[key]
                line = dict(key=key, wav=wav, txt=txt, subdialect=subdialect)
            else:
                assert key in segments_table
                wav_key, start, end = segments_table[key]
                wav = wav_table[wav_key]
                line = dict(
                    key=key,
                    wav=wav,
                    txt=txt,
                    subdialect=subdialect,
                    start=start,
                    end=end,
                )
            json_line = json.dumps(line, ensure_ascii=False)
            fout.write(json_line + "\n")
