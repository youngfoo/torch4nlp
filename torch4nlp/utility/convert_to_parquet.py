#!/usr/bin/env python
#-*- coding: utf-8 -*-

from absl import logging, app, flags
import os
import json
import random
import collections
import pandas as pd
import pyarrow.parquet as pq

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_file", None, "The input file to be converted."
)

flags.DEFINE_string(
    "output_dir", None, "The prefix of output file."
)

flags.DEFINE_integer(
    "output_file_num", 10, "The number of output file."
)

flags.DEFINE_integer(
    "max_output_num", 1000000, "The max number of output line."
)

flags.DEFINE_bool(
    "need_shuffle", True, "Whether needs shuffle"
)

def main(_):
    print('input: ', FLAGS.input_file)
    print('output: ', FLAGS.output_dir)
    print('output_file_num: ', FLAGS.output_file_num)
    print('max_output_num: ', FLAGS.max_output_num)
    print('need_shuffle: ', FLAGS.need_shuffle)
    sessions = collections.defaultdict(list)
    lines = []
    with open(FLAGS.input_file) as f:
        for idx, line in enumerate(f):
            if idx >= FLAGS.max_output_num:
                break
            lines.append(line)
    if FLAGS.need_shuffle:
        random.shuffle(lines)
    for idx, line in enumerate(lines):
        jline = json.loads(line)
        pairs = [
            {
                "prompt": pair["prompt"],
                "response": pair["response"],
                "response_loss_mask": pair.get("response_loss_mask", 1)
            } for pair in jline["session"]
        ]
        sessions["session"].append(pairs)
        sessions["system"].append(jline.get("system", []))
        sessions["file_idx"].append(idx % FLAGS.output_file_num)
    df = pd.DataFrame(sessions)
    print(len(df))
    os.makedirs(FLAGS.output_dir)
    for partition_value, partition_df in df.groupby('file_idx'):
        partition_file_path = f'{FLAGS.output_dir}/{partition_value}.parquet'
        partition_df = partition_df.drop('file_idx', axis=1)
        partition_df.to_parquet(partition_file_path, index=False)

if __name__ == "__main__":
    app.run(main)

# python3 convert2parquet.py --input_file=large_sft_clean_shuf_464w.json --output_dir=large_sft_clean_shuf_50w --output_file_num=10 --max_output_num=500000
