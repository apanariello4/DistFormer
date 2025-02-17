import configparser
import random
from pathlib import Path
from random import shuffle

import pandas as pd

from scripts.motsynth_info import MOTSYNTH_TRAIN_AND_VAL_SEQUENCES

motsynth_motchallenge_format_path = Path("./data/MOTSynth/MOTChallengeFormat")

random.seed(42)
shuffle(MOTSYNTH_TRAIN_AND_VAL_SEQUENCES)

df = pd.DataFrame(columns=["sequence", "weather", "is_night", "is_moving"])

selected_attributes = {}

for seq in MOTSYNTH_TRAIN_AND_VAL_SEQUENCES:
    seq_info_path = motsynth_motchallenge_format_path / f"{seq}/seqinfo.ini"

    config = configparser.ConfigParser()
    config.read(seq_info_path)
    assert config["Sequence"]["name"] == seq
    weather = config["Sequence"]["weather"]
    is_night = config["Sequence"]["isNight"]
    is_moving = config["Sequence"]["isMoving"]
    key = (weather, is_night, is_moving)
    if key not in selected_attributes:
        selected_attributes[key] = seq
    df.loc[len(df)] = [seq, weather, is_night, is_moving]

weathers = df["weather"].unique()
is_nights = df["is_night"].unique()
is_movings = df["is_moving"].unique()

print(f"weather = {weathers}" f"is_night = {is_nights}" f"is_moving = {is_movings}")

print(selected_attributes)
sequences = tuple(selected_attributes.values())
print(sequences)
assert len(sequences) == len(set(sequences))
sequences = sorted(list(sequences))
print(sequences)
