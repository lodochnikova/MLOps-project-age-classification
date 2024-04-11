import os

import pandas as pd

data = pd.read_csv("../faces/train.csv")
class_names = ["YOUNG", "MIDDLE", "OLD"]
N = list(range(len(class_names)))
normal_mapping = dict(zip(class_names, N))
reverse_mapping = dict(zip(N, class_names))
data["label"] = data["Class"].map(normal_mapping)
dir0 = "faces/Train"
data["path"] = data["ID"].apply(lambda x: os.path.join(dir0, x))

data.to_csv("data.csv", index=False)
