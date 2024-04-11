def create_path_label_list(df):
    path_label_list = []
    for _, row in df.iterrows():
        path = row["path"]
        label = row["label"]
        path_label_list.append(("../" + path, label))
    return path_label_list
