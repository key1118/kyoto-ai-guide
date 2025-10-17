import glob
import os


def load_documents(folder_path="data"):
    docs = []
    for path in glob.glob(os.path.join(folder_path, "*.txt")):
        with open(path, "r", encoding="utf-8") as f:
            docs.append(f.read().strip())
    return docs
