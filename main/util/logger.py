import os

from .make_dirs import make_dirs


class Logger:

    def __init__(self, path_dir, name):
        make_dirs(path_dir)
        self.file_path = os.path.join(path_dir, name)

    def __call__(self, input):
        input = str(input)
        with open(self.file_path, "a") as f:
            f.writelines(input + "\n")
        print(input)
