import os


def make_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print("Successfully make dirs: {}".format(dir))
    else:
        print("Existed dirs: {}".format(dir))
