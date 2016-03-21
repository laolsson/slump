import sys
import time
import math
import os
import json
from PIL import Image


def step(files, dirname, names):
    for name in names:
        if name.lower().endswith('jpg') or name.lower().endswith('png'):
            files.append(dirname + '/' + name)

def process_files(image_dir, out_dir):
    from os import listdir
    from os.path import isfile, join
    files = []
    os.path.walk(image_dir, step, files)

    db = {}
    total_count = len(files)
    count = 0

    # now for each file
    for f in files:
        try:
            img = Image.open(f)
            img = img.resize((200, 200), Image.ANTIALIAS)
            img.save(out_dir + f.split('/')[-1])
	    print 'Processing image ', f, 'file ', count, ' out of ', total_count
            count = count + 1
        except Exception, e:
            print e

    return db

def main(argv):
    image_dir = argv[1]
    out_dir = argv[2]

    process_files(image_dir, out_dir)


if __name__ == "__main__":
    main(sys.argv)
