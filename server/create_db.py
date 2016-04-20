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

def create_db(image_dir):
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
            img = img.resize((5, 5), Image.ANTIALIAS)
            #img.save(f + '_small.png')
            d2d = img.load()
            print 'Processing image ', f, 'file ', count, ' out of ', total_count
            points = []
            for y in range(5):
                for x in range(5):
                    points.append(d2d[x, y][0:3])
            db[f] = points
            count = count + 1
        except Exception, e:
            print e

    return db

def main(argv):
    image_dir = argv[1]
    db_name = argv[2]

    db = create_db(image_dir)
    print db
    with open(db_name, 'w') as f:
        print 'db'
        f.write(json.dumps(db))

if __name__ == "__main__":
    main(sys.argv)
