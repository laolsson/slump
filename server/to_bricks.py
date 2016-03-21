import sys
import time
import math
import os
import json
from PIL import Image


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print '%r (%r, %r) %2.5f sec' % \
              (method.__name__, args, kw, te - ts)
        return result

    return timed


# @timeit
def apply_effect(image, overlay_red, overlay_green, overlay_blue):
    '''Small function to apply an effect over an entire image'''
    channels = image.split()

    r = channels[0].point(lambda color: overlay_red - 100 if (133 - color) > 100 else (
    overlay_red + 100 if (133 - color) < -100 else overlay_red - (133 - color)))
    g = channels[1].point(lambda color: overlay_green - 100 if (133 - color) > 100 else (
    overlay_green + 100 if (133 - color) < -100 else overlay_green - (133 - color)))
    b = channels[2].point(lambda color: overlay_blue - 100 if (133 - color) > 100 else (
    overlay_blue + 100 if (133 - color) < -100 else overlay_blue - (133 - color)))

    channels[0].paste(r)
    channels[1].paste(g)
    channels[2].paste(b)

    return Image.merge(image.mode, channels)


def generate_images(out_dir, size, colors):
    lego_image = Image.open('lego.png')
    lego_image = lego_image.resize((size, size), Image.ANTIALIAS)
    for c in colors['colours']:
        name = c['name']
        number = c['number']
        r = int(c['R'])
        g = int(c['G'])
        b = int(c['B'])
        lm = lego_image.copy()
        lm = apply_effect(lego_image, r, g, b)
        lm.save(out_dir + os.path.sep + 'lego_' + number + '.png')



def main(argv):
    out_dir = argv[1]
    size = int(argv[2])
    color_file = argv[3]

    print out_dir, size

    colors = json.loads(open(color_file).read())
    print colors

    generate_images(out_dir, size, colors)


if __name__ == "__main__":
    main(sys.argv)
