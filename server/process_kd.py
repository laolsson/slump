import sys
import time
import math
import os
import json
from PIL import Image

import scipy
import scipy.spatial

import numpy as np


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print '%r (%r, %r) %2.5f sec' % \
              (method.__name__, args, kw, te - ts)
        return result

    return timed

import cProfile

def do_cprofile(func):
    def profiled_func(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats()
    return profiled_func


# just resize once so that we get 5x5 pixels for tile_width and use that
def get_average(img, xt, yt, tw):
    #print xt, yt, tw
    d2d = img.load()
    avg = []
    for y in range(yt):
        for x in range(xt):
            xtw = x * tw
            ytw = y * tw
            si = img.crop((xtw, ytw, xtw + tw, ytw + tw))
            si.load()
            si.resize((5, 5), Image.ANTIALIAS)
            d2d = si.load()
            points = []
            for yy in range(5):
                for xx in range(5):
                    points.append(d2d[xx, yy])
            avg.append(points)

    return avg


cache = {}
used = {}


def diff_points(points, other_points):
    diff = 0
    #for i in [12]:
    for i in [0, 4, 12, 20, 24]:  # range(25):
    #for i in range(25):
        diff = diff + math.fabs(points[i][0] - other_points[i][0]) + math.fabs(
            points[i][1] - other_points[i][1]) + math.fabs(points[i][2] - other_points[i][2])

    return diff


import random
class NSmallest:
    def __init__(self, n, sort_key=None, start_val=999999999999):
        self.n = n
        self.mins = []
        self.sort_key = sort_key
        for i in range(self.n):
            self.mins.append((start_val, 'XXXXXWrong'))

    def add_value(self, v):
        if v[0] < self.mins[-1][0]:
            self.mins.append(v)
            self.mins.sort(key=self.sort_key)
            self.mins = self.mins[:self.n]


    def get_random(self):
         return random.choice(self.mins)


def get_image_new(img_db, average, tile_width):
    global cache, used
    nsmallest = NSmallest(10, lambda x: x[0])
    for k, v in img_db.iteritems():
        diff = diff_points(v, average)
        nsmallest.add_value((diff, k))

    closest = nsmallest.get_random()[1]

    if closest not in cache:
        img = Image.open(closest)
        img = img.resize((tile_width, tile_width), Image.ANTIALIAS)
        cache[closest] = [img, 0]

    cache[closest][1] = cache[closest][1] + 1
    if cache[closest][1] > 100000:
        print 'Used'
        used[closest] = closest

    return cache[closest][0]


def get_image(img_db, average, tile_width):
    global cache, used
    closest = None
    first = True
    close_sum = 0
    to_remove = []
    for k, v in img_db.iteritems():
        if k in used:
            to_remove.append(k)
            continue
        diff = diff_points(v, average)
        if first:
            first = False
            closest = k
            close_sum = diff
        else:
            if diff < close_sum:
                close_sum = diff
                closest = k

    #print closest

    if closest not in cache:
        img = Image.open(closest)
        img = img.resize((tile_width, tile_width), Image.ANTIALIAS)
        cache[closest] = [img, 0]

    cache[closest][1] = cache[closest][1] + 1
    if cache[closest][1] > 100000:
        print 'Used'
        used[closest] = closest

    for k in to_remove:
        print 'dd', k
        del img_db[k]

    return cache[closest][0]

def get_image_threaded(img_db, average, tile_width):
    global cache, used
    closest = None
    first = True
    close_sum = 0
    to_remove = []
    for k, v in img_db.iteritems():
        if k in used:
            to_remove.append(k)
            continue
        diff = diff_points(v, average)
        if first:
            first = False
            closest = k
            close_sum = diff
        else:
            if diff < close_sum:
                close_sum = diff
                closest = k

        return closest





# keep track of N smallest distances and select between them:
#mins = items[:n]
#mins.sort()
#for i in items[n:]:
 #   if i < mins[-1]:
 #       mins.append(i)
 #       mins.sort()
 #       mins= mins[:n]


 # For kd use query(find, k), where k is k number of items to return
 # then pick one out of k

def get_image_kd(kd_db, images, average, tile_width):
    global cache, used
    #find = np.array(average[12])# + average[4] + average[12] + average[20] + average[24])
    find = np.array(average[0] + average[4] + average[12] + average[20] + average[24])
    res = kd_db.query(find, 10)

    closest = images[random.choice(res[1])]

    if closest not in cache:
        img = Image.open(closest)
        img = img.resize((tile_width, tile_width), Image.ANTIALIAS)
        cache[closest] = [img, 0]

    cache[closest][1] = cache[closest][1] + 1
    if cache[closest][1] > 100000:
        print 'Used'
        used[closest] = closest

    return cache[closest][0]


def apply_effect(image, overlay_red, overlay_green, overlay_blue):
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


def average_points(points):
    r, g, b = 0, 0, 0
    for p in points:
        r = r + p[0]
        g = g + p[1]
        b = b + p[2]

    num = len(points)
    return [int(r/num), int(g/num), int(b/num)]

#@do_cprofile
def process_image(image_path, img_db, out_file, tile_width=10, apply_ef=False):
    global cache
    print 'process ', image_path
    img = Image.open(image_path)
    x_tiles = img.size[0] / tile_width
    y_tiles = img.size[1] / tile_width
    average = get_average(img, x_tiles, y_tiles, tile_width)

    total_tiles = y_tiles * x_tiles
    tile_count = 0

    for y in range(y_tiles):
        for x in range(x_tiles):
            if tile_count % 100 == 0:
                print tile_count, ' out of ', total_tiles, ' cache size:', len(cache)
            lm = get_image_new(img_db, average[x + y * x_tiles], tile_width=tile_width)
            if apply_ef:
                av = average_points(average[x + y * x_tiles])
                lm = apply_effect(lm, *av)
            img.paste(lm, (x * tile_width, y * tile_width))
            tile_count = tile_count + 1

    print 'cache size', len(cache)

    #img.save(out_file)


def get_image_threaded(img_db, average, tile_width):
    global cache, used
    closest = None
    first = True
    close_sum = 0
    to_remove = []
    for k, v in img_db.iteritems():
        diff = diff_points(v, average)
        if first:
            first = False
            closest = k
            close_sum = diff
        else:
            if diff < close_sum:
                close_sum = diff
                closest = k


    return closest

def get_image_kd_threaded(kd_db, average, tile_width):
    global cache, used
    #find = np.array(average[12])# + average[4] + average[12] + average[20] + average[24])
    find = np.array(average[0] + average[4] + average[12] + average[20] + average[24])
    res = kd_db.query(find, 5)

    return image_names[random.choice(res[1])]


def process_im_threaded(name, out_image, out, y_start, y_rows, average, x_tiles, y_tiles, tile_width, img_db, apply_ef):
    tile_count = 0
    print 'thread ', name, y_start, tile_width

    img = Image.new("RGB", (x_tiles * tile_width, y_rows * tile_width))

    image_cache = {}

    for y in range(y_start, y_start + y_rows):
        for x in range(x_tiles):
	    #print 'XXX', name, 'XXX  ', x, y
            #if tile_count % 100 == 0:
            #    print tile_count, name
            image_name = get_image_kd_threaded(img_db, average[x + y * x_tiles], tile_width=tile_width)

	    if image_name not in image_cache:
	        #out.append(image_name)
		im = Image.open(image_name)#images[image_name]
		im = im.resize((tile_width, tile_width), Image.ANTIALIAS)
		image_cache[image_name] = im
	    else:
		im = image_cache[image_name]
	    img.paste(im, (x * tile_width, y * tile_width - y_start * tile_width))

	    #out_image.append(im)
	    #print out_image
	    #return
	    tile_count = tile_count + 1

    img.save(name + '.png')
    #out_image.append(img)


import multiprocessing
#@do_cprofile
def process_image_threaded(image_path, img_db, out_file, tile_width=10, apply_ef=False):
    global cache
    print 'process ', image_path, tile_width

    img = Image.open(image_path)
    x_tiles = img.size[0] / tile_width
    y_tiles = img.size[1] / tile_width
    start_average_time = time.time()
    average = get_average(img, x_tiles, y_tiles, tile_width)
    print 'Done average ', time.time() - start_average_time

    total_tiles = y_tiles * x_tiles
    print 'process ', x_tiles, y_tiles, total_tiles

    processes = []
    outs = []
    out_images = []
    # Split up in to N processes
    n_processes = multiprocessing.cpu_count() - 1

    # how many rows per process
    rows_per_process = y_tiles / n_processes

    start_process_time = time.time()

    for p in range(n_processes):
        print p, n_processes
        out = multiprocessing.Manager().list()
	out_image = multiprocessing.Manager().list()
        process = multiprocessing.Process(target=process_im_threaded, args=('%d' % p, out_image, out, p * rows_per_process, rows_per_process, average, x_tiles, y_tiles, tile_width, img_db, apply_ef))
        process.start()
        processes.append(process)
        outs.append(out)
	out_images.append(out_image)

    # wait for all rpcoesses to be done
    for p in processes:
        p.join()

    print 'Done processing ', time.time() - start_process_time

    for o in outs:
        print 'LLLLLLLLLLLLLLL', len(o)

    for o in out_images:
        print 'HHHHHHHH', o
	#print o

    # Now create the actual out image
    y = 0
    x = 0

    start_image_time = time.time()

    img = Image.new("RGB", (x_tiles * tile_width, y_tiles * tile_width))

    for n in range(n_processes):
        im = Image.open(str(n) + '.png')
	img.paste(im, (0, rows_per_process * tile_width * n))

    # for o in outs:
        # for i in o:
	    # #print y, x
	    # if i not in cache:
	        # im = Image.open(i)
                # im = im.resize((tile_width, tile_width), Image.ANTIALIAS)
		# cache[i] = im
            # else:
	        # im = cache[i]
	    # img.paste(im, (x * tile_width, y * tile_width))

	    # if x == x_tiles - 1:
	        # y = y + 1
		# x = 0
	    # else:
	        # x = x + 1

    print 'Done create image time', time.time() - start_image_time

    start_save_time = time.time()
    img.save(out_file)
    print 'Done save ', time.time() - start_save_time, len(cache)
    #img.save(out_file)


#@do_cprofile
def process_image_kd(image_path, kd_db, images, out_file, tile_width=10, apply_ef=False):
    import time
    s=time.time()
    print 'process ', image_path
    img = Image.open(image_path)
    x_tiles = img.size[0] / tile_width
    y_tiles = img.size[1] / tile_width
    average = get_average(img, x_tiles, y_tiles, tile_width)

    total_tiles = y_tiles * x_tiles
    tile_count = 0
    start = time.time()
    for y in range(y_tiles):
        for x in range(x_tiles):
            if tile_count % 100 == 0:
                print tile_count, ' out of ', total_tiles, ' cache size:', len(cache), time.time() - start
		start = time.time()
            lm = get_image_kd(kd_db, images, average[x + y * x_tiles], tile_width=tile_width)
            if apply_ef:
                av = average_points(average[x + y * x_tiles])
                lm = apply_effect(lm, *av)
            img.paste(lm, (x * tile_width, y * tile_width))
            tile_count = tile_count + 1

    print 'cache size', len(cache), time.time()-s

    img.save(out_file)


def load_db(db_name):
    db = None
    with open(db_name) as f:
        db = json.loads(f.read())

    return db

tree=None
images={}
image_names = []
def create_kd_db(db, tile_width):
    global tree, images, image_names
    list = []
    i =0
    for k, v in db.iteritems():
        #print k, v
        #print v[12]
	#li = v[12]
	li = v[0] + v[4] + v[12] + v[20] + v[24]
	if len(li) != 15:
	    print 'MMMMMMMMMMMMMMMMMMMMMMMMMM'
	    continue
        list.append(li)
	#im = Image.open(k)
	#print dir(im)
	#im = im.resize((tile_width, tile_width), Image.ANTIALIAS)
        #images[k] = im.copy()
	#print dir(im)
	#im.close()
	image_names.append(k)
        i = i + 1

    #print list[604], len(list[604]), len(list)
    print list, len(list)
    tree = scipy.spatial.KDTree(list)
    return tree


def main(argv):
    input_image = argv[1]
    tile_width = int(argv[2])
    out_image = argv[3]
    db_name = argv[4]

    print input_image, tile_width, out_image, db_name

    import time
    s=time.time()

    db = load_db(db_name)

    kd_db = create_kd_db(db, tile_width)

    multiprocessing.freeze_support()

    #import time
    #s=time.time()
    #process_image(input_image, db, out_image + '.orig.png', tile_width, False)
    #print 'Orig', time.time()-s

    import time
    s=time.time()
    process_image_threaded(input_image, kd_db, out_image + '.orig.png', tile_width, False)
    print 'Orig', time.time()-s

    #import time
    #s=time.time()
    #process_image_threaded(input_image, kd_db, out_image + '.orig.png', tile_width, False)
    #print 'Orig', time.time()-s

    #global cache
    #cache = {}

    #import time
    #s=time.time()
    #process_image_kd(input_image, kd_db, images, out_image, tile_width, False)
    #print 'New', time.time()-s

    #import time
    #s=time.time()
    #process_image_kd(input_image, kd_db, images, out_image, tile_width, False)
    #print 'New', time.time()-s

if __name__ == "__main__":
    main(sys.argv)
