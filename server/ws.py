import os
import sys
import base64
import bottle
from bottle import request, Bottle, abort
import time
from threading import Thread
import gevent



# Make sure the search path is setup correctly
global_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
print global_base_dir
if global_base_dir not in sys.path:
	sys.path.append(global_base_dir)

import processing.process_image

app = Bottle()

@app.route('/js/<filename>', method=['GET'])
def serve_static_file(filename):
	return bottle.static_file(filename, root='../client/js/')

	
@app.route('/click/<id>', method=['POST'])
def image(id):
	x = request.query.getunicode('x')
	y = request.query.getunicode('y')
	print 'iiiiiiiiiii', id, x, y


@app.route('/image', method=['GET'])
def image():
	print 'GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG'
	

@app.route('/websocket/<id>')
def handle_websocket(id):
	wsock = request.environ.get('wsgi.websocket')
	if not wsock:
		abort(400, 'Expected WebSocket request.')
	counter = 0
	image_cache = {}
	while True:
		try:
			image_name = ['../images/anneliese.jpg', '../images/ao_lego.jpg', '../images/lego.png'][counter % 3]
			#if image_name not in image_cache:
			#	i = open(image_name, 'rb').read()
			#	image_cache[image_name] = i
			#im = image_cache[image_name]
			processing.process_image.process_image(image_name, 'tmp.jpg')
			im = open('tmp.jpg', 'rb').read()
			encoded_string = base64.b64encode(im)
			print 'EEEEE', len(encoded_string), id
			wsock.send(encoded_string)
			gevent.sleep(1)
			counter = counter + 1
		except WebSocketError, we:
			print 'session died', str(we)
			break


from gevent.pywsgi import WSGIServer
from geventwebsocket import WebSocketError
from geventwebsocket.handler import WebSocketHandler
server = WSGIServer(("0.0.0.0", 8090), app,
		    handler_class=WebSocketHandler)
server.serve_forever()