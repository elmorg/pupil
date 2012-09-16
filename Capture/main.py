import os, sys
import numpy as np 
import cv2
from time import sleep
from multiprocessing import Process, Pipe, Event
from multiprocessing.sharedctypes import RawValue, Value # RawValue is shared memory without lock, handle with care, this is usefull for ATB it needs cTypes
from eye import eye
from world import world
from player import player
from methods import Temp
from array import array
from struct import unpack, pack
# import pyaudio
# import waveatb.
# from audio import normalize, trim, add_silence

from ctypes import *

def main():
	
 	# assing the right id to the cameras
	eye_src = 0
	world_src = 1


	audio = False

	# when using the logitech h264 compression camera
	# you can't run world camera in its own process
	# it must reside in the main loop
	# this is all taken care of by setting this to true
	muliprocess_cam = 1
	
	#use video for debugging
	use_video = 1
 	
	if use_video:
		eye_src = "/Users/mkassner/MIT/pupil_google_code/wiki/videos/green_eye_VISandIR_2.mov" # unsing a path to a videofiles allows for developement without a headset.
		world_src = 0

	if muliprocess_cam:
		world_id = world_src
		world_src, world_feed = Pipe()

	# create shared globals 
	g_pool = Temp()
	g_pool.pupil_x = Value('d', 0.0)
	g_pool.pupil_y = Value('d', 0.0)
	g_pool.pattern_x = Value('d', 0.0) 
	g_pool.pattern_y = Value('d', 0.0) 
	g_pool.frame_count_record = Value('i', 0)
	g_pool.calibrate = RawValue(c_bool, 0)
	g_pool.cal9 = RawValue(c_bool, 0)
	g_pool.cal9_stage = Value('i',0)	
	g_pool.cal9_step = Value('i',0)
	g_pool.cal9_circle_id = RawValue('i',0)
	g_pool.pos_record = Value(c_bool, 0)
	g_pool.eye_rx, g_pool.eye_tx = Pipe(False)
	g_pool.audio_record = Value(c_bool,False)
	g_pool.audio_rx, g_pool.audio_tx = Pipe(False)
	g_pool.player_refresh = Event()
	g_pool.play = RawValue(c_bool,0)
	g_pool.quit = RawValue(c_bool,0)

	p_eye = Process(target=eye, args=(eye_src, g_pool))
	p_world = Process(target=world, args=(world_src,g_pool))
	p_player = Process(target=player, args=(g_pool,))

	# Audio:
	# src=3 for logitech, rate=16000 for logitech 
	# defaults for built in MacBook microphone
	if audio: p_audio = Process(target=record_audio, args=(audio_rx,audio_record,3)) 

	p_eye.start()
	sleep(.3)
	p_world.start()
	sleep(.3)
	p_player.start()
	sleep(.3)
	if audio: p_audio.start()
	
	if(muliprocess_cam):
		grab(world_feed,world_id,g_pool)


	p_eye.join()
	p_world.join()
	p_player.join()
	if audio: p_audio.join()

	print "main exit"

def grab(pipe,src_id,g_pool):
	"""grab:
		- Initialize a camera feed
		-this is needed for certain cameras that have to run in the main loop.
		- it pushed image frames to the capture class 
			that it initialize with one pipeend as the source
	"""

	quit = g_pool.quit
	cap = cv2.VideoCapture(src_id)
	size = pipe.recv()
	cap.set(3, size[0])
	cap.set(4, size[1])
			
	while not quit.value:
		try:
			pipe.send(cap.read())
		except:
			pass
	print "Local Grab exit"

def xmos_grab(pipe,src_id,g_pool):
	"""grab:
		- Initialize a camera feed
		-this is needed for certain cameras that have to run in the main loop.
		- it pushed image frames to the capture class 
			that it initialize with one pipeend as the source
	"""

	quit = g_pool.quit
	cap = cv2.VideoCapture(src_id)
	size = pipe.recv()
	size= size[::-1] # swap sizes as numpy is row first

	cam = cam_interface()
	buffer = np.zeros(size, dtype=np.uint8) #this should always be a multiple of 4
	cam.aptina_setWindowSize(cam.id0,(size[1],size[0])) #swap sizes back 
	cam.aptina_setWindowPosition(cam.id0,(240,100))
	cam.aptina_LED_control(cam.id0,Disable = 0,Invert =0)
	cam.aptina_AEC_AGC(cam.id0,1,1) # Auto Exposure Control + Auto Gain Control
	cam.aptina_HDR(cam.id0,1)
	while not quit.value:
		if cam.get_frame(id,buffer): #returns True on sucess
			try:
				pipe.send(True, buffer)
			except:
				pass
	cam.release()	
	print "Local Grab exit"



if __name__ == '__main__':
	main()



