'''
a script that will replay pupil server messages to a osc server.

as implemented here only the pupil_norm_pos is relayed.
implementeing other messages to be send as well is a matter of renaming the vaiables.

installing pyOSC:

git clone git://gitorious.org/pyosc/devel.git pyosc
cd pyosc
python setup.py install (may need sudo)
'''


#zmq setup
#import time
import zmq
port = "5000"
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://127.0.0.1:"+port)
#filter by messages by stating string 'STRING'. '' receives all messages
socket.setsockopt(zmq.SUBSCRIBE, '')

#osc setup
from OSC import OSCClient, OSCMessage

client = OSCClient()
client.connect( ("localhost", 7111) )
def bridge():
    while True:
        msg = socket.recv()
        print "raw msg:\n", msg

        items = msg.split("\n")
        msg_type = items.pop(0)
        items = dict([i.split(':') for i in items[:-1] ])
        
        if msg_type == 'Pupil':
            if 'norm_pos' in items:
                pupil_x2,pupil_y2  = map(float,items['norm_pos'][1:-1].split(','))

            client.send( OSCMessage("/pupil/norm_pos1", ('test',items['norm_pos'])))

        else:
            # process event msgs from plugins here
            pass
