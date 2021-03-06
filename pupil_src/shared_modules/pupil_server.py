'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from plugin import Plugin

import numpy as np
from pyglui import ui
import zmq

from OSC import OSCClient, OSCMessage

import logging
logger = logging.getLogger(__name__)
client = OSCClient()
client.connect( ("localhost", 7111) )


class Pupil_Server(Plugin):
    """pupil server plugin"""
    def __init__(self, g_pool,address="tcp://127.0.0.1:5000"):
        super(Pupil_Server, self).__init__(g_pool)
        self.order = .9
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.address = ''
        self.set_server(address)
        self.menu = None

        self.exclude_list = ['ellipse','pos_in_roi','major','minor','axes','angle','center']

    def init_gui(self):
        if self.g_pool.app == 'capture':
            self.menu = ui.Growing_Menu("Pupil Broadcast Server")
            self.g_pool.sidebar.append(self.menu)
        elif self.g_pool.app == 'player':
            self.menu = ui.Scrolling_Menu("Pupil Broadcast Server")
            self.g_pool.gui.append(self.menu)

        help_str = "Pupil Message server: Using ZMQ and the *Publish-Subscribe* scheme"
        self.menu.append(ui.Info_Text(help_str))
        self.menu.append(ui.Text_Input('address',self,setter=self.set_server,label='Address'))
        self.menu.append(ui.Button('Close',self.close))


    def deinit_gui(self):
        if self.menu:
            if self.g_pool.app == 'capture':
                self.g_pool.sidebar.remove(self.menu)
            elif self.g_pool.app == 'player':
                self.g_pool.gui.remove(self.menu)
            self.menu = None


    def set_server(self,new_address):
        try:
            self.socket.unbind(self.address)
            logger.debug('Detached from %s'%self.address)
        except:
            pass
        try:
            self.socket.bind(new_address)
            self.address = new_address
            logger.debug('Bound to %s'%self.address)

        except zmq.ZMQError as e:
            logger.error("Could not set Socket: %s. Reason: %s"%(new_address,e))

    def update(self,frame,events):
        for p in events.get('pupil_positions',[]):
            msg = "Pupil\n"
            for key,value in p.iteritems():
                if key not in self.exclude_list:
                    msg +=key+":"+str(value)+'\n'
            self.socket.send( msg )


        for g in events.get('gaze_positions',[]):
            msg = "Gaze\n"
            for key,value in g.iteritems():
                if key not in self.exclude_list:
                    msg +=key+":"+str(value)+'\n'
            self.socket.send( msg )
            items = msg.split("\n")
            msg_type = items.pop(0)
            for index, item in enumerate(items):
                if 'base' in item:
                    items.pop(index)
            items = dict([i.split(':') for i in items[:-1] ])
            if 'realtime gaze on mark1' in items:
                pupil_x1,pupil_y1  = map(float,items['realtime gaze on mark1'][1:-1].split(','))
                try:
                    client.send( OSCMessage("/pupil/mark1",(pupil_x1,pupil_y1)))
                except:
                    pass
            else:
                try:
                    client.send( OSCMessage("/pupil/mark1",(100.0,100.0)))
                except:
                    pass
            if 'realtime gaze on mark2' in items:
                pupil_x1,pupil_y1  = map(float,items['realtime gaze on mark2'][1:-1].split(','))
                try:
                    client.send( OSCMessage("/pupil/mark2",(pupil_x1,pupil_y1)))
                except:
                    pass
            else:
                try:
                    client.send( OSCMessage("/pupil/mark2",(100.0,100.0)))
                except:
                    pass
            if 'realtime gaze on circles' in items:
                dist  = float(items.get('realtime gaze on circles'))
                try:
                    client.send( OSCMessage("/pupil/circles",dist))
                except:
                    pass
            else:
                try:
                    client.send( OSCMessage("/pupil/circles",100.0))
                except:
                    pass

        # for e in events:
        #     msg = 'Event'+'\n'
        #     for key,value in e.iteritems():
        #         if key not in self.exclude_list:
        #             msg +=key+":"+str(value).replace('\n','')+'\n'
        #     self.socket.send( msg )

    def close(self):
        self.alive = False


    def get_init_dict(self):
        return {'address':self.address}


    def cleanup(self):
        """gets called when the plugin get terminated.
        This happens either voluntarily or forced.
        """
        self.deinit_gui()
        self.context.destroy()

