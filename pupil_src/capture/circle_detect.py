'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

"""ELM - Plugin for using the calibration marker as a general purpose marker. Gaze distance
from the marker is added to the events """

import os
import cv2
import numpy as np
from methods import normalize,denormalize
from pyglui.cygl.utils import draw_points_norm,draw_polyline,RGBA
from OpenGL.GL import GL_POLYGON
from circle_detector import get_candidate_ellipses

from plugin import Plugin


class Circle_detect(Plugin):
    """Detector looks for a white ring on a black background.
        Using at least 9 positions/points within the FOV
        Ref detector will direct one to good positions with audio cues
        Calibration only collects data at the good positions

        Steps:
            Adaptive threshold to obtain robust edge-based image of marker
            Find contours and filter into 2 level list using RETR_CCOMP
            Fit ellipses
    """
    def __init__(self,g_pool):
        super(Circle_detect, self).__init__(g_pool)
        self.active = True
        self.detected = False
        self.order = .2
        self.g_pool = g_pool
        self.pos = None
        self.smooth_pos = 0.,0.
        self.smooth_vel = 0.
        self.sample_site = (-2,-2)
        self.candidate_ellipses = []
        self.show_edges = 0
        self.aperture = 7
        self.dist_threshold = 10
        self.area_threshold = 30
        self.world_size = None
        self.lastTime = 0.
        self.menu = None
        self.button = None



    def toggle(self,_=None):
        if self.active:
            self.stop()
        else:
            self.start()

    def start(self):
        self.active = True


    def stop(self):
        self.smooth_pos = 0,0
        self.active = False



    def update(self,frame,events):
        """
        gets called once every frame.
        reference positon need to be published to shared_pos
        if no reference was found, publish 0,0
        """
        if self.active:

            gray_img  = frame.gray

            if self.world_size is None:
                self.world_size = frame.width,frame.height
        
            self.candidate_ellipses = get_candidate_ellipses(gray_img,
                                                         area_threshold=self.area_threshold,
                                                         dist_threshold=self.dist_threshold,
                                                         min_ring_count=4,
                                                         visual_debug=self.show_edges)

            if len(self.candidate_ellipses) > 0:
                self.detected = True
                self.lastTime = frame.timestamp
                marker_pos = self.candidate_ellipses[0][0]
                self.pos = normalize(marker_pos,(frame.width,frame.height),flip_y=True)
                smoother = 0.3
                smooth_pos = np.array(self.smooth_pos)
                pos = np.array(self.pos)
                new_smooth_pos = smooth_pos + smoother*(pos-smooth_pos)
                smooth_pos = new_smooth_pos
                self.smooth_pos = list(smooth_pos)
            
                for p in events.get('gaze_positions',[]):
                        #gp_on_s = tuple(s.img_to_ref_surface(np.array(p['norm_gaze'])))
                        dist = np.linalg.norm(np.array(p['norm_pos'])-self.smooth_pos)
                        p['realtime gaze on circles'] = dist



#            else:
#                self.detected = False
#                timeDiff = frame.timestamp - self.lastTime
#                if timeDiff < 0.5:
#                    for p in events.get('gaze_positions',[]):
#                        #gp_on_s = tuple(s.img_to_ref_surface(np.array(p['norm_gaze'])))
#                        dist = np.linalg.norm(np.array(p['norm_pos'])-self.smooth_pos)
#                        p['realtime gaze on circles'] = dist
#                self.pos = None #indicate that no reference is detected





        else:
            pass



    def gl_display(self):
        """
        use gl calls to render
        at least:
            the published position of the reference
        better:
            show the detected postion even if not published
        """

        if self.active and self.detected:
            draw_points_norm([self.smooth_pos],size=15,color=RGBA(1.,1.,0.,.5))
            for e in self.candidate_ellipses:
                pts = cv2.ellipse2Poly( (int(e[0][0]),int(e[0][1])),
                                       (int(e[1][0]/2),int(e[1][1]/2)),
                                       int(e[-1]),0,360,15)
                draw_polyline(pts,color=RGBA(0.,1.,0,1.))

        else:
            pass

    def cleanup(self):
        """gets called when the plugin get terminated.
        This happends either volunatily or forced.
        if you have an atb bar or glfw window destroy it here.
        """
        if self.active:
            self.stop()

#    def get_init_dict(self):
#        return {}
