
cdef extern from "singleeyefitter/singleeyefitter.h" namespace "singleeyefitter":
    cdef cppclass EyeModelFitter:
        EyeModelFitter(double focal_length, double x_disp, double y_disp)
        # EyeModelFitter(double focal_length)
        void reset()
        void initialise_model()
        void unproject_observations(double pupil_radius, double eye_z )
        void add_observation(double center_x,double center_y, double major_radius, double minor_radius, double angle)
        void print_eye()
        void print_ellipse(size_t id)

        float model_version

cdef class PyEyeModelFitter:
    cdef EyeModelFitter *thisptr
    cdef public int counter
    # def __cinit__(self, focal_length):
    #     self.thisptr = new EyeModelFitter(focal_length)
    def __cinit__(self, focal_length, x_disp, y_disp):
        self.thisptr = new EyeModelFitter(focal_length, x_disp, y_disp)

    def __init__(self,focal_length, x_disp, y_disp):
        self.counter = 0

    def __dealloc__(self):
        del self.thisptr

    def reset(self):
        self.thisptr.reset()

    def initialise_model(self):
        self.thisptr.initialise_model()

    def unproject_observations(self, pupil_radius = 1, eye_z = 20):
        self.thisptr.unproject_observations(pupil_radius,eye_z)

    def update_model(self, pupil_radius = 1, eye_z = 20):
        # this function runs unproject_observations and initialise_model, and prints
        # the eye model once every 30 iterations.
        if self.counter >= 30:
            self.counter = 0
            self.thisptr.print_eye()
        self.counter += 1
        self.thisptr.unproject_observations(pupil_radius,eye_z)
        self.thisptr.initialise_model()

    def add_observation(self,center,major_radius,minor_radius,angle):
        #standard way of adding an observation
        self.thisptr.add_observation(center[0], center[1],major_radius,minor_radius,angle)

    def add_pupil_labs_observation(self,e_dict):
        # a special method for taking in arguments from eye.py
        a,b = e_dict['axes']
        a,b = e_dict['axes']
        if a > b:
            major_radius = a/2
            minor_radius = b/2
            angle = e_dict['angle']*3.1415926535/180
        else:
            major_radius = b/2
            minor_radius = a/2
            angle = (e_dict['angle']+90)*3.1415926535/180 # not importing np just for pi constant
        self.thisptr.add_observation(e_dict['center'][0],e_dict['center'][1],major_radius,minor_radius,angle)

    def print_eye(self):
        self.thisptr.print_eye()
        # return temp[0],temp[1],temp[2], temp[3]

    def print_ellipse(self,index):
        self.thisptr.print_ellipse(index)

    def get_ellipse(self,index):
        pass

    property model_version:
        def __get__(self):
            return self.thisptr.model_version



