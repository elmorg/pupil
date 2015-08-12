
cdef extern from "singleeyefitter/singleeyefitter.h" namespace "singleeyefitter":
    cdef cppclass EyeModelFitter:
        # EyeModelFitter(double focal_length, double region_band_width, double region_step_epsilon)
        EyeModelFitter(double focal_length, double x_disp, double y_disp)
        # EyeModelFitter(double focal_length)
        void reset()
        void initialise_model()
        void unproject_observations(double pupil_radius, double eye_z )
        void add_observation(double center_x,double center_y, double major_radius, double minor_radius, double angle)
        double get_eye()

        float model_version
        float intrinsicsval # can delete later, for testing purpose

cdef class PyEyeModelFitter:
    cdef EyeModelFitter *thisptr
    # def __cinit__(self, focal_length):
    #     self.thisptr = new EyeModelFitter(focal_length)
    def __cinit__(self, focal_length, x_disp, y_disp):
        self.thisptr = new EyeModelFitter(focal_length, x_disp, y_disp)

    def __init__(self,focal_length, x_disp, y_disp):
        pass
    def __dealloc__(self):
        del self.thisptr

    def reset(self):
        self.thisptr.reset()

    def initialise_model(self):
        self.thisptr.initialise_model()

    def unproject_observations(self, pupil_radius = 1, eye_z = 20):
        self.thisptr.unproject_observations(pupil_radius,eye_z)

    def add_observation(self,center_x, center_y,major_radius,minor_radius,angle):
        self.thisptr.add_observation(center_x, center_y,major_radius,minor_radius,angle)

    def get_eye_model(self):
        return self.thisptr.get_eye()

    property model_version:
        def __get__(self):
            return self.thisptr.model_version

    property intrinsicsval: #can delete later, for testing purpose
        def __get__(self):
            return self.thisptr.intrinsicsval




