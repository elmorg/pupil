
cdef extern from "singleeyefitter/singleeyefitter.h" namespace "singleeyefitter":
    cdef cppclass EyeModelFitter:
        # EyeModelFitter(double focal_length, double region_band_width, double region_step_epsilon)
        EyeModelFitter(double focal_length, double x_disp, double y_disp)
        # EyeModelFitter(double focal_length)
        void reset()
        void initialise_model()

        float model_version
        float intrinsicsval # can delete later, for testing purpose
        float focal_length

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

    property model_version:
        def __get__(self):
            return self.thisptr.model_version

    property focal_length:
        def __get__(self):
            return self.thisptr.focal_length

    property intrinsicsval: #can delete later, for testing purpose
        def __get__(self):
            return self.thisptr.intrinsicsval



