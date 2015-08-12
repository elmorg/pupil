if __name__ == '__main__':
    import subprocess as sp
    sp.call("python setup.py build_ext --inplace",shell=True)
print "BUILD COMPLETE ______________________"

import eye_model_3d

############# TESTING MODEL ##################
model = eye_model_3d.PyEyeModelFitter(focal_length=100, x_disp = 50, y_disp = 80)
print model
print model.intrinsicsval

model.add_observation(20,20,5,3,1)
model.add_observation(40,20,5,2,1)
model.unproject_observations()
model.initialise_model()
print model.get_eye_model()
