#ifndef SingleEyeFitter_h__
#define SingleEyeFitter_h__

#include <mutex>
#include <Eigen/Core>
// #include <opencv2/core/core.hpp>
#include <singleeyefitter/Circle.h>
#include <singleeyefitter/Ellipse.h>
#include <singleeyefitter/Sphere.h>
// #include <singleeyefitter/Geometry.h>


namespace singleeyefitter {

    class EyeModelFitter {
    public:
        // structures
        struct PupilParams {
            double theta, psi, radius;
            PupilParams();
            PupilParams(double theta, double psi, double radius);
        };
        struct Pupil {
            // Observation observation;
            // modifying to have it similar to sphere_fitter/__init__.py
            singleeyefitter::Ellipse2D<double>ellipse;
            singleeyefitter::Circle3D<double> circle;
            PupilParams params;
            bool init_valid = false;
            Circle[] projected_circles; // is this the right syntax?
            Eigen::ParametrizedLine<Scalar, 2>& line; //self.line

            Pupil(Ellipse ellipse,double focal_length);
        };

        // Typedefs
        typedef Eigen::Matrix<double, 2, 1> Vector2;
        typedef Eigen::Matrix<double, 3, 1> Vector3;
        typedef Eigen::ParametrizedLine<double, 2> Line;
        typedef Eigen::ParametrizedLine<double, 3> Line3;
        typedef singleeyefitter::Circle3D<double> Circle;
        typedef singleeyefitter::Ellipse2D<double> Ellipse;
        typedef singleeyefitter::Sphere<double> Sphere;
        typedef size_t Index;

        // Public fields
        Eigen::Matrix<double, 3, 4> intrinsics;
        double focal_length;
        static const Vector3 camera_center(0,0,0);
        Sphere eye;
        Ellipse projected_eye;
        std::vector<Pupil> pupils;
        double scale = 1;
        std::mutex model_mutex;
        int model_version = 0; // not exactly used

        Eigen::ParametrizedLine<Scalar, 2>& pupil_gazelines_proj;
        Eigen::Matrix<double, 2,2> twoDim_A;        
        Vector2 twoDim_B;

        double count;

        // Constructors
        EyeModelFitter();
        // EyeModelFitter(Eigen::Matrix<double, 3, 4> intrinsics);
        EyeModelFitter(double focal_length);

        // functions
        Index add_observation(Ellipse pupil);
        Index add_pupil_labs_observation(Ellipse pupil);
        void reset();
        Circle circleFromParams(PupilParams& params);
        void initialise_model();
        Circle& initialise_single_observation(Pupil& pupil);
        void update_model();
        void unproject_observations(double eye_z = 20, bool use_ransac = false);

    };

}

#endif // SingleEyeFitter_h__
