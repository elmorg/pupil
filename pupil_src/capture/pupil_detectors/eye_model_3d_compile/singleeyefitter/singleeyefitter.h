#ifndef SingleEyeFitter_h__
#define SingleEyeFitter_h__

#include <mutex>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <singleeyefitter/cvx.h>
#include <singleeyefitter/Circle.h>
#include <singleeyefitter/Ellipse.h>
#include <singleeyefitter/Sphere.h>

namespace singleeyefitter {

    // not sure what these 5 functions are used for but leaving it here for now doesn't hurt.
    template<typename Scalar>
    inline Eigen::Matrix<Scalar, 2, 1> toEigen(const cv::Point2f& point) {
        return Eigen::Matrix<Scalar, 2, 1>(static_cast<Scalar>(point.x),
            static_cast<Scalar>(point.y));
    }
    template<typename Scalar>
    inline cv::Point2f toPoint2f(const Eigen::Matrix<Scalar, 2, 1>& point) {
        return cv::Point2f(static_cast<float>(point[0]),
            static_cast<float>(point[1]));
    }
    template<typename Scalar>
    inline cv::Point toPoint(const Eigen::Matrix<Scalar, 2, 1>& point) {
        return cv::Point(static_cast<int>(point[0]),
            static_cast<int>(point[1]));
    }
    template<typename Scalar>
    inline cv::RotatedRect toRotatedRect(const Ellipse2D<Scalar>& ellipse) {
        return cv::RotatedRect(toPoint2f(ellipse.center),
            cv::Size2f(static_cast<float>(2 * ellipse.major_radius),
            static_cast<float>(2 * ellipse.minor_radius)),
            static_cast<float>(ellipse.angle * 180 / PI));
    }
    template<typename Scalar>
    inline Ellipse2D<Scalar> toEllipse(const cv::RotatedRect& rect) {
        return Ellipse2D<Scalar>(toEigen<Scalar>(rect.center),
            static_cast<Scalar>(rect.size.width / 2),
            static_cast<Scalar>(rect.size.height / 2),
            static_cast<Scalar>(rect.angle*PI / 180));
    }

    class EyeModelFitter {
    public:
        // Typedefs
        typedef Eigen::Matrix<double, 2, 1> Vector2;
        typedef Eigen::Matrix<double, 3, 1> Vector3;
        typedef Eigen::ParametrizedLine<double, 2> Line;
        typedef Eigen::ParametrizedLine<double, 3> Line3;
        typedef singleeyefitter::Circle3D<double> Circle;
        typedef singleeyefitter::Ellipse2D<double> Ellipse;
        typedef singleeyefitter::Sphere<double> Sphere;
        typedef size_t Index;

        // structures
        struct PupilParams {
            double theta, psi, radius;
            PupilParams();
            PupilParams(double theta, double psi, double radius);
        };
        struct Pupil {
            // Observation observation;
            // modifying to have it similar to sphere_fitter/__init__.py
            singleeyefitter::Ellipse2D<double> ellipse;
            singleeyefitter::Circle3D<double> circle;
            PupilParams params;
            bool init_valid = false;
            std::pair<Circle3D<double>, Circle3D<double>> projected_circles;
            Eigen::ParametrizedLine<double, 2>& line; //self.line

            Pupil(Ellipse pupil_ellipse,Eigen::Matrix<double, 3, 4> intrinsics);
        };

        // Variables I use
        Eigen::Matrix<double, 3, 4> intrinsics;
        // intrinsics << 1, 0, 0, 0,
        //             0, -1, 0, 0,
        //             0, 0, 0, 0; // default argument
        static const Vector3 camera_center;
        Sphere eye;
        Ellipse projected_eye;
        std::vector<Pupil> pupils;
        double scale = 1;
        std::mutex model_mutex;
        // Model version gets incremented on initialisation/reset, so that long-running background-thread refines don't overwrite the model
        int model_version = 0;

        // Nonessential Variables I use
        Eigen::ParametrizedLine<double, 2>& pupil_gazelines_proj;
        Eigen::Matrix<double, 2,2> twoDim_A;        
        Vector2 twoDim_B;
        double count;

        // Variables I don't use, but swirski uses
        double focal_length;
        double region_band_width;
        double region_step_epsilon;
        double region_scale;

        // Constructors
        EyeModelFitter();
        EyeModelFitter(double focal_length, double region_band_width, double region_step_epsilon);
        EyeModelFitter(Eigen::Matrix<double, 3, 4> intrinsics);
        EyeModelFitter(double focal_length);
        void reset();

        // Functions I use       
        Index add_observation(Ellipse pupil_ellipse);
        Index add_pupil_labs_observation(Ellipse pupil_ellipse);
        void unproject_observations(double pupil_radius = 1, double eye_z = 20, bool use_ransac = true);
        void initialise_model();
        Circle circleFromParams(const PupilParams& params) const;
        const Circle& initialise_single_observation(Index id);
        const Circle& initialise_single_observation(Pupil& pupil);

        // functions I don't use
        typedef std::function<void(const Sphere&, const std::vector<Circle>&)> CallbackFunction;
        // void refine_with_region_contrast(const CallbackFunction& callback = CallbackFunction()); // uses observation object
        // void refine_with_inliers(const CallbackFunction& callback = CallbackFunction()); // uses observation object
        const Circle& unproject_single_observation(Index id, double pupil_radius = 1);
        // const Circle& refine_single_with_contrast(Index id); // uses observation object
        // double single_contrast_metric(Index id) const; // uses observation object
        // void print_single_contrast_metric(Index id) const; // uses observation object
        const Circle& unproject_single_observation(Pupil& pupil, double pupil_radius = 1) const;
        // const Circle& refine_single_with_contrast(Pupil& pupil); // uses observation object
        // double single_contrast_metric(const Pupil& pupil) const; // uses observation object
        // void print_single_contrast_metric(const Pupil& pupil) const; // uses observation object
        static Circle circleFromParams(const Sphere& eye, const PupilParams& params);
    };

}

#endif // SingleEyeFitter_h__
