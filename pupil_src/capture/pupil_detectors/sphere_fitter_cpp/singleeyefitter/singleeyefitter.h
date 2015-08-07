#ifndef SingleEyeFitter_h__
#define SingleEyeFitter_h__

#include <mutex>
#include <Eigen/Core>
// #include <opencv2/core/core.hpp>
#include <singleeyefitter/Circle.h>
#include <singleeyefitter/Ellipse.h>
#include <singleeyefitter/Sphere.h>

namespace singleeyefitter {


    // template<typename Scalar>
    // inline Eigen::Matrix<Scalar, 2, 1> toEigen(const cv::Point2f& point) {
    //     return Eigen::Matrix<Scalar, 2, 1>(static_cast<Scalar>(point.x),
    //         static_cast<Scalar>(point.y));
    // }
    // template<typename Scalar>
    // inline cv::Point2f toPoint2f(const Eigen::Matrix<Scalar, 2, 1>& point) {
    //     return cv::Point2f(static_cast<float>(point[0]),
    //         static_cast<float>(point[1]));
    // }
    // template<typename Scalar>
    // inline cv::Point toPoint(const Eigen::Matrix<Scalar, 2, 1>& point) {
    //     return cv::Point(static_cast<int>(point[0]),
    //         static_cast<int>(point[1]));
    // }
    // template<typename Scalar>
    // inline cv::RotatedRect toRotatedRect(const Ellipse2D<Scalar>& ellipse) {
    //     return cv::RotatedRect(toPoint2f(ellipse.centre),
    //         cv::Size2f(static_cast<float>(2 * ellipse.major_radius),
    //         static_cast<float>(2 * ellipse.minor_radius)),
    //         static_cast<float>(ellipse.angle * 180 / PI));
    // }
    // template<typename Scalar>
    // inline Ellipse2D<Scalar> toEllipse(const cv::RotatedRect& rect) {
    //     return Ellipse2D<Scalar>(toEigen<Scalar>(rect.center),
    //         static_cast<Scalar>(rect.size.width / 2),
    //         static_cast<Scalar>(rect.size.height / 2),
    //         static_cast<Scalar>(rect.angle*PI / 180));
    // }

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
            Circle circle;
            PupilParams params;
            bool init_valid;
            Circle[] projected_circles; // is this the right syntax?
            Eigen::ParametrizedLine<Scalar, 2>& line; //self.line

            Pupil();
            // Pupil(Observation observation);
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

        static const Vector3 camera_centre;

        // Public fields
        double focal_length;
        // double region_band_width;
        // double region_step_epsilon;
        // double region_scale;


        // Constructors
        EyeModelFitter();
        EyeModelFitter(double focal_length, double region_band_width, double region_step_epsilon);

        // functions
        Index add_observation(Ellipse pupil);
        Index add_pupil_labs_observation(Ellipse pupil);
        void reset();
        Circle circleFromParams(const PupilParams& params) const;
        void initialise_model();
        const Circle& initialise_single_observation(Pupil& pupil);
        void unproject_observations(double pupil_radius = 1, double eye_z = 20, bool use_ransac = false);
        void update_model();

        Sphere eye;
        std::vector<Pupil> pupils;
        std::mutex model_mutex;
        // Model version gets incremented on initialisation/reset, so that long-running background-thread refines don't overwrite the model
        int model_version = 0;

        const Circle& unproject_single_observation(Pupil& pupil, double pupil_radius = 1) const;

        static Circle circleFromParams(const Sphere& eye, const PupilParams& params);

        // DON'T NEED OBSERVATION STRUCTURE. see sphere_fitter/__init__.py
        // struct Observation {
        //     cv::Mat image;
        //     Ellipse ellipse;
        //     std::vector<cv::Point2f> inliers;

        //     Observation();
        //     Observation(cv::Mat image, Ellipse ellipse, std::vector<cv::Point2f> inliers);
        // };

        // Unused functions and stuff
        // void refine_with_region_contrast(const CallbackFunction& callback = CallbackFunction());
        // void refine_with_inliers(const CallbackFunction& callback = CallbackFunction());
        // const Circle& refine_single_with_contrast(Pupil& pupil);
        // double single_contrast_metric(const Pupil& pupil) const;
        // void print_single_contrast_metric(const Pupil& pupil) const;
        // typedef std::function<void(const Sphere&, const std::vector<Circle>&)> CallbackFunction;
        // const Circle& unproject_single_observation(Index id, double pupil_radius = 1);
        // const Circle& refine_single_with_contrast(Index id);
        // double single_contrast_metric(Index id) const;
        // void print_single_contrast_metric(Index id) const;

    };

}

#endif // SingleEyeFitter_h__
