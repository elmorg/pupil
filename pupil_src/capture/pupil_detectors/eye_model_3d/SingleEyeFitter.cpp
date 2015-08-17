// SingleEyeFitter.cpp : Defines the entry point for the console application.

#include <boost/math/special_functions/sign.hpp>

#include <Eigen/StdVector>

#include <ceres/ceres.h>
#include <ceres/problem.h>
#include <ceres/autodiff_cost_function.h>
#include <ceres/solver.h>
#include <ceres/jet.h>

#include <singleeyefitter/singleeyefitter.h>

#include <singleeyefitter/utils.h>
#include <singleeyefitter/cvx.h>
#include <singleeyefitter/Conic.h>
#include <singleeyefitter/Ellipse.h>
#include <singleeyefitter/Circle.h>
#include <singleeyefitter/Conicoid.h>
#include <singleeyefitter/Sphere.h>
#include <singleeyefitter/solve.h>
#include <singleeyefitter/intersect.h>
#include <singleeyefitter/projection.h>
#include <singleeyefitter/fun.h>
#include <singleeyefitter/math.h>

#include "distance.h"

#include <spii/spii.h>
#include <spii/term.h>
#include <spii/function.h>
#include <spii/solver.h>

namespace ceres {
    using singleeyefitter::math::sq;

    template<typename T, int N>
    inline Jet<T,N> sq(Jet<T,N> val) {
        val.v *= 2*val.a;
        val.a *= val.a;
        return val;
    }
}

namespace singleeyefitter {

struct scalar_tag{};
struct ceres_jet_tag{};

template<typename T, typename Enabled=void>
struct ad_traits;

template<typename T>
struct ad_traits<T, typename std::enable_if< std::is_arithmetic<T>::value >::type >
{
    typedef scalar_tag ad_tag;
    typedef T scalar;
    static inline scalar value(const T& x) { return x; }
};

template<typename T, int N>
struct ad_traits<::ceres::Jet<T,N>>
{
    typedef ceres_jet_tag ad_tag;
    typedef T scalar;
    static inline scalar get(const ::ceres::Jet<T,N>& x) { return x.a; }
};

template<typename T>
struct ad_traits<T, typename std::enable_if< !std::is_same<T, typename std::decay<T>::type>::value >::type >
    : public ad_traits<typename std::decay<T>::type>
{
};

template<typename T>
inline T norm(T x, T y, scalar_tag) {
    using std::sqrt;
    using math::sq;

    return sqrt(sq(x) + sq(y));
}
template<typename T, int N>
inline ::ceres::Jet<T,N> norm(const ::ceres::Jet<T,N>& x, const ::ceres::Jet<T,N>& y, ceres_jet_tag) {
    T anorm = norm<T>(x.a, y.a, scalar_tag());

    ::ceres::Jet<T,N> g;
    g.a = anorm;
    g.v = (x.a/anorm)*x.v + (y.a/anorm)*y.v;

    return g;
}
template<typename T>
inline typename std::decay<T>::type norm(T&& x, T&& y) {
    return norm(std::forward<T>(x), std::forward<T>(y), typename ad_traits<T>::ad_tag());
}

template<typename Scalar>
cv::Rect bounding_box(const Ellipse2D<Scalar>& ellipse) {
    using std::sin;
    using std::cos;
    using std::sqrt;
    using std::floor;
    using std::ceil;

    Scalar ux = ellipse.major_radius * cos(ellipse.angle);
    Scalar uy = ellipse.major_radius * sin(ellipse.angle);
    Scalar vx = ellipse.minor_radius * cos(ellipse.angle + PI/2);
    Scalar vy = ellipse.minor_radius * sin(ellipse.angle + PI/2);

    Scalar bbox_halfwidth = sqrt(ux*ux + vx*vx);
    Scalar bbox_halfheight = sqrt(uy*uy + vy*vy);

    return cv::Rect(floor(ellipse.center[0] - bbox_halfwidth), floor(ellipse.center[1] - bbox_halfheight),
                    2*ceil(bbox_halfwidth) + 1, 2*ceil(bbox_halfheight) + 1);
}

// Calculates the x crossings of a conic at a given y value. Returns the number of crossings (0, 1 or 2)
template<typename Scalar>
int getXCrossing(const Conic<Scalar>& conic, Scalar y, Scalar& x1, Scalar& x2) {
    using std::sqrt;

    Scalar a = conic.A;
    Scalar b = conic.B*y + conic.D;
    Scalar c = conic.C*y*y + conic.E*y + conic.F;

    Scalar det = b*b - 4*a*c;
    if (det == 0) {
        x1 = -b/(2*a);
        return 1;
    } else if (det < 0) {
        return 0;
    } else {
        Scalar sqrtdet = sqrt(det);
        x1 = (-b - sqrtdet)/(2*a);
        x2 = (-b + sqrtdet)/(2*a);
        return 2;
    }
}

template<template<class, int> class Jet, class T, int N>
typename std::enable_if<std::is_same<typename ad_traits<Jet<T,N>>::ad_tag, ceres_jet_tag>::value, Ellipse2D<T>>::type
toConst(const Ellipse2D<Jet<T,N>>& ellipse) {
    return Ellipse2D<T>(
        ellipse.center[0].a,
        ellipse.center[1].a,
        ellipse.major_radius.a,
        ellipse.minor_radius.a,
        ellipse.angle.a);
}

template<class T>
Ellipse2D<T> scaledMajorRadius(const Ellipse2D<T>& ellipse, const T& target_radius) {
    return Ellipse2D<T>(
        ellipse.center[0],
        ellipse.center[1],
        target_radius,
        target_radius * ellipse.minor_radius/ellipse.major_radius,
        ellipse.angle);
};

template<typename T>
Eigen::Matrix<T,3,1> sph2cart(T r, T theta, T psi) {
    using std::sin;
    using std::cos;

    return r * Eigen::Matrix<T,3,1>(sin(theta)*cos(psi), cos(theta), sin(theta)*sin(psi));
}

template<typename T>
T angleDiffGoodness(T theta1, T psi1, T theta2, T psi2, typename ad_traits<T>::scalar sigma) {
    using std::sin;
    using std::cos;
    using std::acos;
    using std::asin;
    using std::atan2;
    using std::sqrt;

    if (theta1 == theta2 && psi1 == psi2) {
        return T(1);
    }

    // Haversine distance
    auto dist = T(2)*asin(sqrt(sq(sin((theta1-theta2)/T(2))) + cos(theta1)*cos(theta2)*sq(sin((psi1-psi2)/T(2)))));
    return exp(-sq(dist)/sq(sigma));
}

template<typename T>
Circle3D<T> circleOnSphere(const Sphere<T>& sphere, T theta, T psi, T circle_radius) {
    typedef Eigen::Matrix<T,3,1> Vector3;

    Vector3 radial = sph2cart<T>(T(1), theta, psi);
    return Circle3D<T>(sphere.center + sphere.radius * radial,
        radial,
        circle_radius);
}

const EyeModelFitter::Vector3 EyeModelFitter::camera_center = EyeModelFitter::Vector3::Zero();

// EyeModelFitter::Pupil::Pupil(Ellipse ellipse) : ellipse(ellipse), params(0, 0, 0){}
EyeModelFitter::Pupil::Pupil(Ellipse ellipse, Eigen::Matrix<double,3,3> intrinsics) : ellipse(ellipse){
    params = PupilParams(0,0,0);

    // performance enhancements originally in unproject_observations()
    projected_circles = unproject_intrinsics(ellipse, 1.0, intrinsics); //getting pupil circles, force radius to be double.
    Vector3 c = projected_circles.first.center; // get projected circles, gaze vectors
    Vector3 v = projected_circles.first.normal;
    Vector2 c_proj = project_point(c,intrinsics);
    Vector2 v_proj = project_point(v+c, intrinsics) - c_proj;
    v_proj.normalize();
    line = Line(c_proj,v_proj);
}
EyeModelFitter::Pupil::Pupil(){}

EyeModelFitter::PupilParams::PupilParams(double theta, double psi, double radius) : theta(theta), psi(psi), radius(radius){}
EyeModelFitter::PupilParams::PupilParams() : theta(0), psi(0), radius(0){}

}

singleeyefitter::EyeModelFitter::EyeModelFitter() {}
singleeyefitter::EyeModelFitter::EyeModelFitter(double focal_length, double x_disp, double y_disp){
    intrinsics(0,0) = focal_length; // setting the intrinsics value
    intrinsics(1,1) = -focal_length;
    intrinsics(2,0) = x_disp; //should be 0,2 technically, but 
    intrinsics(2,1) = y_disp; //should be 1,2
} 
singleeyefitter::EyeModelFitter::EyeModelFitter(double focal_length) {
    intrinsics(0,0) = focal_length;
    intrinsics(1,1) = -focal_length;
}

void singleeyefitter::EyeModelFitter::add_observation( // factoring in can't feed in ellipse from python
    double center_x, double center_y, double major_radius, double minor_radius, double angle){
    std::lock_guard<std::mutex> lock_model(model_mutex);
    Vector2 center(center_x,center_y);
    Ellipse pupil(center, major_radius, minor_radius, angle);
    pupils.emplace_back(pupil, intrinsics); // this should call EyeModelFitter::Pupil::Pupil(Ellipse ellipse)
    // optimization
    pupil_gazelines_projection.push_back(pupils[pupils.size()-1].line);
    auto vi = pupils[pupils.size()-1].line.direction();
    auto pi = pupils[pupils.size()-1].line.origin();
    Eigen::Matrix2d Ivivi = Eigen::Matrix2d::Identity() - vi * vi.transpose();
    twoDim_A += Ivivi;
    twoDim_B += (Ivivi * pi);
    // pupils[pupils.size()-1].line.clear(); // don't need it anymore, though code doesn't currently work
}

singleeyefitter::EyeModelFitter::Index singleeyefitter::EyeModelFitter::add_pupil_labs_observation(Ellipse pupil){
    std::lock_guard<std::mutex> lock_model(model_mutex);
    pupils.emplace_back(pupil, intrinsics); // this should call EyeModelFitter::Pupil::Pupil(Ellipse ellipse)
    return pupils.size() - 1;
}

void EyeModelFitter::reset(){
    std::lock_guard<std::mutex> lock_model(model_mutex);
    pupils.clear();
    eye = Sphere::Null;
    model_version++;
}

void  singleeyefitter::EyeModelFitter::print_eye(){
    std::cout << eye << std::endl;
}

void singleeyefitter::EyeModelFitter::print_ellipse(Index id){
    std::cout << pupils[id].ellipse << std::endl;
}

singleeyefitter::EyeModelFitter::Circle singleeyefitter::EyeModelFitter::circleFromParams(const Sphere& eye, const PupilParams& params){
    if (params.radius == 0)
        return Circle::Null;

    Vector3 radial = sph2cart<double>(double(1), params.theta, params.psi);
    return Circle(eye.center + eye.radius * radial,
        radial,
        params.radius);
}

singleeyefitter::EyeModelFitter::Circle singleeyefitter::EyeModelFitter::circleFromParams(const PupilParams& params) const{
    return circleFromParams(eye, params);
}

const singleeyefitter::EyeModelFitter::Circle& singleeyefitter::EyeModelFitter::initialise_single_observation(Pupil& pupil)
{
    // Ignore the pupil circle normal, and intersect the pupil circle
    // center projection line with the eyeball sphere
    try {
        auto pupil_center_sphere_intersect = intersect(Line3(camera_center, pupil.circle.center.normalized()),
            eye);
        auto new_pupil_center = pupil_center_sphere_intersect.first;

        // Now that we have 3D positions for the pupil (rather than just a
        // projection line), recalculate the pupil radius at that position.
        auto pupil_radius_at_1 = pupil.circle.radius / pupil.circle.center.z();
        auto new_pupil_radius = pupil_radius_at_1 * new_pupil_center.z();

        // Parametrise this new pupil position using spherical coordinates
        Vector3 center_to_pupil = new_pupil_center - eye.center;
        double r = center_to_pupil.norm();
        pupil.params.theta = acos(center_to_pupil[1] / r);
        pupil.params.psi = atan2(center_to_pupil[2], center_to_pupil[0]);
        pupil.params.radius = new_pupil_radius;

        // Update pupil circle to match parameters
        // pupil.circle = circleFromParams(pupil.params); 
        // skip this line since in initialize model, will recreate pupil.circle
    }
    catch (no_intersection_exception&) {
        pupil.circle = Circle::Null;
        pupil.params.theta = 0;
        pupil.params.psi = 0;
        pupil.params.radius = 0;
    }
    return pupil.circle;
}

const singleeyefitter::EyeModelFitter::Circle& singleeyefitter::EyeModelFitter::initialise_single_observation(Index id)
{
    initialise_single_observation(pupils[id]);
    return pupils[id].circle;
}

const singleeyefitter::EyeModelFitter::Circle& singleeyefitter::EyeModelFitter::unproject_single_observation(Pupil& pupil, double pupil_radius /*= 1*/) const
{
    if (eye == Sphere::Null) {
        throw std::runtime_error("Need to get eye center estimate first (by unprojecting multiple observations)");
    }

    // Single pupil version of "unproject_observations"
    auto unprojection_pair = unproject_intrinsics(pupil.ellipse, pupil_radius, intrinsics);

    const Eigen::Vector3d c = unprojection_pair.first.center;
    const Eigen::Vector3d v = unprojection_pair.first.normal;
    Vector2 c_proj = project_point(c, intrinsics);
    Vector2 v_proj = project_point(v + c, intrinsics) - c_proj;
    v_proj.normalize();
    Vector2 eye_center_proj = project_point(eye.center, intrinsics);

    if ((c_proj - eye_center_proj).dot(v_proj) >= 0) {
        pupil.circle = std::move(unprojection_pair.first);
    }
    else {
        pupil.circle = std::move(unprojection_pair.second);
    }

    return pupil.circle;
}

const singleeyefitter::EyeModelFitter::Circle& singleeyefitter::EyeModelFitter::unproject_single_observation(Index id, double pupil_radius /*= 1*/)
{
    return unproject_single_observation(pupils[id], pupil_radius);
}

void singleeyefitter::EyeModelFitter::initialise_model()
{
    std::lock_guard<std::mutex> lock_model(model_mutex);
    if (eye == Sphere::Null) {
        return;
    }

    // Find pupil positions on eyeball to get radius
    // For each image, calculate the 'most likely' position of the pupil
    // circle given the eyeball sphere estimate and gaze vector. Re-estimate
    // the gaze vector to be consistent with this position.

    // First estimate of pupil center, used only to get an estimate of eye radius

    double eye_radius_acc = 0;
    int eye_radius_count = 0;

    for (const auto& pupil : pupils) {
        if (!pupil.circle) {
            continue;
        }
        if (!pupil.init_valid) {
            continue;
        }

        // Intersect the gaze from the eye center with the pupil circle
        // center projection line (with perfect estimates of gaze, eye
        // center and pupil circle center, these should intersect,
        // otherwise find the nearest point to both lines)

        Vector3 pupil_center = nearest_intersect(Line3(eye.center, pupil.circle.normal),
            Line3(camera_center, pupil.circle.center.normalized()));

        auto distance = (pupil_center - eye.center).norm();

        eye_radius_acc += distance;
        ++eye_radius_count;
    }

    // Set the eye radius as the mean distance from pupil centers to eye center
    eye.radius = eye_radius_acc / eye_radius_count;

    // Second estimate of pupil radius, used to get position of pupil on eye

    for (auto& pupil : pupils) {
        initialise_single_observation(pupil);
    }

    // Scale eye to anthropomorphic average radius of 12mm
    auto scale = 12.0 / eye.radius;
    eye.radius = 12.0;
    eye.center *= scale;
    for (auto& pupil : pupils) {
        pupil.params.radius *= scale;
        pupil.circle = circleFromParams(pupil.params);
    }
    model_version++;
}

void singleeyefitter::EyeModelFitter::unproject_observations(double pupil_radius, double eye_z)
{
    using math::sq;
    std::lock_guard<std::mutex> lock_model(model_mutex);

    if (pupils.size() < 2) {
        throw std::runtime_error("Need at least two observations");
    }
    // Get eyeball center
    // Find a least-squares 'intersection' (point nearest to all lines) of
    // the projected 2D gaze vectors. Then, unproject that circle onto a
    // point a fixed distance away.
    Vector2 eye_center_proj = twoDim_A.partialPivLu().solve(twoDim_B);
    // Vector2 eye_center_proj = nearest_intersect(pupil_gazelines_projection);
    eye.center = unproject_point(eye_center_proj,eye_z, intrinsics);
    eye.radius = 1;
    projected_eye = project_sphere(eye,intrinsics); //projection.h function

    // Disambiguate pupil circles using projected eyeball center
    for (size_t i = 0; i < pupils.size(); ++i) {
        const auto& c_proj = pupil_gazelines_projection[i].origin(); // pupil_gazelines_proj is line
        const auto& v_proj = pupil_gazelines_projection[i].direction();

        // Check if v_proj going away from est eye center. If it is, then
        // the first circle was correct. Otherwise, take the second one.
        // The two normals will point in opposite directions, so only need
        // to check one.
        if ((c_proj - eye_center_proj).dot(v_proj) >= 0) {
            pupils[i].circle = pupils[i].projected_circles.first;
        }
        else {
            pupils[i].circle = pupils[i].projected_circles.second;
        }
        pupils[i].init_valid = true;
    }

    model_version++;
}
