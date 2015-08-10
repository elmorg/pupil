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
// #include <singleeyefitter/cvx.h>
// #include <singleeyefitter/Conic.h>
// #include <singleeyefitter/Ellipse.h>
// #include <singleeyefitter/Circle.h>
// #include <singleeyefitter/Conicoid.h>
// #include <singleeyefitter/Sphere.h>
#include <singleeyefitter/geometry.h>
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
inline T smootherstep(T edge0, T edge1, T x, scalar_tag)
{
    if (x >= edge1)
        return T(1);
    else if (x <= edge0)
        return T(0);
    else {
        x = (x - edge0)/(edge1 - edge0);
        return x*x*x*(x*(x*T(6) - T(15)) + T(10));
    }
}

template<typename T, int N>
inline ::ceres::Jet<T,N> smootherstep(T edge0, T edge1, const ::ceres::Jet<T,N>& f, ceres_jet_tag)
{
    if (f.a >= edge1)
        return ::ceres::Jet<T,N>(1);
    else if (f.a <= edge0)
        return ::ceres::Jet<T,N>(0);
    else {
        T x = (f.a - edge0)/(edge1 - edge0);

        // f is referenced by this function, so create new value for return.
        ::ceres::Jet<T,N> g;
        g.a = x*x*x*(x*(x*T(6) - T(15)) + T(10));
        g.v = f.v * (x*x*(x*(x*T(30) - T(60)) + T(30))/(edge1 - edge0));
        return g;
    }
}

template<typename T, int N>
inline ::ceres::Jet<T,N> smootherstep(T edge0, T edge1, ::ceres::Jet<T,N>&& f, ceres_jet_tag)
{
    if (f.a >= edge1)
        return ::ceres::Jet<T,N>(1);
    else if (f.a <= edge0)
        return ::ceres::Jet<T,N>(0);
    else {
        T x = (f.a - edge0)/(edge1 - edge0);

        // f is moved into this function, so reuse it.
        f.a = x*x*x*(x*(x*T(6) - T(15)) + T(10));
        f.v *= (x*x*(x*(x*T(30) - T(60)) + T(30))/(edge1 - edge0));
        return f;
    }
}

template<typename T>
inline auto smootherstep(typename ad_traits<T>::scalar edge0, typename ad_traits<T>::scalar edge1, T&& val)
    -> decltype(smootherstep(edge0, edge1, std::forward<T>(val), typename ad_traits<T>::ad_tag()))
{
    return smootherstep(edge0, edge1, std::forward<T>(val), typename ad_traits<T>::ad_tag());
}

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

template<typename T>
inline auto Heaviside(T&& val, typename ad_traits<T>::scalar epsilon) -> decltype(smootherstep(-epsilon, epsilon, std::forward<T>(val))) {
    return smootherstep(-epsilon, epsilon, std::forward<T>(val));
}

// Calculates:
//     r * (1 - ||A(p - t)||)
//
//          ||A(p - t)||   maps the ellipse to a unit circle
//      1 - ||A(p - t)||   measures signed distance from unit circle edge
// r * (1 - ||A(p - t)||)  scales this to major radius of ellipse, for (roughly) pixel distance
//
// Actually use (r - ||rAp - rAt||) and precalculate r, rA and rAt.

template<typename T>
class EllipseDistCalculator { // not sure if I need this class but I'll keep it here for now.
public:
    typedef typename ad_traits<T>::scalar Const;

    EllipseDistCalculator(const Ellipse2D<T>& ellipse) : r(ellipse.major_radius)
    {
        using std::sin;
        using std::cos;
        rA << r*cos(ellipse.angle)/ellipse.major_radius, r*sin(ellipse.angle)/ellipse.major_radius,
             -r*sin(ellipse.angle)/ellipse.minor_radius, r*cos(ellipse.angle)/ellipse.minor_radius;
        rAt = rA*ellipse.centre;
    }
    template<typename U>
    T operator()(U&& x, U&& y) {
        return calculate(std::forward<U>(x), std::forward<U>(y), typename ad_traits<T>::ad_tag(), typename ad_traits<U>::ad_tag());
    }

    template<typename U>
    T calculate(U&& x, U&& y, scalar_tag, scalar_tag) {
        T rAxt((rA(0,0) * x + rA(0,1) * y) - rAt[0]);
        T rAyt((rA(1,0) * x + rA(1,1) * y) - rAt[1]);

        T xy_dist = norm(rAxt, rAyt);

        return (r - xy_dist);
    }

    // Expanded versions for Jet calculations so that Eigen can do some of its expression magic
    template<typename U>
    T calculate(U&& x, U&& y, scalar_tag, ceres_jet_tag) {
        T rAxt(rA(0,0) * x.a + rA(0,1) * y.a - rAt[0],
            rA(0,0) * x.v + rA(0,1) * y.v);
        T rAyt(rA(1,0) * x.a + rA(1,1) * y.a - rAt[1],
            rA(1,0) * x.v + rA(1,1) * y.v);

        T xy_dist = norm(rAxt, rAyt);

        return (r - xy_dist);
    }
    template<typename U>
    T calculate(U&& x, U&& y, ceres_jet_tag, scalar_tag) {
        T rAxt(rA(0,0).a * x + rA(0,1).a * y - rAt[0].a,
            rA(0,0).v * x + rA(0,1).v * y - rAt[0].v);
        T rAyt(rA(1,0).a * x + rA(1,1).a * y - rAt[1].a,
            rA(1,0).v * x + rA(1,1).v * y - rAt[1].v);

        T xy_dist = norm(rAxt, rAyt);

        return (r - xy_dist);
    }
    template<typename U>
    T calculate(U&& x, U&& y, ceres_jet_tag, ceres_jet_tag) {
        T rAxt(rA(0,0).a * x.a + rA(0,1).a * y.a - rAt[0].a,
            rA(0,0).v * x.a + rA(0,0).a * x.v + rA(0,1).v * y.a + rA(0,1).a * y.v - rAt[0].v);
        T rAyt(rA(1,0).a * x.a + rA(1,1).a * y.a - rAt[1].a,
            rA(1,0).v * x.a + rA(1,0).a * x.v + rA(1,1).v * y.a + rA(1,1).a * y.v - rAt[1].v);

        T xy_dist = norm(rAxt, rAyt);

        return (r - xy_dist);
    }
private:
    Eigen::Matrix<T, 2, 2> rA;
    Eigen::Matrix<T, 2, 1> rAt;
    T r;
};

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
        ellipse.centre[0].a,
        ellipse.centre[1].a,
        ellipse.major_radius.a,
        ellipse.minor_radius.a,
        ellipse.angle.a);
}

template<class T>
Ellipse2D<T> scaledMajorRadius(const Ellipse2D<T>& ellipse, const T& target_radius) {
    return Ellipse2D<T>(
        ellipse.centre[0],
        ellipse.centre[1],
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
    return Circle3D<T>(sphere.centre + sphere.radius * radial,
        radial,
        circle_radius);
}

template<typename Scalar>
struct EllipsePointDistanceFunction {
    EllipsePointDistanceFunction(const Ellipse2D<Scalar>& el, Scalar x, Scalar y) : el(el), x(x), y(y) {}

    template <typename T>
    bool operator()(const T* const t, T* e) const
    {
        using std::sin;
        using std::cos;

        auto&& pt = pointAlongEllipse(el, t[0]);
        e[0] = norm(x - pt.x(), y - pt.y());

        return true;
    }

    const Ellipse2D<Scalar>& el;
    Scalar x, y;
};

template<bool has_eye_var=true>

const EyeModelFitter::Vector3 EyeModelFitter::camera_centre = EyeModelFitter::Vector3::Zero();


EyeModelFitter::Pupil::Pupil(Ellipse e,Eigen::Matrix<double, 3, 4> i){
    ellipse = e;
    intrinsics = i;
    
    init_valid = false;

    // get pupil circles
    // Do a per-image unprojection of the pupil ellipse into the two fixed
    // size circles that would project onto it. The size of the circles
    // doesn't matter here, only their center and normal does.
    
    projected_circles = self.ellipse.unproject(radius = 1, intrinsics= intrinsics);
    // get projected circles and gaze vectors
    // Project the circle centers and gaze vectors down back onto the image plane.
    // We're only using them as line parameterizations, so it doesn't matter which of the two centers/gaze
    // vectors we use, as the two gazes are parallel and the centers are co-linear
    
    // why do I default use the 0th one, not the 1st one???
    // here maybe write some function that determines which line is better
    c = np.reshape(self.projected_circles[0].center, (3,1)); //it is a 3D circle
    v = np.reshape(self.projected_circles[0].normal, (3,1));
    c_proj = geometry.project_point(c,intrinsics);
    v_proj = geometry.project_point(v + c, intrinsics) - c_proj;
    v_proj = v_proj/np.linalg.norm(v_proj); //normalizing
    self.line = geometry.Line2D(c_proj, v_proj); //append this to self.pupil_gazeline_proj

    // currently in python, need to write as cpp
}
// EyeModelFitter::Pupil::Pupil(){}

// Pupil Params Init
EyeModelFitter::PupilParams::PupilParams(double theta, double psi, double radius) : theta(theta), psi(psi), radius(radius){}
EyeModelFitter::PupilParams::PupilParams() : theta(0), psi(0), radius(0){}

singleeyefitter::EyeModelFitter::EyeModelFitter() : region_band_width(5), region_step_epsilon(0.5), region_scale(1)
{

}
singleeyefitter::EyeModelFitter::EyeModelFitter(double focal_length, double region_band_width, double region_step_epsilon) : focal_length(focal_length), region_band_width(region_band_width), region_step_epsilon(region_step_epsilon), region_scale(1)
{

}

singleeyefitter::EyeModelFitter::Index singleeyefitter::EyeModelFitter::add_observation(Ellipse pupil, int n_pseudo_inliers /*= 0*/)
{
    for (int i = 0; i < n_pseudo_inliers; ++i) {
        auto p = pointAlongEllipse(pupil, i * 2 * M_PI / n_pseudo_inliers);
        pupil_inliers.emplace_back(static_cast<float>(p[0]), static_cast<float>(p[1]));
    }
    return add_observation(std::move(image), std::move(pupil), std::move(pupil_inliers));
}

singleeyefitter::EyeModelFitter::Index singleeyefitter::EyeModelFitter::add_observation(Ellipse pupil)
{
    assert(image.channels() == 1 && image.depth() == CV_8U);

    std::lock_guard<std::mutex> lock_model(model_mutex);

    pupils.emplace_back(
        Observation(std::move(image), std::move(pupil), std::move(pupil_inliers))
        );
    return pupils.size() - 1;
}

void EyeModelFitter::reset()
{
    std::lock_guard<std::mutex> lock_model(model_mutex);
    pupils.clear();
    eye = Sphere::Null;
    model_version++;
}

singleeyefitter::EyeModelFitter::Circle singleeyefitter::EyeModelFitter::circleFromParams(const Sphere& eye, const PupilParams& params)
{
    if (params.radius == 0)
        return Circle::Null;

    Vector3 radial = sph2cart<double>(double(1), params.theta, params.psi);
    return Circle(eye.centre + eye.radius * radial,
        radial,
        params.radius);
}

singleeyefitter::EyeModelFitter::Circle singleeyefitter::EyeModelFitter::circleFromParams(const PupilParams& params) const
{
    return circleFromParams(eye, params);
}

const singleeyefitter::EyeModelFitter::Circle& singleeyefitter::EyeModelFitter::initialise_single_observation(Pupil& pupil)
{
    // Ignore the pupil circle normal, and intersect the pupil circle
    // centre projection line with the eyeball sphere
    try {
        auto pupil_centre_sphere_intersect = intersect(Line3(camera_centre, pupil.circle.centre.normalized()),
            eye);
        auto new_pupil_centre = pupil_centre_sphere_intersect.first;

        // Now that we have 3D positions for the pupil (rather than just a
        // projection line), recalculate the pupil radius at that position.
        auto pupil_radius_at_1 = pupil.circle.radius / pupil.circle.centre.z();
        auto new_pupil_radius = pupil_radius_at_1 * new_pupil_centre.z();

        // Parametrise this new pupil position using spherical coordinates
        Vector3 centre_to_pupil = new_pupil_centre - eye.centre;
        double r = centre_to_pupil.norm();
        pupil.params.theta = acos(centre_to_pupil[1] / r);
        pupil.params.psi = atan2(centre_to_pupil[2], centre_to_pupil[0]);
        pupil.params.radius = new_pupil_radius;

        // Update pupil circle to match parameters
        pupil.circle = circleFromParams(pupil.params);
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
        throw std::runtime_error("Need to get eye centre estimate first (by unprojecting multiple observations)");
    }

    // Single pupil version of "unproject_observations"

    auto unprojection_pair = unproject(pupil.observation.ellipse, pupil_radius, focal_length);

    const Vector3& c = unprojection_pair.first.centre;
    const Vector3& v = unprojection_pair.first.normal;

    Vector2 c_proj = project(c, focal_length);
    Vector2 v_proj = project(v + c, focal_length) - c_proj;

    v_proj.normalize();

    Vector2 eye_centre_proj = project(eye.centre, focal_length);

    if ((c_proj - eye_centre_proj).dot(v_proj) >= 0) {
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
    //
    // For each image, calculate the 'most likely' position of the pupil
    // circle given the eyeball sphere estimate and gaze vector. Re-estimate
    // the gaze vector to be consistent with this position.

    // First estimate of pupil centre, used only to get an estimate of eye radius

    double eye_radius_acc = 0;
    int eye_radius_count = 0;

    for (const auto& pupil : pupils) {
        if (!pupil.circle) {
            continue;
        }
        if (!pupil.init_valid) {
            continue;
        }

        // Intersect the gaze from the eye centre with the pupil circle
        // centre projection line (with perfect estimates of gaze, eye
        // centre and pupil circle centre, these should intersect,
        // otherwise find the nearest point to both lines)

        Vector3 pupil_centre = nearest_intersect(Line3(eye.centre, pupil.circle.normal),
            Line3(camera_centre, pupil.circle.centre.normalized()));

        auto distance = (pupil_centre - eye.centre).norm();

        eye_radius_acc += distance;
        ++eye_radius_count;
    }

    // Set the eye radius as the mean distance from pupil centres to eye centre
    eye.radius = eye_radius_acc / eye_radius_count;

    // Second estimate of pupil radius, used to get position of pupil on eye

    for (auto& pupil : pupils) {
        initialise_single_observation(pupil);
    }

    // Scale eye to anthropomorphic average radius of 12mm
    auto scale = 12.0 / eye.radius;
    eye.radius = 12.0;
    eye.centre *= scale;
    for (auto& pupil : pupils) {
        pupil.params.radius *= scale;
        pupil.circle = circleFromParams(pupil.params);
    }

    model_version++;
}

void singleeyefitter::EyeModelFitter::unproject_observations(double pupil_radius /*= 1*/, double eye_z /*= 20*/, bool use_ransac /*= true*/)
{
    using math::sq;

    std::lock_guard<std::mutex> lock_model(model_mutex);

    if (pupils.size() < 2) {
        throw std::runtime_error("Need at least two observations");
    }

    std::vector<std::pair<Circle, Circle>> pupil_unprojection_pairs;
    std::vector<Line> pupil_gazelines_proj;

    for (const auto& pupil : pupils) {
        // Get pupil circles (up to depth)
        //
        // Do a per-image unprojection of the pupil ellipse into the two fixed
        // size circles that would project onto it. The size of the circles
        // doesn't matter here, only their centre and normal does.
        auto unprojection_pair = unproject(pupil.observation.ellipse,
            pupil_radius, focal_length);

        // Get projected circles and gaze vectors
        //
        // Project the circle centres and gaze vectors down back onto the image
        // plane. We're only using them as line parametrisations, so it doesn't
        // matter which of the two centres/gaze vectors we use, as the
        // two gazes are parallel and the centres are co-linear.

        const auto& c = unprojection_pair.first.centre;
        const auto& v = unprojection_pair.first.normal;

        Vector2 c_proj = project(c, focal_length);
        Vector2 v_proj = project(v + c, focal_length) - c_proj;

        v_proj.normalize();

        pupil_unprojection_pairs.push_back(std::move(unprojection_pair));
        pupil_gazelines_proj.emplace_back(c_proj, v_proj);
    }


    // Get eyeball centre
    //
    // Find a least-squares 'intersection' (point nearest to all lines) of
    // the projected 2D gaze vectors. Then, unproject that circle onto a
    // point a fixed distance away.
    //
    // For robustness, use RANSAC to eliminate stray gaze lines
    //
    // (This has to be done here because it's used by the pupil circle
    // disambiguation)

    Vector2 eye_centre_proj;
    bool valid_eye;

    // fun.h and utils.h are only used here.
    // if (use_ransac) {
    //     auto indices = fun::range_<std::vector<size_t>>(pupil_gazelines_proj.size());

    //     const int n = 2;
    //     double w = 0.3;
    //     double p = 0.9999;
    //     int k = ceil(log(1 - p) / log(1 - pow(w, n)));

    //     double epsilon = 10;
    //     auto huber_error = [&](const Vector2& point, const Line& line) {
    //         double dist = euclidean_distance(point, line);
    //         if (sq(dist) < sq(epsilon))
    //             return sq(dist) / 2;
    //         else
    //             return epsilon*(abs(dist) - epsilon / 2);
    //     };
    //     auto m_error = [&](const Vector2& point, const Line& line) {
    //         double dist = euclidean_distance(point, line);
    //         if (sq(dist) < sq(epsilon))
    //             return sq(dist);
    //         else
    //             return sq(epsilon);
    //     };
    //     auto error = m_error;

    //     auto best_inlier_indices = decltype(indices)();
    //     Vector2 best_eye_centre_proj;// = nearest_intersect(pupil_gazelines_proj);
    //     double best_line_distance_error = std::numeric_limits<double>::infinity();// = fun::sum(LAMBDA(const Line& line)(error(best_eye_centre_proj,line)), pupil_gazelines_proj);

    //     for (int i = 0; i < k; ++i) {
    //         auto index_sample = singleeyefitter::randomSubset(indices, n);
    //         auto sample = fun::map([&](size_t i){ return pupil_gazelines_proj[i]; }, index_sample);

    //         auto sample_centre_proj = nearest_intersect(sample);

    //         auto index_inliers = fun::filter(
    //             [&](size_t i){ return euclidean_distance(sample_centre_proj, pupil_gazelines_proj[i]) < epsilon; },
    //             indices);
    //         auto inliers = fun::map([&](size_t i){ return pupil_gazelines_proj[i]; }, index_inliers);

    //         if (inliers.size() <= w*pupil_gazelines_proj.size()) {
    //             continue;
    //         }

    //         auto inlier_centre_proj = nearest_intersect(inliers);

    //         double line_distance_error = fun::sum(
    //             [&](size_t i){ return error(inlier_centre_proj, pupil_gazelines_proj[i]); },
    //             indices);

    //         if (line_distance_error < best_line_distance_error) {
    //             best_eye_centre_proj = inlier_centre_proj;
    //             best_line_distance_error = line_distance_error;
    //             best_inlier_indices = std::move(index_inliers);
    //         }
    //     }

    //     std::cout << "Inliers: " << best_inlier_indices.size()
    //         << " (" << (100.0*best_inlier_indices.size() / pupil_gazelines_proj.size()) << "%)"
    //         << " = " << best_line_distance_error
    //         << std::endl;

    //     for (auto& pupil : pupils) {
    //         pupil.init_valid = false;
    //     }
    //     for (auto& i : best_inlier_indices) {
    //         pupils[i].init_valid = true;
    //     }

    //     if (best_inlier_indices.size() > 0) {
    //         eye_centre_proj = best_eye_centre_proj;
    //         valid_eye = true;
    //     }
    //     else {
    //         valid_eye = false;
    //     }
    // }

    else {
        for (auto& pupil : pupils) {
            pupil.init_valid = true;
        }
        eye_centre_proj = nearest_intersect(pupil_gazelines_proj);
        valid_eye = true;
    }

    if (valid_eye) {
        eye.centre << eye_centre_proj * eye_z / focal_length,
            eye_z;
        eye.radius = 1;

        // Disambiguate pupil circles using projected eyeball centre
        //
        // Assume that the gaze vector points away from the eye centre, and
        // so projected gaze points away from projected eye centre. Pick the
        // solution which satisfies this assumption
        for (size_t i = 0; i < pupils.size(); ++i) {
            const auto& pupil_pair = pupil_unprojection_pairs[i];
            const auto& line = pupil_gazelines_proj[i];

            const auto& c_proj = line.origin();
            const auto& v_proj = line.direction();

            // Check if v_proj going away from est eye centre. If it is, then
            // the first circle was correct. Otherwise, take the second one.
            // The two normals will point in opposite directions, so only need
            // to check one.
            if ((c_proj - eye_centre_proj).dot(v_proj) >= 0) {
                pupils[i].circle = std::move(pupil_pair.first);
            }
            else {
                pupils[i].circle = std::move(pupil_pair.second);
            }
        }
    }
    else {
        // No inliers, so no eye
        eye = Sphere::Null;

        // Arbitrarily pick first circle
        for (size_t i = 0; i < pupils.size(); ++i) {
            const auto& pupil_pair = pupil_unprojection_pairs[i];
            pupils[i].circle = std::move(pupil_pair.first);
        }
    }

    model_version++;
}