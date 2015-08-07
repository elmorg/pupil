#ifndef _CIRCLE_H_
#define _CIRCLE_H_

#include <Eigen/Core>

namespace singleeyefitter {

    template<typename T>
    class Circle3D {
    public:
        typedef T Scalar;
        typedef Eigen::Matrix<Scalar, 3, 1> Vector;

        Vector centre, normal;
        Scalar radius;

        Circle3D() : centre(0, 0, 0), normal(0, 0, 0), radius(0)
        {
        }
        Circle3D(Vector centre, Vector normal, Scalar radius)
            : centre(std::move(centre)), normal(std::move(normal)), radius(std::move(radius))
        {
        }

        static const Circle3D Null;

    private:
        // Safe bool stuff
        typedef void (Circle3D::*safe_bool_type)() const;
        void this_type_does_not_support_comparisons() const {}
    public:
        operator safe_bool_type() const {
            return *this != Null ? &Circle3D::this_type_does_not_support_comparisons : 0;
        }
    };

    template<typename Scalar>
    const Circle3D<Scalar> Circle3D<Scalar>::Null = Circle3D<Scalar>();

    template<typename Scalar>
    bool operator== (const Circle3D<Scalar>& s1, const Circle3D<Scalar>& s2) {
        return s1.centre == s2.centre
            && s1.normal == s2.normal
            && s1.radius == s2.radius;
    }
    template<typename Scalar>
    bool operator!= (const Circle3D<Scalar>& s1, const Circle3D<Scalar>& s2) {
        return s1.centre != s2.centre
            || s1.normal != s2.normal
            || s1.radius != s2.radius;
    }

    template<typename T>
    std::ostream& operator<< (std::ostream& os, const Circle3D<T>& circle) {
        return os << "Circle { centre: (" << circle.centre[0] << "," << circle.centre[1] << "," << circle.centre[2] << "), "
            "normal: (" << circle.normal[0] << "," << circle.normal[1] << "," << circle.normal[2] << "), "
            "radius: " << circle.radius << " }";
    }

}
#endif//_CIRCLE_H_
