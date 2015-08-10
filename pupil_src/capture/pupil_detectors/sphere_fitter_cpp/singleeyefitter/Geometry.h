// Andrew Xia, August 10th 2015.
// geometry.h should parallel geometry.py in the python version of singleeyefitter.

// Modules to include: 
// Circle3D (from circle.h), Conic (from conic.h), Conicoid (from conicoid.h), 
// Ellipse (from ellipse.h), Sphere (from sphere.h)

// Not included:
// PupilParams (staying in singleeyefitter.h) Line (since is eigen module)

// see ../../eye_model_3d folder to recover the individual geometry header files

#ifndef GEOMETRY_H_
#define GEOMETRY_H_

#include <Eigen/Core>
#include <singleeyefitter/math.h> // for conic
#include <boost/math/constants/constants.hpp> // for accessing pi in ellipse

/////////////////////////////////////////////////////////////////////
// CIRCLE

namespace singleeyefitter {

    template<typename T>
    class Circle3D {
    public:
        typedef T Scalar;
        typedef Eigen::Matrix<Scalar, 3, 1> Vector;

        Vector center, normal;
        Scalar radius;

        Circle3D() : center(0, 0, 0), normal(0, 0, 0), radius(0)
        {
        }
        Circle3D(Vector center, Vector normal, Scalar radius)
            : center(std::move(center)), normal(std::move(normal)), radius(std::move(radius))
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
        return s1.center == s2.center
            && s1.normal == s2.normal
            && s1.radius == s2.radius;
    }
    template<typename Scalar>
    bool operator!= (const Circle3D<Scalar>& s1, const Circle3D<Scalar>& s2) {
        return s1.center != s2.center
            || s1.normal != s2.normal
            || s1.radius != s2.radius;
    }

    template<typename T>
    std::ostream& operator<< (std::ostream& os, const Circle3D<T>& circle) {
        return os << "Circle { center: (" << circle.center[0] << "," << circle.center[1] << "," << circle.center[2] << "), "
            "normal: (" << circle.normal[0] << "," << circle.normal[1] << "," << circle.normal[2] << "), "
            "radius: " << circle.radius << " }";
    }

/////////////////////////////////////////////////////////////////////
// CONIC

    template<typename T>
    class Ellipse2D;

    template<typename T>
    class Conic {
    public:
        typedef T Scalar;

        Scalar A, B, C, D, E, F;
        Conic(Scalar A, Scalar B, Scalar C, Scalar D, Scalar E, Scalar F)
            : A(A), B(B), C(C), D(D), E(E), F(F) {}

        template<typename U>
        explicit Conic(const Ellipse2D<U>& ellipse) {
            using std::sin;
            using std::cos;
            using singleeyefitter::math::sq;

            auto ax = cos(ellipse.angle);
            auto ay = sin(ellipse.angle);

            auto a2 = sq(ellipse.major_radius);
            auto b2 = sq(ellipse.minor_radius);

            A = ax*ax / a2 + ay*ay / b2;
            B = 2 * ax*ay / a2 - 2 * ax*ay / b2;
            C = ay*ay / a2 + ax*ax / b2;
            D = (-2 * ax*ay*ellipse.center[1] - 2 * ax*ax*ellipse.center[0]) / a2
                + (2 * ax*ay*ellipse.center[1] - 2 * ay*ay*ellipse.center[0]) / b2;
            E = (-2 * ax*ay*ellipse.center[0] - 2 * ay*ay*ellipse.center[1]) / a2
                + (2 * ax*ay*ellipse.center[0] - 2 * ax*ax*ellipse.center[1]) / b2;
            F = (2 * ax*ay*ellipse.center[0] * ellipse.center[1] + ax*ax*ellipse.center[0] * ellipse.center[0] + ay*ay*ellipse.center[1] * ellipse.center[1]) / a2
                + (-2 * ax*ay*ellipse.center[0] * ellipse.center[1] + ay*ay*ellipse.center[0] * ellipse.center[0] + ax*ax*ellipse.center[1] * ellipse.center[1]) / b2
                - 1;
        }

        Scalar operator()(Scalar x, Scalar y) const {
            return A*x*x + B*x*y + C*y*y + D*x + E*y + F;
        }
    };

    template<typename T>
    std::ostream& operator<< (std::ostream& os, const Conic<T>& conic) {
        return os << "Conic { " << conic.A << "x^2 + " << conic.B << "xy + " << conic.C << "y^2 + " << conic.D << "x + " << conic.E << "y + " << conic.F << " = 0 } ";
    }

/////////////////////////////////////////////////////////////////////
// CONICOID

    template<typename T>
    class Conic;

    // Conicoid (quartic surface) of the form:
    // Ax^2 + By^2 + Cz^2 + 2Fyz + 2Gzx + 2Hxy + 2Ux + 2Vy + 2Wz + D = 0
    template<typename T>
    class Conicoid {
    public:
        typedef T Scalar;

        Scalar A, B, C, F, G, H, U, V, W, D;

        Conicoid(Scalar A, Scalar B, Scalar C, Scalar F, Scalar G, Scalar H, Scalar D)
            : A(A), B(B), C(C), F(F), G(G), H(H), U(U), V(V), W(W), D(D)
        {
        }

        template<typename ConicScalar, typename Derived>
        explicit Conicoid(const Conic<ConicScalar>& conic, const Eigen::MatrixBase<Derived>& vertex) {
            static_assert(Derived::IsVectorAtCompileTime && Derived::SizeAtCompileTime == 3, "Cone vertex requires 3 element vector as vector type");
            using math::sq;

            // Finds conicoid with given conic base and vertex
            // Assumes conic is on the plane z = 0

            auto alpha = vertex[0];
            auto beta = vertex[1];
            auto gamma = vertex[2];

            A = sq(gamma) * conic.A;
            B = sq(gamma) * conic.C;
            C = conic.A*sq(alpha) + conic.B*alpha*beta + conic.C*sq(beta) + conic.D*alpha + conic.E*beta + conic.F;
            F = -gamma * (conic.C * beta + conic.B / 2 * alpha + conic.E / 2);
            G = -gamma * (conic.B / 2 * beta + conic.A * alpha + conic.D / 2);
            H = sq(gamma) * conic.B / 2;
            U = sq(gamma) * conic.D / 2;
            V = sq(gamma) * conic.E / 2;
            W = -gamma * (conic.E / 2 * beta + conic.D / 2 * alpha + conic.F);
            D = sq(gamma) * conic.F;
        }

        Scalar operator()(Scalar x, Scalar y, Scalar z) const {
            return A*sq(x) + B*sq(y) + C*sq(z) + 2 * F*y*z + 2 * G*x*z + 2 * H*x*y + 2 * U*x + 2 * V*y + 2 * W*z + D;
        }

        Conic<Scalar> intersectZ(Scalar z = Scalar(0)) const {
            // Finds conic at given z intersection

            // Ax^2 + By^2 + Cz^2 + 2Fyz + 2Gzx + 2Hxy + 2Ux + 2Vy + 2Wz + D = 0
            // becomes
            // Ax^2 + Bxy + Cy^2 + Fx + Ey + D = 0

            return Conic<Scalar>(A,
                2 * H,
                B,
                2 * G*z + 2 * U,
                2 * F*z + 2 * V,
                C*sq(z) + 2 * W*z + D);
        }
    };

    template<typename T>
    std::ostream& operator<< (std::ostream& os, const Conicoid<T>& conicoid) {
        return os << "Conicoid { " << conicoid.A << "x^2 + " << conicoid.B << "y^2 + " << conicoid.C << "z^2 + "
            "2*" << 2 * conicoid.F << "yz + 2*" << 2 * conicoid.G << "zx + 2*" << 2 * conicoid.H << "xy + "
            "2*" << 2 * conicoid.U << "x + 2*" << 2 * conicoid.V << "y + 2*" << 2 * conicoid.W << "z + " << conicoid.D << " = 0 }";
    }

/////////////////////////////////////////////////////////////////////
// ELLIPSE

    template<typename T>
    class Conic;

    template<typename T>
    class Ellipse2D {
    public:
        typedef T Scalar;
        typedef Eigen::Matrix<Scalar, 2, 1> Vector;
        Vector center;
        Scalar major_radius;
        Scalar minor_radius;
        Scalar angle;

        Ellipse2D()
            : center(0, 0), major_radius(0), minor_radius(0), angle(0)
        {
        }
        template<typename Derived>
        Ellipse2D(const Eigen::EigenBase<Derived>& center, Scalar major_radius, Scalar minor_radius, Scalar angle)
            : center(center), major_radius(major_radius), minor_radius(minor_radius), angle(angle)
        {
        }
        Ellipse2D(Scalar x, Scalar y, Scalar major_radius, Scalar minor_radius, Scalar angle)
            : center(x, y), major_radius(major_radius), minor_radius(minor_radius), angle(angle)
        {
        }
        template<typename U>
        explicit Ellipse2D(const Conic<U>& conic) {
            using std::atan2;
            using std::sin;
            using std::cos;
            using std::sqrt;
            using std::abs;

            angle = 0.5*atan2(conic.B, conic.A - conic.C);
            auto cost = cos(angle);
            auto sint = sin(angle);
            auto sin_squared = sint * sint;
            auto cos_squared = cost * cost;

            auto Ao = conic.F;
            auto Au = conic.D * cost + conic.E * sint;
            auto Av = -conic.D * sint + conic.E * cost;
            auto Auu = conic.A * cos_squared + conic.C * sin_squared + conic.B * sint * cost;
            auto Avv = conic.A * sin_squared + conic.C * cos_squared - conic.B * sint * cost;

            // ROTATED = [Ao Au Av Auu Avv]

            auto tucenter = -Au / (2.0*Auu);
            auto tvcenter = -Av / (2.0*Avv);
            auto wcenter = Ao - Auu*tucenter*tucenter - Avv*tvcenter*tvcenter;

            center[0] = tucenter * cost - tvcenter * sint;
            center[1] = tucenter * sint + tvcenter * cost;

            major_radius = sqrt(abs(-wcenter / Auu));
            minor_radius = sqrt(abs(-wcenter / Avv));

            if (major_radius < minor_radius) {
                std::swap(major_radius, minor_radius);
                angle = angle + boost::math::double_constants::pi / 2;
            }
            if (angle > boost::math::double_constants::pi)
                angle = angle - boost::math::double_constants::pi;
        }

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF((sizeof(Vector) % 16) == 0)

        Vector major_axis() const {
            using std::sin;
            using std::cos;
            return Vector(major_radius*sin(angle), major_radius*cos(angle));
        }
        Vector minor_axis() const {
            using std::sin;
            using std::cos;
            return Vector(-minor_radius*cos(angle), minor_radius*sin(angle));
        }

        static const Ellipse2D Null;

    private:
        // Safe bool stuff
        typedef void (Ellipse2D::*safe_bool_type)() const;
        void this_type_does_not_support_comparisons() const {}
    public:
        operator safe_bool_type() const {
            return *this != Null ? &Ellipse2D::this_type_does_not_support_comparisons : 0;
        }
    };

    template<typename Scalar>
    const Ellipse2D<Scalar> Ellipse2D<Scalar>::Null = Ellipse2D<Scalar>();

    template<typename T, typename U>
    bool operator==(const Ellipse2D<T>& el1, const Ellipse2D<U>& el2) {
        return el1.center[0] == el2.center[0] &&
            el1.center[1] == el2.center[1] &&
            el1.major_radius == el2.major_radius &&
            el1.minor_radius == el2.minor_radius &&
            el1.angle == el2.angle;
    }
    template<typename T, typename U>
    bool operator!=(const Ellipse2D<T>& el1, const Ellipse2D<U>& el2) {
        return !(el1 == el2);
    }

    template<typename T>
    std::ostream& operator<< (std::ostream& os, const Ellipse2D<T>& ellipse) {
        return os << "Ellipse { center: (" << ellipse.center[0] << "," << ellipse.center[1] << "), a: " <<
            ellipse.major_radius << ", b: " << ellipse.minor_radius << ", theta: " << (ellipse.angle / boost::math::double_constants::pi) << "pi }";
    }

    template<typename T, typename U>
    Ellipse2D<T> scaled(const Ellipse2D<T>& ellipse, U scale) {
        return Ellipse2D<T>(
            ellipse.center[0].a,
            ellipse.center[1].a,
            ellipse.major_radius.a,
            ellipse.minor_radius.a,
            ellipse.angle.a);
    }

    template<class Scalar, class Scalar2>
    inline Eigen::Matrix<typename std::common_type<Scalar, Scalar2>::type, 2, 1> pointAlongEllipse(const Ellipse2D<Scalar>& el, Scalar2 t)
    {
        using std::sin;
        using std::cos;
        auto xt = el.center.x() + el.major_radius*cos(el.angle)*cos(t) - el.minor_radius*sin(el.angle)*sin(t);
        auto yt = el.center.y() + el.major_radius*sin(el.angle)*cos(t) + el.minor_radius*cos(el.angle)*sin(t);
        return Eigen::Matrix<typename std::common_type<Scalar, Scalar2>::type, 2, 1>(xt, yt);
    }

/////////////////////////////////////////////////////////////////////
// SPHERE

    template<typename T>
    class Sphere {
    public:
        typedef T Scalar;
        typedef Eigen::Matrix<Scalar, 3, 1> Vector;

        Vector center;
        Scalar radius;

        Sphere() : center(0, 0, 0), radius(0){}
        Sphere(Vector center, Scalar radius)
            : center(std::move(center)), radius(std::move(radius)){}

        static const Sphere Null;

    private:
        // Safe bool stuff
        typedef void (Sphere::*safe_bool_type)() const;
        void this_type_does_not_support_comparisons() const {}
    public:
        operator safe_bool_type() const {
            return *this != Null ? &Sphere::this_type_does_not_support_comparisons : 0;
        }
    };

    template<typename Scalar>
    const Sphere<Scalar> Sphere<Scalar>::Null = Sphere<Scalar>();

    template<typename Scalar>
    bool operator== (const Sphere<Scalar>& s1, const Sphere<Scalar>& s2) {
        return s1.center == s2.center
            && s1.radius == s2.radius;
    }
    template<typename Scalar>
    bool operator!= (const Sphere<Scalar>& s1, const Sphere<Scalar>& s2) {
        return s1.center != s2.center
            || s1.radius != s2.radius;
    }

    template<typename T>
    std::ostream& operator<< (std::ostream& os, const Sphere<T>& circle) {
        return os << "Sphere { center: (" << circle.center[0] << "," << circle.center[1] << "," << circle.center[2] << "), "
            "radius: " << circle.radius << " }";
    }

}

#endif