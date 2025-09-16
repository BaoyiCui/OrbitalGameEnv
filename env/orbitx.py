import numpy as np
import math
import datetime
from numpy import sin, cos, sqrt
from scipy.optimize import fsolve


def stumpC(z):
    if z > 0:
        return (1 - np.cos(np.sqrt(z))) / z
    elif z < 0:
        return (np.cosh(np.sqrt(-z)) - 1) / (-z)
    return 1 / 2


def stumpS(z):
    if z > 0:
        return (np.sqrt(z) - np.sin(np.sqrt(z))) / (np.sqrt(z)) ** 3
    elif z < 0:
        return (np.sinh(np.sqrt(-z)) - np.sqrt(-z)) / (np.sqrt(-z)) ** 3
    return 1 / 6


def kepler_U(dt, ro, vro, a):
    mu = 398600

    error = 1.e-8
    nMax = 1000
    x = math.sqrt(mu) * abs(a) * dt

    n = 0
    ratio = 1
    while (abs(ratio) > error) & (n <= nMax):
        n = n + 1
        C = stumpC(a * x ** 2)
        S = stumpS(a * x ** 2)
        F = ro * vro / math.sqrt(mu) * x ** 2 * C + (1 - a * ro) * x ** 3 * S + ro * x - math.sqrt(mu) * dt
        dFdx = ro * vro / math.sqrt(mu) * x * (1 - a * x ** 2 * S) + (1 - a * ro) * x ** 2 * C + ro

        ratio = F / dFdx
        x = x - ratio

    return x


def f_and_g(x, t, ro, a):
    mu = 398600

    z = a * x ** 2

    # ...Equation 3.66a:
    f = 1 - x ** 2 / ro * stumpC(z)

    # ...Equation 3.66b:
    g = t - 1 / math.sqrt(mu) * x ** 3 * stumpS(z)

    return f, g


def fDot_and_gDot(x, r, ro, a):
    mu = 398600
    z = a * x ** 2

    # ...Equation 3.66c:
    fdot = math.sqrt(mu) / r / ro * (z * stumpS(z) - 1) * x

    # Equation 3.66d:
    gdot = 1 - x ** 2 / r * stumpC(z)

    return fdot, gdot


def rv_from_r0v0(RV0, t):
    mu = 398600
    R0 = RV0[0:3]
    V0 = RV0[3:6]

    r0 = np.linalg.norm(R0)
    v0 = np.linalg.norm(V0)
    vr0 = np.dot(R0, V0) / r0
    alpha = 2 / r0 - v0 ** 2 / mu

    # 计算x，利用开普勒——U
    x = kepler_U(t, r0, vr0, alpha)

    f, g = f_and_g(x, t, r0, alpha)

    R = f * R0 + g * V0

    r = np.linalg.norm(R)

    fdot, gdot = fDot_and_gDot(x, r, r0, alpha)

    V = fdot * R0 + gdot * V0

    RV = np.array([R[0], R[1], R[2], V[0], V[1], V[2]]).T

    return RV


def solar_position(jd):
    #
    # This function alculates the geocentric equatorial position vector
    # of the sun, given the julian date.
    # -------------------------------------------------------------------------
    # ...Astronomical unit (km):
    AU = 149597870.691
    # ...Julian days since J2000:
    n = jd - 2451545
    # ...Julian centuries since J2000:
    cy = n / 36525
    # ...Mean anomaly (deg{:
    M = 357.528 + 0.9856003 * n
    M = M % 360

    # ...Mean longitude (deg):
    L = 280.460 + 0.98564736 * n
    L = L % 360

    # ...Apparent ecliptic longitude (deg):
    lamda = L + 1.915 * np.sin(np.deg2rad(M)) + 0.020 * np.sin(np.deg2rad(2 * M))
    lamda = lamda % 360

    # ...Obliquity of the ecliptic (deg):
    eps = 23.439 - 0.0000004 * n

    # ...Unit vector from earth to sun:
    # u     = [cosd(lamda) sind(lamda)*cosd(eps) sind(lamda)*sind(eps)]
    u = np.array([np.cos(np.deg2rad(lamda)), np.sin(np.deg2rad(lamda)) * np.cos(np.deg2rad(eps)),
                  np.sin(np.deg2rad(lamda)) * np.sin(np.deg2rad(eps))])

    # ...Distance from earth to sun (km):
    rS = (1.00014 - 0.01671 * np.cos(np.deg2rad(M)) - 0.000140 * np.cos(np.deg2rad(2 * M))) * AU

    # ...Geocentric position vector (km):
    r_S = rS * u  # solar_position
    return r_S


def time2jd(dateT):
    t0 = datetime.datetime(1858, 11, 17, 0, 0, 0, 0)  # 简化儒略日起始日
    mjd = (dateT - t0).days
    mjd_s = dateT.hour * 3600.0 + dateT.minute * 60.0 + dateT.second + dateT.microsecond / 1000000.0
    jd = mjd + mjd_s / 86400.0 + 2400000.5  # 简化儒略日转化为儒略日
    return jd


# jd转datetime类
def jd2time(jd):
    t0 = datetime.datetime(1858, 11, 17, 0, 0, 0, 0)  # 简化儒略日起始日
    mjd = jd - 2400000.5
    return t0 + datetime.timedelta(days=mjd)


def dFdz(z, r1, r2, A):
    if z == 0:
        dum = np.sqrt(2) / 40 * (r1 + r2 + A * (z * stumpS(z) - 1) / np.sqrt(stumpC(z))) ** 1.5 + A / 8 * (
                np.sqrt(np.abs(r1 + r2 + A * (z * stumpS(z) - 1) / np.sqrt(stumpC(z)))) + A * np.sqrt(
            1 / 2 / (r1 + r2 + A * (z * stumpS(z) - 1) / np.sqrt(stumpC(z)))))
    else:
        dum = ((r1 + r2 + A * (z * stumpS(z) - 1) / np.sqrt(stumpC(z))) / stumpC(z)) ** 1.5 * (
                1 / 2 / z * (stumpC(z) - 3 * stumpS(z) / 2 / stumpC(z)) + 3 * stumpS(z) ** 2 / 4 / stumpC(
            z)) + A / 8 * (
                      3 * stumpS(z) / stumpC(z) * np.sqrt(
                  np.abs(r1 + r2 + A * (z * stumpS(z) - 1) / np.sqrt(stumpC(z)))) + A * np.sqrt(
                  stumpC(z) / (r1 + r2 + A * (z * stumpS(z) - 1) / np.sqrt(stumpC(z)))))
    return dum


def lambert(R1, R2, t):
    mu = 398600
    r1 = np.linalg.norm(R1)
    r2 = np.linalg.norm(R2)
    c12 = np.cross(R1, R2)
    theta = np.arccos(np.dot(R1, R2) / (r1 * r2))
    if c12[2] <= 0:
        theta = 2 * np.pi - theta

    A = np.sin(theta) * np.sqrt(r1 * r2 / (1 - np.cos(theta)))

    z = -100
    #  (((r1 + r2 + A*(z*stumpS(z) - 1)/sqrt(stumpC(z)))/stumpC(z))^1.5*stumpS(z) + A*sqrt((r1 + r2 + A*(z*stumpS(z) - 1)/sqrt(stumpC(z)))) - sqrt(mu)*t) < 0
    xx = (((((r1 + r2 + A * (z * stumpS(z) - 1) / np.sqrt(stumpC(z))) / stumpC(z)) ** 1.5) * stumpS(z) + A * np.sqrt(
        np.abs(r1 + r2 + A * (z * stumpS(z) - 1) / np.sqrt(stumpC(z))))) - np.sqrt(mu) * t)
    while np.isnan(xx):
        z += 0.1
        xx = (((((r1 + r2 + A * (z * stumpS(z) - 1) / np.sqrt(stumpC(z))) / stumpC(z)) ** 1.5) * stumpS(
            z) + A * np.sqrt(np.abs(r1 + r2 + A * (z * stumpS(z) - 1) / np.sqrt(stumpC(z))))) - np.sqrt(mu) * t)

    while xx < 0:
        z += 0.1
        xx = (((((r1 + r2 + A * (z * stumpS(z) - 1) / np.sqrt(stumpC(z))) / stumpC(z)) ** 1.5) * stumpS(
            z) + A * np.sqrt(np.abs(r1 + r2 + A * (z * stumpS(z) - 1) / np.sqrt(stumpC(z))))) - np.sqrt(mu) * t)

    tol = 1e-8
    nmax = 5000

    ratio = 1
    n = 0
    while (abs(ratio) > tol) and (n <= nmax):
        n += 1
        ratio = ((((r1 + r2 + A * (z * stumpS(z) - 1) / np.sqrt(stumpC(z))) / stumpC(z)) ** 1.5 * stumpS(
            z) + A * np.sqrt((r1 + r2 + A * (z * stumpS(z) - 1) / np.sqrt(stumpC(z))))) - np.sqrt(mu) * t) / dFdz(z, r1,
                                                                                                                  r2, A)
        z -= ratio

    if n >= nmax:
        print('\n\n **Number of iterations exceeds %g in ''lambert'' \n\n ' % nmax)

    f = 1 - (r1 + r2 + A * (z * stumpS(z) - 1) / np.sqrt(stumpC(z))) / r1

    g = A * np.sqrt((r1 + r2 + A * (z * stumpS(z) - 1) / np.sqrt(stumpC(z))) / mu)

    gdot = 1 - (r1 + r2 + A * (z * stumpS(z) - 1) / np.sqrt(stumpC(z))) / r2

    V1 = 1 / g * (R2 - f * R1)

    V2 = 1 / g * (gdot * R2 - R1)
    return V1, V2


def coe2rv(coe):
    mu = 398600  # gravitational constant (km**3/sec**2)
    r = np.array([0.0, 0.0, 0.0])
    v = np.array([0.0, 0.0, 0.0])

    # unload orbital elements array
    sma = coe[0]  # 半长轴    semimajor axis (kilometers)
    ecc = coe[1]  # 偏心率    orbital eccentricity (non-dimensional)
    inc = coe[2] * np.pi / 180  # 轨道倾角  orbital inclination (deg)
    raan = coe[3] * np.pi / 180  # 升交点赤经 right ascension of ascending node (deg)
    argper = coe[4] * np.pi / 180  # 偏近点角  argument of perigee (deg)
    M = coe[5] * np.pi / 180  # 平近点角

    tanom = Keplers_Eqn(M, ecc)
    # tanom = coe[5] * np.pi/180  # 真近点角

    slr = sma * (1 - ecc * ecc)

    rm = slr / (1 + ecc * np.cos(tanom))

    arglat = argper + tanom

    sarglat = np.sin(arglat)
    carglat = np.cos(arglat)

    c4 = np.sqrt(mu / slr)
    c5 = ecc * np.cos(argper) + carglat
    c6 = ecc * np.sin(argper) + sarglat

    sinc = np.sin(inc)
    cinc = np.cos(inc)

    sraan = np.sin(raan)
    craan = np.cos(raan)

    # position vector
    r[0] = rm * (craan * carglat - sraan * cinc * sarglat)
    r[1] = rm * (sraan * carglat + cinc * sarglat * craan)
    r[2] = rm * sinc * sarglat

    # velocity vector
    v[0] = -c4 * (craan * c6 + sraan * cinc * c5)
    v[1] = -c4 * (sraan * c6 - craan * cinc * c5)
    v[2] = c4 * c5 * sinc
    return np.array([r, v]).reshape(6)


# #### Method 2 for coe2rv
#     # INPUT:
#     #
#     #       a = semi-major axis                          [length]*  km
#     #       e = eccentrity                               [rad]
#     #       i = inclination                              [rad]
#     #   OMEGA = right ascension of the ascending node    [rad]
#     #       w = argument of perigee                      [rad]
#     #       M = mean anomaly                             [rad]
#     #
#     # -------------------------------------------------------------------------
#     # OUTPUT:
#     #
#     #       X = ECI position vector of the spacecraft    [length]*
#     #       V = ECI veloicty vector of the spacecraft    [length / time]*
#     #
#     # -------------------------------------------------------------------------
#     # NOTES:
#     #
#     # * This quantity can be expressed in either m or km or etc as long
#     #   as the global value of mu (the Earth's gravitational parameter) is in
#     #   consitant units.
#     #
#     ## GLOBAL VARIABLES USED

#     mu = 398600.4418

#     ## MAIN ALGORITHM
#     # # unload orbital elements array
#     a = coe[0]                 # 半长轴    semimajor axis (kilometers)
#     e = coe[1]                 # 偏心率    orbital eccentricity (non-dimensional)
#     i = coe[2]* np.pi/180      # 轨道倾角  orbital inclination (deg)
#     OMEGA = coe[3] * np.pi/180 # 升交点赤经 right ascension of ascending node (deg)
#     w = coe[4] * np.pi/180     # 近地点幅角  argument of perigee (deg)
#     M = coe[5] * np.pi/180     # 平近点角  Mean

#     # Handle special cases:

#     # Circular equitorial orbit.
#     if e == 0 and i == 0:
#         w     = 0
#         OMEGA = 0
#     # Circular inclined orbit.
#     elif e == 0:
#         w     = 0
#     # Elliptical equitorial.
#     elif i == 0:
#         OMEGA = 0
#     # Compute the semi-latus rectum.
#     p = a * ( 1 - e ** 2)
#     # Convert mean anomaly to true anomaly.
#     # First, compute the eccentric anomaly.
#     Ea = Keplers_Eqn(M,e)
#     # Compute the true anomaly f.
#     yy = np.sin(Ea)*np.sqrt(1-e**2)/(1-e*np.cos(Ea))
#     xx = (np.cos(Ea)-e)/(1-e*np.cos(Ea))

#     f = math.atan2(yy,xx)

#     X = np.mat([[0],[0],[0]])
#     # Define the position vector in perifocal PQW coordinates.
#     X[0][0] = p*np.cos(f)/(1+e*np.cos(f))
#     X[1][0] = p*np.sin(f)/(1+e*np.cos(f))
#     X[2][0] = 0

#     print(X)
#     V = np.mat([[0],[0],[0]])

#     # Define the velocity vector in perifocal PQW coordinates.
#     V[0][0] = -np.sqrt(mu/p)*np.sin(f)
#     V[1][0] =  np.sqrt(mu/p)*(e+np.cos(f))
#     V[2][0] =  0
#     print(V)
#     Trans = np.array([[0,0,0],[0,0,0],[0,0,0]])
#     # Define Transformation Matrix To IJK.
#     Trans[0][0] =  cos(OMEGA)*cos(w)-sin(OMEGA)*sin(w)*cos(i)
#     Trans[0][1] = -cos(OMEGA)*sin(w)-sin(OMEGA)*cos(w)*cos(i)
#     Trans[0][2] =  sin(OMEGA)*sin(i)

#     Trans[1][0] =  sin(OMEGA)*cos(w)+cos(OMEGA)*sin(w)*cos(i)
#     Trans[1][1] = -sin(OMEGA)*sin(w)+cos(OMEGA)*cos(w)*cos(i)
#     Trans[1][2] = -cos(OMEGA)*sin(i)

#     Trans[2][0] =  sin(w)*sin(i)
#     Trans[2][1] =  cos(w)*sin(i)
#     Trans[2][2] =  cos(i)

#     print('Trans',Trans)

#     # Transform to the ECI coordinate system.
#     X = Trans*X
#     V = Trans*V
#     print('X',X)
#     print('V',V)

#     return np.array([X[0][0],X[1][0],X[2][0],V[0][0],V[1][0],V[2][0]]).reshape(6)

def Keplers_Eqn(M, e):
    # -------------------------------------------------------------------------
    # Given the mean anomaly M, eccentricity e, compute the eccentric anomaly E
    # -------------------------------------------------------------------------
    # INPUT:
    #       e = eccentrity                               [rad]
    #       M = mean anomaly                             [rad]
    # -------------------------------------------------------------------------
    # OUTPUT:
    #       E = eccentric anomaly                        [rad]

    if -np.pi < M < 0 or M > np.pi:
        E_a = M - e
    else:
        E_a = M + e
    # Define tolerance.
    tol = 1e-12
    test = 999  # Dummy variable.
    # Implement Newton's method.
    while test > tol:
        E_new = E_a + (M - E_a + e * np.sin(E_a)) / (1 - e * np.cos(E_a))
        test = np.abs(E_new - E_a)
        E_a = E_new
    return E_a


# def eci2hills(x0, x1):
#     # TO DO,cannot run
#     ro = x0[:3].reshape(3,-1)
#     vo = x0[3:].reshape(3,-1)

#     x = ro/np.linalg.norm(ro)                                   # Unit( ro ); % x is radial
#     z = np.cross( ro, vo)/np.linalg.norm(np.cross( ro, vo))     # Unit( cross( ro, vo) ); % z is + orbit normal
#     y = np.cross( z, x )/np.linalg.norm(np.cross( z, x ))       #  Unit( cross( z, x ) ); % y completes RHS


#     A = np.array([x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2]]).reshape(3,3)
#     xH = np.array([0,0,0,0,0,0]).reshape(6,-1)
#     xH[:3][0] = A * (x1[:3][0]-x0[:3][0])

#     w_eci = np.cross(x0[:3][0],x0[3:][0])/np.linalg.norm(x0[:3][0]) ** 2
#     wmat = np.array([0,-w_eci(2),w_eci(1), w_eci(2),0,-w_eci(0),  -w_eci(1),w_eci(0),0]).reshape(3,3)
#     xH[3:][0] = A * (x1[3:][0]-x0[3:][0]) - A*wmat*(x1[:3][0]-x0[:3][0])
#     return xH


# ------------------------------
##  eci2hills  ##

def ECI2Hills(x0, x1):
    ro = x0[:3]
    vo = x0[3:]

    x = ro / np.linalg.norm(ro)  # Unit( ro ); % x is radial
    z = np.cross(ro, vo) / np.linalg.norm(np.cross(ro, vo))  # Unit( cross( ro, vo) ); % z is + orbit normal
    y = np.cross(z, x) / np.linalg.norm(np.cross(z, x))  # Unit( cross( z, x ) ); % y completes RHS

    xtemp = x1[:3] - x0[:3]
    A = np.array([x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2]]).reshape(3, 3)

    xHr = np.array([np.dot(x, xtemp), np.dot(y, xtemp), np.dot(z, xtemp)])

    w_eci = np.cross(x0[:3], x0[3:]) / np.linalg.norm(x0[:3]) ** 2
    # wmat = np.array([0,-w_eci[2],w_eci[1], w_eci[2],0,-w_eci[0],  -w_eci[1],w_eci[0],0]).reshape(3,3)

    wmat1 = np.array([0, -w_eci[2], w_eci[1]])
    wmat2 = np.array([w_eci[2], 0, -w_eci[0]])
    wmat3 = np.array([-w_eci[1], w_eci[0], 0])

    vtemp = x1[3:] - x0[3:]
    vtemp = vtemp - np.array([np.dot(wmat1, xtemp), np.dot(wmat2, xtemp), np.dot(wmat3, xtemp)])
    xHv = np.array([np.dot(x, vtemp), np.dot(y, vtemp), np.dot(z, vtemp)])

    return np.array([xHr, xHv]).reshape(6)


# ------------------------------
##  hills2eci  ##
def Hills2ECI(x0, xH):
    # TO DO,cannot run
    ro = x0[:3]
    vo = x0[3:]

    x = ro / np.linalg.norm(ro)  # Unit( ro ); % x is radial
    z = np.cross(ro, vo) / np.linalg.norm(np.cross(ro, vo))  # Unit( cross( ro, vo) ); % z is + orbit normal
    y = np.cross(z, x) / np.linalg.norm(np.cross(z, x))  # Unit( cross( z, x ) ); % y completes RHS

    A = np.array([x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2]]).reshape(3, 3)

    xx = np.array([x[0], y[0], z[0]])
    yy = np.array([x[1], y[1], z[1]])
    zz = np.array([x[2], y[2], z[2]])

    rH = xH[:3]
    vH = xH[3:]
    x1r = ro + np.array([np.dot(xx, rH), np.dot(yy, rH), np.dot(zz, rH)])
    # xHr = np.array([np.dot(xx, rH), np.dot(yy, rH), np.dot(zz, rH)])

    w_eci = np.cross(x0[:3], x0[3:]) / np.linalg.norm(x0[:3]) ** 2
    # wmat = np.array([0,-w_eci[2],w_eci[1], w_eci[2],0,-w_eci[0],  -w_eci[1],w_eci[0],0]).reshape(3,3)

    wmat1 = np.array([0, -w_eci[2], w_eci[1]])
    wmat2 = np.array([w_eci[2], 0, -w_eci[0]])
    wmat3 = np.array([-w_eci[1], w_eci[0], 0])

    vtemp = np.array([np.dot(xx, rH), np.dot(yy, rH), np.dot(zz, rH)])
    vtemp = np.array([np.dot(wmat1, vtemp), np.dot(wmat2, vtemp), np.dot(wmat3, vtemp)])
    x1v = vo + np.array([np.dot(xx, vH), np.dot(yy, vH), np.dot(zz, vH)]) + vtemp

    return np.array([x1r, x1v]).reshape(6)


def HCW(t, n, rv0):
    # TO DO TEST
    mu = 398600
    a = np.linalg.norm(rv0[:3])
    n = sqrt(mu / a ** 3)
    x0 = rv0[0]
    y0 = rv0[1]
    z0 = rv0[2]
    vx0 = rv0[3]
    vy0 = rv0[4]
    vz0 = rv0[5]
    x_f = (vx0 / n) * sin(n * t) - (3 * x0 + 2 * vy0 / n) * cos(n * t) + 4 * x0 + 2 * vy0 / n
    y_f = (6 * x0 + 4 * vy0 / n) * sin(n * t) + (2 * vx0 / n) * cos(n * t) + y0 - 2 * vx0 / n - (
            6 * n * x0 + 3 * vy0) * t
    z_f = z0 * cos(n * t) + vz0 * sin(n * t) / n
    vx_f = (2 * vy0 + 3 * n * x0) * sin(n * t) + vx0 * cos(n * t)
    vy_f = -2 * vx0 * sin(n * t) + (4 * vy0 + 6 * n * x0) * cos(n * t) - 3 * (vy0 + 2 * n * x0)
    vz_f = -n * z0 * sin(n * t) + vz0 * cos(n * t)
    return np.array([x_f, y_f, z_f, vx_f, vy_f, vz_f])


def HCW_pulse(t0, n, X_t0, tf, X_tf):
    t = tf - t0
    A = np.array([4 - 3 * cos(n * t), 0, 0, 6 * (sin(n * t) - n * t), 1, 0, 0, 0, cos(n * t)]).reshape(3, 3)
    B = np.array(
        [sin(n * t) / n, 2 * (1 - cos(n * t)) / n, 0, -2 * (1 - cos(n * t)) / n, 4 * sin(n * t) / n - 3 * t, 0, 0, 0,
         sin(n * t) / n]).reshape(3, 3)
    C = np.array([3 * n * sin(n * t), 0, 0, -6 * n * (1 - cos(n * t)), 0, 0, 0, 0, -n * sin(n * t)]).reshape(3, 3)
    D = np.array([cos(n * t), 2 * sin(n * t), 0, -2 * sin(n * t), 4 * cos(n * t) - 3, 0, 0, 0, cos(n * t)]).reshape(3,
                                                                                                                    3)

    deltaV0 = np.dot(np.linalg.inv(B), (X_tf[:3] - np.dot(A, X_t0[:3]))) - X_t0[3:]

    # print('deltaV0',deltaV0)
    deltaVf = X_tf[3:] - (np.dot(C, X_t0[:3]) + np.dot(D, X_t0[3:]) + np.dot(D, deltaV0))
    # print('deltaVf',deltaVf)
    return deltaV0, deltaVf


def HCW_pulse_optmz(mu, rv0, rvf, trans_time):
    ##  function for Culculating the two pulse orbital trans in HCW frame.
    #    Using the Target optimize method.
    #    Last Edited By Xuxusheng in 20240527.
    #    Required:twobody2,  ECI2Hills, HCW_Pulse, Hills2ECI, pinv

    # Target point in Hills Frame
    mu = 398600

    tarpoint = rvf
    tarpoint_rv0 = rv_from_r0v0(rv0, trans_time)
    tarpoint_rel = ECI2Hills(tarpoint_rv0, tarpoint)

    # Target point in Hills Frame
    n = sqrt(mu / np.linalg.norm(tarpoint_rv0[:3]) ** 3)
    deltaV0, deltaVf = HCW_pulse(0, n, np.array([0, 0, 0, 0, 0, 0]), trans_time, tarpoint_rel)

    deltaV0eci = Hills2ECI(rv0, np.array([0, 0, 0, deltaV0[0], deltaV0[1], deltaV0[2]])) - rv0
    deltaV0eci = deltaV0eci[3:]

    deltaVfeci = Hills2ECI(tarpoint_rv0, np.array([0, 0, 0, deltaVf[0], deltaVf[1], deltaVf[2]])) - tarpoint_rv0
    deltaVfeci = deltaVfeci[3:]

    fai_v1 = np.array([sin(n * trans_time) / n, 2 * (1 - cos(n * trans_time)) / n, 0.0, 0.0, 0.0, 0.0])
    fai_v2 = np.array(
        [-2 * (1 - cos(n * trans_time)) / n, 4 * sin(n * trans_time) / n - 3 * trans_time, 0.0, 0.0, 0.0, 0.0])
    fai_v3 = np.array([0, 0.0, sin(n * trans_time) / n, 0.0, 0.0, 0.0])
    fai_v4 = np.array([cos(n * trans_time), 2 * sin(n * trans_time), 0.0, 1.0, 0.0, 0.0])
    fai_v5 = np.array([-2 * sin(n * trans_time), 4 * cos(n * trans_time) - 3, 0.0, 0.0, 1.0, 0.0])
    fai_v6 = np.array([0, 0.0, cos(n * trans_time), 0.0, 0.0, 1.0])
    fai_v = np.array([fai_v1, fai_v2, fai_v3, fai_v4, fai_v5, fai_v6]).reshape(6, 6)

    x_need_tar_eci = rv_from_r0v0(rv0 + np.array([0, 0, 0, deltaV0eci[0], deltaV0eci[1], deltaV0eci[2]]), trans_time)
    x_need_tar_eci = x_need_tar_eci + np.array([0, 0, 0, deltaVfeci[0], deltaVfeci[1], deltaVfeci[2]])
    x_need_tar = ECI2Hills(tarpoint_rv0, x_need_tar_eci)

    xtar = tarpoint_rel
    epsillon_x = xtar - x_need_tar

    delta_v = np.array([deltaV0[0], deltaV0[1], deltaV0[2], deltaVf[0], deltaVf[1], deltaVf[2]])
    Next_delta_v = delta_v + np.dot(np.linalg.inv(fai_v), epsillon_x)

    cnt = 1000  # 最大迭代次数
    for i in range(cnt):
        delta_v = Next_delta_v

        deltaV0 = delta_v[:3]

        deltaV0eci = Hills2ECI(rv0, np.array([0, 0, 0, deltaV0[0], deltaV0[1], deltaV0[2]])) - rv0
        deltaV0eci = deltaV0eci[3:]

        deltaVf = delta_v[3:]
        deltaVfeci = Hills2ECI(tarpoint_rv0, np.array([0, 0, 0, deltaVf[0], deltaVf[1], deltaVf[2]])) - tarpoint_rv0
        deltaVfeci = deltaVfeci[3:]

        x_need_tar_eci = rv_from_r0v0(rv0 + np.array([0, 0, 0, deltaV0eci[0], deltaV0eci[1], deltaV0eci[2]]),
                                      trans_time)
        x_need_tar_eci = x_need_tar_eci + np.array([0, 0, 0, deltaVfeci[0], deltaVfeci[1], deltaVfeci[2]])
        x_need_tar = ECI2Hills(tarpoint_rv0, x_need_tar_eci)

        epsillon_x = xtar - x_need_tar
        Next_delta_v = delta_v + np.dot(np.linalg.inv(fai_v), epsillon_x)

        if (np.linalg.norm(epsillon_x) < 1e-6):
            # print('----------  Finish optimized !!!  ----------------')
            break

    return deltaV0eci, deltaVfeci