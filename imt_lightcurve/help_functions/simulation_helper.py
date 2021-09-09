from math import log, sqrt, pi, acos

def ellk(k) -> float:
    """
    Computes polynomial approximation for the complete
    elliptic integral of the first kind (Hasting's approximation)
    https://doi.org/10.2307/2004103
    Table II, coefficients values from n=4
    :param FLOAT? k: 
    :return: The complete elliptical integral of the first kind
    :rtype: FLOAT?
    """
    m1 = 1 - k**2

    # Coefficients for K*
    a0 = log(4)
    a1 = 0.09666344259
    a2 = 0.03590092383
    a3 = 0.03742563713
    a4 = 0.01451196212
    b0 = 0.5
    b1 = 0.12498593597
    b2 = 0.06880248576
    b3 = 0.03328355346
    b4 = 0.00441787012

    ek1 = a0+m1*(a1+m1*(a2+m1*(a3+m1*a4)))
    try:
        ek2 = (b0+m1*(b1+m1*(b2+m1*(b3+m1*b4))))*log(m1)
    except ValueError:
        print('Trying to calculate log of', m1)
        print('m1 = 1 - k**2')
        print('k = ', k, end='\n\n')
        raise
        # pass
    

    return ek1 - ek2

def ellec(k) -> float:
    """
    Computes polynomial approximation for the complete
    elliptic integral of the second kind (Hasting's approximation)
    https://doi.org/10.2307/2004103
    Table III, coefficients values from n=4
    :param float k:
    :return: The complete elliptical integral of the second kind
    :rtype: 
    """
    m1 = 1 - k**2

    # Coefficients for E*
    c1 = 0.44325141463
    c2 = 0.06260601220
    c3 = 0.04757383546
    c4 = 0.01736506451
    d1 = 0.24998368310
    d2 = 0.09200180037
    d3 = 0.04069697526
    d4 = 0.00526449639

    ee1 = 1+m1*(c1+m1*(c2+m1*(c3+m1*c4)))
    ee2 = m1*(d1+m1*(d2+m1*(d3+m1*d4)))*log(1/m1)

    return ee1 + ee2

def ellpic_bulirsch(n, k) -> float:
    """
    Computes the complete elliptical integral of the third kind 
    using the algorithm of Bulirsch (1965)
    https://doi.org/10.1007/BF02165405
    :param FLOAT? n: 
    :param FLOAT? k: 
    :return: The complete elliptical integral of the third kind
    :rtype: 
    """
    kc = sqrt(1 - k**2)
    p = n + 1

    # if min(p) < 0:
    if p < 0:
        print('Negative p')

    m0 = 1
    c = 1
    p = sqrt(p)
    d = 1/p
    e = kc
    d = 1/p
    e = kc

    iter = 0
    while iter < 20:
        f = c
        c = d/p+c
        g = e/p
        d = 2*(f*g + d)

        p = g + p
        g = m0
        m0 = kc + m0

        # if max(abs(1 - kc/g)) > 1-8:
        if (abs(1 - kc/g)) > 1-8:
            kc = 2 * sqrt(e)
            e = kc * m0

        else:
            return 0.5*pi*(c*m0+d)/(m0*(m0+p))

        iter += 1

    return 0.5*pi*(c*m0+d)/(m0*(m0+p))


## Functions from Table 1, Mandel & Agol (2008)
def calculate_lambda_1(a, b, k, p, q, w, z) -> float:
    return (1/(9*pi*sqrt(p*z[w]))) * (((1-b)*(2*b+a-3)-3*q*(b-2))*ellk(k)+4*p*z[w]*(z[w]**2+7*p**2-4)*ellec(k)-(3*q/a)*ellpic_bulirsch(abs((a-1)/a), k))

def calculate_lambda_2(a, b, k, p, q, w, z) -> float:
    return (2/(9*pi*sqrt(1-a))) * ((1-5*z[w]**2+p**2 + q**2)*ellk(k**(-1))+(1-a)*(z[w]**2+7*p**2-4)*ellec(k**(-1))-(3*q/a)*ellpic_bulirsch(abs((a-b)/a), k**(-1)))

def calculate_lambda_3(p) -> float:
    return (1/3)+((16*p)/(9*pi))*(2*p**2-1)*ellec(1/(2*p))-(((1-4*p**2)*(3-8*p**2))/(9*pi*p)*ellk(1/(2*p)))

def calculate_lambda_4(p) -> float:
    return (1/3)+(2/(9*pi))*(4*(2*p**2-1)*ellec(2*p)+(1-4*p**2)*ellk(2*p))

def calculate_lambda_5(p) -> float:
    if (p <= 0.5):
        return (2*(3*pi))*acos(1-2*p)-(4/(9*pi))*(3+2*p-8*p**2)*sqrt(p*(1-p))
    elif (p > 0.5):
        return (2*(3*pi))*acos(1-2*p)-(4/(9*pi))*(3+2*p-8*p**2)*sqrt(p*(1-p))-(2/3)

def calculate_lambda_6(p) -> float:
    return -(2/3)*(1-p**2)**(3/2)

def calculate_eta_2(p, w, z) -> float:
    return ((p**2)/2)*(p**2+2*z[w]**2)

def calculate_eta_1(a, b, k0, k1, p, w, z) -> float:
    return (2*pi)**(-1)*(k1+2*calculate_eta_2(p, w, z)*k0-0.25*(1+5*p**2+z[w]**2)*sqrt((1-a)*(b-1)))

def calculate_flux(c2, c4, Omega, lambda_e, lambda_d, eta_d, p, w, z) -> float:
    if (p <= z[w]):
        return 1-(4*Omega)**(-1)*((1-c2)*lambda_e+c2*(lambda_d-c4*eta_d))

    if (p > z[w]):
        return 1-(4*Omega)**(-1)*((1-c2)*lambda_e+c2*(lambda_d+(2/3)-c4*eta_d))
