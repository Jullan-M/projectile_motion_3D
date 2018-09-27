__author__ = 'Jullan'
# -*- coding: utf-8 -*-
#Made by Jullan

class Rk4:
    def __init__(self, t0, y0, dt, f):
        self.t0 = t0
        self.ti = t0
        self.y0 = y0
        self.yi = y0
        self.dt = dt
        self.f = f
        self.n = 0

    def rk4(self):
        f1 = self.f(self.ti, self.yi)
        f2 = self.f(self.ti + self.dt / 2.0, self.yi + self.dt / 2.0 * f1)
        f3 = self.f(self.ti + self.dt / 2.0, self.yi + self.dt / 2.0 * f2)
        f4 = self.f(self.ti + self.dt, self.yi + self.dt * f3)

        self.yi = self.yi + self.dt / 6.0 * (f1 + 2.0 * f2 + 2.0 * f3 + f4)
        self.n = self.n + 1
        self.ti = self.ti + self.dt

    def test(self, t_max):
        import platform

        print('')
        print('RK4_TEST')
        print('  RK4 takes one Runge-Kutta step for a scalar ODE.')

        print('')
        print('\t n \t\t\t t \t\t\t\t y(t)')

        t_num = int((t_max-self.t0)/self.dt)

        for i in range(t_num): #  Stop if we've exceeded TMAX.
            #  Print (n,t,y).
            print('  %4d  %15.10f  %15.10g' % (self.n, self.ti, self.yi))
            self.rk4()
        print('  %4d  %15.10f  %15.10g' % (self.n, self.ti, self.yi))

    def reset(self): # Resets the counter and current step of rk4.
        self.ti = self.t0
        self.yi = self.y0
        self.n = 0

    def append_y(self, np_arr, t_max): # Makes array of y-values from ti to t_max. Assumes empty numpy array.
        t_num = int((t_max-self.t0)/self.dt)

        np_arr.append(self.yi)
        for i in range(t_num-1):
            self.rk4()
            np_arr.append(self.yi)


def rk4_test_f(t, y):
    value = t*y+t**3
    return value

if (__name__ == '__main__'):
    problem = Rk4(0, 1, 0.05, rk4_test_f)
    problem.test(1.0)