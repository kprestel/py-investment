from cmath import log

from scipy.optimize import newton


class TVM(object):
    """Do TVM stuff here."""

    begin, end = 0, 1

    def __init__(
        self,
        periods: float = 0.0,
        rate: float = 0.0,
        pv: float = 0.0,
        pmt: float = 0.0,
        fv: float = 0.0,
        mode: int = end,
    ):
        """
        Constructor stuff...
        :param periods: The number of periods to use in the tvm calculations.
        :param rate: The coupon rate. In percent.
        :param pv: The current present value.
        :param pmt: The amount the bond pays.
        :param fv: The future value of the bond.
        :param mode:
        """
        self.periods = periods
        self.rate = rate
        self.pv = pv
        self.pmt = pmt
        self.fv = fv
        self.discount_factor = pow(1 + self.rate, -self.periods)
        self.mode = mode

        try:
            pva = self.pmt / self.rate
        except ZeroDivisionError:
            self.pva = 0.0
        else:
            if self.mode == TVM.begin:
                self.pva = pva + self.pmt
            else:
                self.pva = pva

    def calc_pv(self):
        """Calculate present value."""
        if self.pmt == 0.0:
            return -((self.periods * self.rate) / 360 - self.fv)
        return -(self.pv * self.discount_factor + (1 - self.discount_factor) * self.pva)

    def calc_fv(self):
        return -(self.pv + (1 - self.discount_factor) * self.pva) / self.discount_factor

    def calc_pmt(self):
        if self.mode == TVM.begin:
            return (
                (self.pv + self.fv * self.discount_factor)
                * self.rate
                / (self.discount_factor - 1)
                / (1 + self.rate)
            )
        else:
            return (
                (self.pv + self.fv * self.discount_factor)
                * self.rate
                / (self.discount_factor - 1)
            )

    def calc_periods(self):
        discount = (-self.discount_factor - self.pv) / (self.fv - self.discount_factor)
        return -log(discount) / log(1 + self.rate)

    def calc_discount_rate(self):
        def fv_(r, self):
            z = pow(1 + r, -self.periods)
            pva = self.pmt / r
            if self.mode == TVM.begin:
                pva += self.pmt
            return -(self.pv + (1 - z) * pva) / z

        return newton(func=fv_, x0=0.05, args=(self,), maxiter=1000, tol=0.00001)
