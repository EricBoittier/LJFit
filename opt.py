import numpy as np
import pandas as pd

def LJ(sig, ep, r, a=6, b=2, c=2):
    """
    Lennard-Jones potential for a pair of atoms
    """
    r6 = (sig / r) ** a
    return ep * (r6 ** b - c * r6)

def freeLJ(sig, ep, r, a, b, c):
    """
    Lennard-Jones potential for a pair of atoms
    """
    return ep * ((sig / r) ** a - c * (sig / r) ** b)

#  double exp. pot.
#  https://chemrxiv.org/engage/chemrxiv/article-details/6401c0a163e8d44e594addea
def DE(x, a, b, c, e):
    """
    Double exponential potential
    """
    return e * (((b * np.exp(a)) / (a - b)) * np.exp(-a * (x / c))
                - ((a * np.exp(b)) / (a - b)) * np.exp(-b * (x / c)))

# def DEplus(x, a, b, c, e, f, g):
#     """
#     Double exponential potential
#     """
#     return e * (
#             (( b * np.exp(a)) / (a - b)) * np.exp(-a * (x / c))
#                 - ((a * np.exp(b)) / (a - b)) * np.exp(-b * (x / c))
#             + (((a * b) * np.exp(f)) / (a - b)) * np.exp(-f * (x / c))
#                 )

# def DEplus(x, a, b, c, e, f, g):
#     """
#     Double exponential potential
#     """
#     return e * (((b * np.exp(a)) / (a - b)) * np.exp(-a * (x / c))
#                 - ((a * np.exp(b)) / (a - b)) * np.exp(-b * (x / c))) - f*(x/(g))**(-2)


def DEplus(x, a, b, c, e, f, g):
    """
    Double exponential potential
    """
    return e * (((b * np.exp(a)) / (a - b)) * np.exp(-a * (x / c))
                - ((a * np.exp(b)) / (a - b)) * np.exp(-b * (x / c))) - f * (c/x)**g





def get_LJs(i, Osig, Hsig, Oep, Hep, OH, HH, OO):
    a, b, c = 6, 2, 2

    OH_sig = (Osig + Hsig)
    HH_sig = (Hsig + Hsig)
    OO_sig = (Osig + Osig)

    OH_ep = (Oep * Hep) ** 0.5
    HH_ep = (Hep * Hep) ** 0.5
    OO_ep = (Oep * Oep) ** 0.5

    OH_en = np.sum(LJ(OH_sig, OH_ep, OH[i], a, b, c))
    HH_en = np.sum(LJ(HH_sig, HH_ep, HH[i], a, b, c))
    OO_en = np.sum(LJ(OO_sig, OO_ep, OO[i], a, b, c))

    return np.sum([OO_en, OH_en, HH_en])

def get_freeLJs(i, Osig, Hsig, Oep, Hep, a, b, c, OH, HH, OO):
    # print(i)
    OH_sig = (Osig + Hsig)
    HH_sig = (Hsig + Hsig)
    OO_sig = (Osig + Osig)

    OH_ep = (Oep * Hep) ** 0.5
    HH_ep = (Hep * Hep) ** 0.5
    OO_ep = (Oep * Oep) ** 0.5

    OH_en = np.sum(LJ(OH_sig, OH_ep, OH[i], a, b, c))
    HH_en = np.sum(LJ(HH_sig, HH_ep, HH[i], a, b, c))
    OO_en = np.sum(LJ(OO_sig, OO_ep, OO[i], a, b, c))

    return np.sum([OO_en, OH_en, HH_en])


def get_DEs(i, Osig, Hsig, Oep, Hep, OH, HH, OO):
    OH_sig = (Osig + Hsig)
    HH_sig = (Hsig + Hsig)
    OO_sig = (Osig + Osig)

    OH_ep = (Oep * Hep) ** 0.5
    HH_ep = (Hep * Hep) ** 0.5
    OO_ep = (Oep * Oep) ** 0.5

    a = 16.76632136164
    b = 4.426755081718

    OH_en = np.sum(DE(OH[i], a, b, OH_sig, OH_ep))
    HH_en = np.sum(DE(HH[i], a, b, HH_sig, HH_ep))
    OO_en = np.sum(DE(OO[i], a, b, OO_sig, OO_ep))

    return np.sum([OO_en, OH_en, HH_en])


def get_freeDEs(i, Osig, Hsig, Oep, Hep, a, b, OH, HH, OO):
    OH_sig = (Osig + Hsig)
    HH_sig = (Hsig + Hsig)
    OO_sig = (Osig + Osig)

    OH_ep = (Oep * Hep) ** 0.5
    HH_ep = (Hep * Hep) ** 0.5
    OO_ep = (Oep * Oep) ** 0.5

    OH_en = np.sum(DE(OH[i], a, b, OH_sig, OH_ep))
    HH_en = np.sum(DE(HH[i], a, b, HH_sig, HH_ep))
    OO_en = np.sum(DE(OO[i], a, b, OO_sig, OO_ep))

    return np.sum([OO_en, OH_en, HH_en])

def get_DEplus(i, Osig, Hsig, Oep, Hep, a, b, f, g, OH, HH, OO):
    OH_sig = (Osig + Hsig)
    HH_sig = (Hsig + Hsig)
    OO_sig = (Osig + Osig)

    OH_ep = (Oep * Hep) ** 0.5
    HH_ep = (Hep * Hep) ** 0.5
    OO_ep = (Oep * Oep) ** 0.5

    OH_en = np.sum(DEplus(OH[i], a, b,  OH_sig, OH_ep, f, g))
    HH_en = np.sum(DEplus(HH[i], a, b,  OH_sig, OH_ep, f, g))
    OO_en = np.sum(DEplus(OO[i], a, b,  OH_sig, OH_ep, f, g))

    return np.sum([OO_en, OH_en, HH_en])


class FF:
    def __init__(self, data, dists, func, nobj=4, elec="ele"):
        self.data = data
        self.df = data.copy()
        self.dists = dists
        self.OO = self.dists["OO"]
        self.OH = self.dists["OH"]
        self.HH = self.dists["HH"]
        self.func = func
        self.nobj = nobj
        self.elec = elec
        self.opt_parm = None
        self.opt_results = []

    def set_dists(self, dists):
        """Overwrite distances"""
        self.dists = dists
        self.OO = self.dists["OO"]
        self.OH = self.dists["OH"]
        self.HH = self.dists["HH"]

    def eval_func(self, x):
        Osig, Hsig, Oep, Hep = x[:4]
        tmp = self.data.copy()

        # function has 4 parameters
        if len(x) == 4:
            tmp["LJ"] = [self.func(i, Osig, Hsig, Oep, Hep, self.OH, self.HH, self.OO)
                         for i,_ in enumerate(tmp.index)]
        #  function has 6 parameters
        elif len(x) == 6:
            a, b = x[4:]
            tmp["LJ"] = [self.func(i, Osig, Hsig, Oep, Hep, a, b, self.OH, self.HH, self.OO)
                         for i,_ in enumerate(tmp.index)]
        #  function has 7 parameters
        elif len(x) == 7:
            a, b, c = x[4:]
            tmp["LJ"] = [self.func(i, Osig, Hsig, Oep, Hep, a, b, c, self.OH, self.HH, self.OO)
                         for i,_ in enumerate(tmp.index)]

        #  function has 7 parameters
        elif len(x) == 8:
            a, b, c, d = x[4:]
            tmp["LJ"] = [self.func(i, Osig, Hsig, Oep, Hep, a, b, c, d, self.OH, self.HH, self.OO)
                         for i,_ in enumerate(tmp.index)]

        return tmp

    def get_loss(self, x):
        tmp = self.eval_func(x)
        #  get squared error
        tmp["LJ_SE"] = (tmp["ETOT"] - (tmp[self.elec] + tmp["LJ"])) ** 2
        loss = tmp["LJ_SE"].mean()
        return loss

    def get_best_loss(self):
        results = pd.DataFrame(self.opt_results)
        best = results[results["fun"] == results["fun"].min()]
        return best
    
    def get_best_df(self):
        self.set_best_parm()
        tmp = self.eval_func(self.opt_parm)
        #  get squared error
        tmp["LJ_SE"] = (tmp["ETOT"] - (tmp[self.elec] + tmp["LJ"])) ** 2
        loss = tmp["LJ_SE"].mean()
        return tmp

    def set_best_parm(self):
        best = self.get_best_loss()
        self.opt_parm = best["x"].values[0]
        print("Set optimized parameters to FF object, "
              "use FF.opt_parm to get the optimized parameters")

    def eval_best_parm(self):
        self.set_best_parm()
        tmp = self.eval_func(self.opt_parm)
        print("Set optimized parameters to FF object, self.df[\"LJ\"] is updated.")
        self.df["LJ"] = tmp["LJ"]
        self.df["ETOT_LJ"] = tmp["LJ"] + self.df[self.elec]

    def fit_repeat(self, N, bounds=None, maxfev=10000, method="Nelder-Mead", quiet=False):
        for i in range(N):
            self.fit_func(None, bounds=bounds, maxfev=maxfev, method=method, quiet=quiet)
        self.get_best_loss()
        self.eval_best_parm()

    def fit_func(self, x0, bounds=None, maxfev=10000, method="Nelder-Mead", quiet=False):
        from scipy.optimize import minimize

        if x0 is None and bounds is not None:
            x0 = [np.random.uniform(low=a, high=b) for a, b in bounds]

        if not quiet:
            print(f"Optimizing LJ parameters...\n"
                  f"function: {self.func.__name__}\n"
                  f"bounds: {bounds}\n"
                  f"maxfev: {maxfev}\n"
                    f"initial guess: {x0}")

        res = minimize(self.get_loss, x0, method=method,
                       tol=1e-6,
                       bounds=bounds,
                       options={"maxfev": maxfev})

        if not quiet:
            print("final_loss_fn: ", res.fun)
            print(res)

        self.opt_parm = res.x
        self.opt_results.append(res)
        tmp = self.eval_func(self.opt_parm)

        if not quiet:
            print("Set optimized parameters to FF object, self.df[\"LJ\"] is updated.")

        self.df["LJ"] = tmp["LJ"]
        self.df["ETOT_LJ"] = tmp["LJ"] + self.df[self.elec]

        return res


# class EvalLoss:
#     def __init__(self, data, dists, func, elec="ele"):
#         self.data = data
#         self.dists = dists
#         self.OO = self.dists["OO"]
#         self.OH = self.dists["OH"]
#         self.HH = self.dists["HH"]
#         self.func = func
#         self.elec = elec
#
#     def get_loss(self, x):
#         Osig, Hsig, Oep, Hep = x[:4]
#         tmp = self.data.copy()
#
#         if len(x) == 4:
#             tmp["LJ"] = [self.func(i, Osig, Hsig, Oep, Hep, self.OH, self.HH, self.OO)
#                          for i in tmp.index]
#
#         elif len(x) == 6:
#             a, b = x[4:]
#             tmp["LJ"] = [self.func(i, Osig, Hsig, Oep, Hep, a, b, self.OH, self.HH, self.OO)
#                          for i in tmp.index]
#
#         elif len(x) == 7:
#             a, b, c = x[4:6]
#             tmp["LJ"] = [self.func(i, Osig, Hsig, Oep, Hep, a, b, self.OH, self.HH, self.OO)
#                          for i in tmp.index]
#
#         tmp["LJ_SE"] = (tmp["ETOT"] - (tmp[self.elec] + tmp["LJ"])) ** 2
#         loss = tmp["LJ_SE"].mean()
#         # print(loss)
#         return loss
#
#     def fit_func(self, x0, bounds=None, maxfev=10000):
#         from scipy.optimize import minimize
#         print("fitting...")
#         res = minimize(self.get_loss, x0, method="Nelder-Mead",
#                        tol=1e-6,
#                        bounds=bounds,
#                        options={"maxfev": maxfev})
#         return res

    # def get_loss_6_12(self, x):
    #     Osig, Hsig, Oep, Hep = x
    #     tmp = self.data.copy()
    #     tmp["LJ"] = [self.func(i, Osig, Hsig, Oep, Hep, 6, 2, 2, self.OH, self.HH, self.OO)
    #                  for i,_ in enumerate(tmp.index)]
    #     tmp["LJ_SE"] = (tmp["ETOT"] - (tmp[self.elec] + tmp["LJ"])) ** 2
    #     loss = tmp["LJ_SE"].mean()
    #     # print(loss)
    #     return loss
    #
    # def get_loss_7_14(self, x):
    #     Osig, Hsig, Oep, Hep = x
    #     tmp = self.data.copy()
    #     tmp["LJ"] = [self.func(i, Osig, Hsig, Oep, Hep, 7, 2, 2, self.OH, self.HH, self.OO)
    #                  for i in tmp.index]
    #     tmp["LJ_SE"] = (tmp["ETOT"] - (tmp[self.elec] + tmp["LJ"])) ** 2
    #     loss = tmp["LJ_SE"].mean()
    #     return loss
    #
    # def get_loss_freeDE(self, x):
    #     Osig, Hsig, Oep, Hep, a, b = x
    #     tmp = self.data.copy()
    #     tmp["LJ"] = [self.func(i, Osig, Hsig, Oep, Hep, a, b, self.OH, self.HH, self.OO)
    #                  for i in tmp.index]
    #     tmp["LJ_SE"] = (tmp["ETOT"] - (tmp[self.elec] + tmp["LJ"])) ** 2
    #     loss = tmp["LJ_SE"].mean()
    #     return loss
    #
    # def get_loss_freeLJ(self, x):
    #     Osig, Hsig, Oep, Hep, a, b, c = x
    #     tmp = self.data.copy()
    #     tmp["LJ"] = [self.func(i, Osig, Hsig, Oep, Hep, a, b, c, self.OH, self.HH, self.OO)
    #                  for i in tmp.index]
    #     tmp["LJ_SE"] = (tmp["ETOT"] - (tmp[self.elec] + tmp["LJ"])) ** 2
    #     loss = tmp["LJ_SE"].mean()
    #     return loss


def fit_func(func, x0, bounds=None):
    from scipy.optimize import minimize
    print("fitting...")
    res = minimize(func, x0, method="Nelder-Mead",
                   tol=1e-6,
                   bounds=bounds,
                   options={"maxfev": 10000})
    return res
