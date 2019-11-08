from scipy.optimize import minimize
import os
from timeit import default_timer as timer
import celerite
from celerite import terms

import h5py
import matplotlib.pyplot as plt
import numpy as np

import emcee

def import_sampler(filename, name, read_only=False):
    start = timer()
    statinfo = os.stat(filename)
    res = emcee.backends.HDFBackend(filename, name=name, read_only=read_only)
    end = timer()
    print('Backend',filename,'opened in',end-start,'seconds.\nFile size is',statinfo.st_size,'.')
    return res

##### Functions from the emcee documentation #####

def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i

def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2*n)
    acf = np.fft.ifft(f * np.conjugate(f))[:len(x)].real
    acf /= 4*n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf

# Automated windowing procedure following Sokal (1989)
def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

# Following the suggestion from Goodman & Weare (2010)
def autocorr_gw2010(y, c=5.0):
    f = autocorr_func_1d(np.mean(y, axis=0))
    taus = 2.0*np.cumsum(f)-1.0
    window = auto_window(taus, c)
    return taus[window]

def autocorr_new(y, c=5.0):
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0*np.cumsum(f)-1.0
    window = auto_window(taus, c)
    return taus[window]

def autocorr_ml(y, thin=1, c=5.0):
    # Compute the initial estimate of tau using the standard method
    init = autocorr_new(y, c=c)
    z = y[:, ::thin]
    N = z.shape[1]

    # Build the GP model
    tau = max(1.0, init / thin)
    kernel = terms.RealTerm(
        np.log(0.9 * np.var(z)), -np.log(tau), bounds=[(-5.0, 5.0), (-np.log(N), 0.0)]
    )
    kernel += terms.RealTerm(
        np.log(0.1 * np.var(z)),
        -np.log(0.5 * tau),
        bounds=[(-5.0, 5.0), (-np.log(N), 0.0)],
    )
    gp = celerite.GP(kernel, mean=np.mean(z))
    gp.compute(np.arange(z.shape[1]))

    # Define the objective
    def nll(p):
        # Update the GP model
        gp.set_parameter_vector(p)

        # Loop over the chains and compute likelihoods
        v, g = zip(*(gp.grad_log_likelihood(z0, quiet=True) for z0 in z))

        # Combine the datasets
        return -np.sum(v), -np.sum(g, axis=0)

    # Optimize the model
    p0 = gp.get_parameter_vector()
    bounds = gp.get_parameter_bounds()
    soln = minimize(nll, p0, jac=True, bounds=bounds)
    gp.set_parameter_vector(soln.x)

    # Compute the maximum likelihood tau
    a, c = kernel.coefficients[:2]
    tau = thin * 2 * np.sum(a / c) / np.sum(a)
    return tau

def gelman_rubin(chain):
    si2 = np.var(chain, axis=0, ddof=1)
    W = np.mean(si2, axis=0)
    ximean = np.mean(chain, axis=0)
    xmean = np.mean(ximean, axis=0)
    n = chain.shape[0]
    m = chain.shape[1]
    B = n / (m - 1) * np.sum((ximean - xmean)**2, axis=0)
    sigmahat2 = (n - 1) / n * W + 1 / n * B
    Vhat = sigmahat2+B/m/n
    varVhat = ((n-1)/n)**2 * 1/m * np.var(si2, axis=0)+((m+1)/(m*n))**2 * 2/(m-1) * B**2 + 2*((m+1)*(n-1)/(m*(n**2)))*n/m *(np.cov(si2,ximean**2)[0,1]-2*xmean*np.cov(si2,ximean)[0,1])
    df = (2*Vhat**2) / varVhat
    # Rh = np.sqrt((Vhat / W)*df/(df-2)) #incorrect Gelman-Rubin df
    Rh = np.sqrt((Vhat / W)*(df+3)/(df+1)) #correct Brooks-Gelman df
    return [Rh, Vhat, W]

# "DFM 2017: https://dfm.io/posts/autocorr/"
def show_dist_and_autocorr(chains, pars, labels, save=False, methods=["G&W 2010", "DFM 2017", "DFM 2017: ML"]):
    for par in pars:
        chain = chains[:, :, par].T
        counts, bins = np.histogram(chain.flatten(), 100)
        integral = counts.sum()
        plt.grid(linestyle="--", dashes=(5, 5))
        plt.step(bins[:-1], counts/integral, where='post')
        plt.xlabel(r"$%s$" % (labels[par].replace('$', '')))
        plt.ylabel(r"$p(%s)$" % (labels[par].replace('$', '')))
        plt.tight_layout()
        if type(save) == bool:
            if save:
                title_time = timer()
                plt.savefig('../paper/figs/00/figure_autocorr_dist_' +
                            str(par)+'_'+str(title_time)+'.pdf')
                print('Saved', '../paper/figs/00/figure_autocorr_dist_' +
                      str(par)+'_'+str(title_time)+'.pdf')
        elif type(save) == str:
            save = save.replace('.pdf', '_')
            title_time = timer()
            plt.savefig(save+'dist_'+str(par)+'_'+str(title_time)+'.pdf')
            print('Saved', save+'dist_'+str(par)+'_'+str(title_time)+'.pdf')
        plt.show()
        plt.close()

        N = np.exp(np.linspace(np.log(100), np.log(
            chain.shape[1]), 10)).astype(int)
        # GW10 method
        if "G&W 2010" in methods:
            gw2010 = np.empty(len(N))
        # New method
        if "DFM 2017" in methods:
            new = np.empty(len(N))
        # Approx method (Maximum Likelihood)
        if "DFM 2017: ML" in methods:
            new = np.empty(len(N))
            ml = np.empty(len(N))
            ml[:] = np.nan

        for i, n in enumerate(N):
            # GW10 method
            if "G&W 2010" in methods:
                gw2010[i] = autocorr_gw2010(chain[:, :n])
            # New method
            if "DFM 2017" in methods:
                new[i] = autocorr_new(chain[:, :n])
            # Approx method (Maximum Likelihood)
        if "DFM 2017: ML" in methods:
            for i, n in enumerate(N[1:-1]):
                k = i + 1
                thin = max(1, int(0.05 * new[k]))
                ml[k] = autocorr_ml(chain[:, :n], thin=thin)
        # Plot the comparisons
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.grid(linestyle="--", dashes=(5, 5))
        plt.plot(N, N / 50.0, "--k", label=r"$\tau = S/50$")
        #plt.plot(N, N / 100.0, "--k", label=r"$\tau = S/100$")
        # GW10 method
        if "G&W 2010" in methods:
            plt.loglog(N, gw2010, "o-", label=r"G\&W 2010")
        # New method
        if "DFM 2017" in methods:
            plt.loglog(N, new, "o-", label="DFM 2017")
        # Approx method (Maximum Likelihood)
        if "DFM 2017: ML" in methods:
            plt.loglog(N, ml, "o-", label="DFM 2017: ML")
        ylim = plt.gca().get_ylim()
        plt.ylim(ylim)
        plt.xlabel(r"number of steps, $S$")
        plt.ylabel(r"$\tau_{%s}$ estimates" % (labels[par].replace('$', '')))
        plt.legend()
        plt.tight_layout()
        if type(save) == bool:
            if save:
                title_time = timer()
                plt.savefig('../paper/figs/00/figure_autocorr_' +
                            str(par)+'_'+str(title_time)+'.pdf')
                print('Saved', '../paper/figs/00/figure_autocorr_' +
                      str(par)+'_'+str(title_time)+'.pdf')
        elif type(save) == str:
            save = save.replace('.pdf', '_')
            title_time = timer()
            plt.savefig(save+str(par)+'_'+str(title_time)+'.pdf')
            print('Saved', save+str(par)+'_'+str(title_time)+'.pdf')
        plt.show()
        plt.close
