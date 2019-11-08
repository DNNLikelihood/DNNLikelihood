from scipy.optimize import minimize
from toy_likelihood_data import *

##### Preprocess toy likelihood input data #####

n_bins=nbins
del(nbins)
n_cats = 3
tmp = np.array(list(map(lambda x: float("{:.1f}".format(x)),exp_background)))
exp_background = {'all': tmp, 'cat_1': tmp[0:30],'cat_2': tmp[30:60],'cat_3': tmp[60:90]}
tmp = np.array(signal)
exp_signal = {'all': tmp, 'cat_1': tmp[0:30],'cat_2': tmp[30:60],'cat_3': tmp[60:90]}
del(signal)
tmp = np.array(data)
observed = {'all': tmp, 'cat_1': tmp[0:30],'cat_2': tmp[30:60],'cat_3': tmp[60:90]}
del(data)
kpI, kmI = {}, {}
tmp = np.array(kpI_ISR)
kpI['ISR'] = {'all': tmp, 'cat_1': tmp[0:30],'cat_2': tmp[30:60],'cat_3': tmp[60:90]}
tmp = np.array(kmI_ISR)
kmI['ISR'] = {'all': tmp, 'cat_1': tmp[0:30],'cat_2': tmp[30:60],'cat_3': tmp[60:90]}
tmp = np.array(kpI_JES)
kpI['JES'] = {'all': tmp, 'cat_1': tmp[0:30],'cat_2': tmp[30:60],'cat_3': tmp[60:90]}
tmp = np.array(kmI_JES)
kmI['JES'] = {'all': tmp, 'cat_1': tmp[0:30],'cat_2': tmp[30:60],'cat_3': tmp[60:90]}
nuis_bin_keys = kmI.keys()
epsI_MCstat_tmp=epsI_MCstat
tmp = np.array(epsI_MCstat_tmp)
epsI_MCstat = {'all': tmp, 'cat_1': tmp[0:30],'cat_2': tmp[30:60],'cat_3': tmp[60:90]}
Kp, Km = {}, {}
Kp['ISR'], Km['ISR'] = np.array(Kp_ISR), np.array(Km_ISR)
Kp['JES'], Km['JES'] = np.array(Kp_JES), np.array(Km_JES)
Kp['LV'], Km['LV'] = np.array(Kp_LV), np.array(Km_LV)
Kp['CR'], Km['CR'] = np.array(Kp_CR), np.array(Km_CR)
nuis_cat_keys = Km.keys()
del(tmp)
exp_background_norm = np.array([exp_background['cat_1'].sum(),exp_background['cat_2'].sum(),exp_background['cat_3'].sum()])
tmp = exp_background['cat_1']/exp_background_norm[0]
tmp = np.append(tmp,exp_background['cat_2']/exp_background_norm[1])
fI0 = np.append(tmp,exp_background['cat_3']/exp_background_norm[2])
del(tmp)
parameters = np.insert(np.append(np.array(['nuis_mc_'+str(i) for i in range(90)]), np.array(['ISR_bin','JES_bin','ISR_cat','JES_cat','LV_cat','CR_cat']),axis=0),0,'mu')
labels = [r'$\mu$']
for pars in range(94):
    labels.append(r'$\delta_{'+str(pars+1)+'}$')

##### Likelihood and related quantities functions #####

def dN_MC(delta):
    return np.power(np.full(n_bins,1)+epsI_MCstat['all'],delta)

def pIj(delta_j):
    d = delta_j
    I = np.full(n_bins,1)
    res = {}
    for nuis in nuis_bin_keys:
        if abs(d) <= 1: 
            res[nuis] = 1/2*d*(d-1)*kmI[nuis]['all']-(d-1)*(d+1)*I+1/2*d*(d+1)*kpI[nuis]['all']
        elif d > 1:
            res[nuis] = (1/2*(3*kpI[nuis]['all']+kmI[nuis]['all'])-2*I)*d-1/2*(kpI[nuis]['all']+kmI[nuis]['all'])+2*I
        elif d < -1:
            res[nuis] = (2*I-1/2*(3*kmI[nuis]['all']+kpI[nuis]['all']))*d-1/2*(kpI[nuis]['all']+kmI[nuis]['all'])+2*I
    return res

def F(delta):
    d = {}
    nuis = nuis_bin_keys
    for i in range(len(nuis)):
        d[list(nuis)[i]] = delta[i]   
    FF = fI0*np.prod(list(map(lambda x: pIj(d[x])[x],nuis)),0)
    return np.sum(FF.reshape([3,30]),1)

def fI(delta):
    d = {}
    nuis = nuis_bin_keys
    for i in range(len(nuis)):
        d[list(nuis)[i]] = delta[i]
    FF = fI0*np.prod(list(map(lambda x: pIj(d[x])[x],nuis)),0)
    return np.array(np.append(FF[0:30]/F(delta)[0],[FF[30:60]/F(delta)[1],FF[60:90]/F(delta)[2]]))

def dN(delta_j,nuis):
    d = delta_j
    I = np.full(n_cats,1)
    if d >= 0:
        res = np.power((I+Kp[nuis]),d)
    elif d < 0:
        res = np.power((I+Km[nuis]),-d)
    return res

def nbI(delta):
    d_MC = delta[:90]
    dI, d = {}, {}
    for i in range(len(nuis_bin_keys)):
        dI[list(nuis_bin_keys)[i]] = delta[i+90]
    for i in range(len(nuis_cat_keys)):
        d[list(nuis_cat_keys)[i]] = delta[i+90]
    prod = fI(list(dI.values()))*dN_MC(d_MC)
    cat1 = exp_background_norm[0]*np.prod(np.array([dN(d[nuis],nuis)[0] for nuis in nuis_cat_keys]))*prod[0:30]
    cat2 = exp_background_norm[1]*np.prod(np.array([dN(d[nuis],nuis)[1] for nuis in nuis_cat_keys]))*prod[30:60]
    cat3 = exp_background_norm[2]*np.prod(np.array([dN(d[nuis],nuis)[2] for nuis in nuis_cat_keys]))*prod[60:90]
    return np.array(np.append(cat1,[cat2,cat3]))

def nsI(mu):
    return mu*exp_signal['all']

def expected(pars):
    mu = pars[0]
    delta = pars[1:]
    return np.array(nsI(mu)+nbI(delta))

def lik(pars):
    obs = observed['all']
    exp = expected(pars)
    fact = np.array(list(map(lambda x: np.float64(np.math.factorial(x)),obs)))
    res = np.exp(-exp)*(np.power(exp,obs))/fact
    res = np.prod(res)
    if np.isinf(res):
        return 0
    return res

def loglik(pars):
    obs = observed['all']
    exp = expected(pars)
    logfact = np.array(list(map(lambda x: np.math.lgamma(x+1), obs)))
    res = -1*logfact+obs*np.log(exp)-exp
    res = np.sum(res)
    if np.isnan(res):
        return -np.inf
    return res

def logprior(pars):
    mu = pars[0]
    delta = pars[1:]
    nuis_prior = -1/2*np.sum(delta**2+np.full(len(delta),np.log(2*np.pi)))
    if (mu > -1) and (mu < 5):
        return nuis_prior-np.log(5)
    else:
        return -np.inf+nuis_prior

def logprob(pars):
    lprior=logprior(pars)
    if np.isinf(lprior):
        return lprior
    else:
        llik=loglik(pars)
        return llik+lprior

def logprob_only_prior(pars):
    lprior=logprior(pars)
    return lprior

def minus_logprob(sample):
    return -logprob(sample)

def minus_logprob_delta(delta,mu):
    pars = np.concatenate((np.array([mu]),delta))
    return -logprob(pars)

def tmu(mu):
    minimum_logprob = minimize(minus_logprob,np.full(95,0),method='Powell')['x']
    L_muhat_deltahat = -minus_logprob(minimum_logprob)
    minimum_logprob_delta = np.concatenate((np.array([mu]),minimize(lambda x: minus_logprob_delta(x,mu), np.full(94,0),method='Powell')['x']))
    L_mu_deltahat = -minus_logprob(minimum_logprob_delta)
    return np.array([mu,L_muhat_deltahat,L_mu_deltahat,-2*(L_mu_deltahat-L_muhat_deltahat)])  

def tmu_sample(mu, samples, logprobs, binwidth):
    maxL = np.amax(logprobs)
    maxLprof = np.amax(
        logprobs[[i > mu-binwidth/2 and i < mu+binwidth/2 for i in samples[:, 0]]])
    return np.array([binwidth,mu,maxL,maxLprof,-2*(maxLprof-maxL)])

##### Pseudo-experiments for frequentist analysis related functions #####

def loglik_toys(pars, obs):
    exp = expected(pars)
    logfact = np.array(list(map(lambda x: np.math.lgamma(x+1), obs)))
    res = -1*logfact+obs*np.log(exp)-exp
    res = np.sum(res)
    if np.isnan(res):
        return -np.inf
    return res

def logprob_toys(pars, obs):
    lprior = logprior(pars)
    if np.isinf(lprior):
        return lprior
    else:
        llik = loglik_toys(pars, obs)
        return llik+lprior

def minus_logprob_toys(sample, obs):
    return -logprob_toys(sample, obs)

# These functions are necessary for use in parallel Pool, where lambda function cannot be called explicitly

def custom_func(t):
    return -minimize(lambda x: minus_logprob_toys(x, t), np.full(95, 0), method='Powell')['fun']

def custom_func00(t):
    return -minimize(lambda x: minus_logprob_toys(np.concatenate((np.array([0]), x)), t), np.full(94, 0), method='Powell')['fun']

def custom_func01(t):
    return -minimize(lambda x: minus_logprob_toys(np.concatenate((np.array([0.1]), x)), t), np.full(94, 0), method='Powell')['fun']

def custom_func02(t):
    return -minimize(lambda x: minus_logprob_toys(np.concatenate((np.array([0.2]), x)), t), np.full(94, 0), method='Powell')['fun']

def custom_func03(t):
    return -minimize(lambda x: minus_logprob_toys(np.concatenate((np.array([0.3]), x)), t), np.full(94, 0), method='Powell')['fun']

def custom_func04(t):
    return -minimize(lambda x: minus_logprob_toys(np.concatenate((np.array([0.4]), x)), t), np.full(94, 0), method='Powell')['fun']

def custom_func05(t):
    return -minimize(lambda x: minus_logprob_toys(np.concatenate((np.array([0.5]), x)), t), np.full(94, 0), method='Powell')['fun']

def custom_func06(t):
    return -minimize(lambda x: minus_logprob_toys(np.concatenate((np.array([0.6]), x)), t), np.full(94, 0), method='Powell')['fun']

def custom_func07(t):
    return -minimize(lambda x: minus_logprob_toys(np.concatenate((np.array([0.7]), x)), t), np.full(94, 0), method='Powell')['fun']

def custom_func08(t):
    return -minimize(lambda x: minus_logprob_toys(np.concatenate((np.array([0.8]), x)), t), np.full(94, 0), method='Powell')['fun']

def custom_func09(t):
    return -minimize(lambda x: minus_logprob_toys(np.concatenate((np.array([0.9]), x)), t), np.full(94, 0), method='Powell')['fun']

def custom_func10(t):
    return -minimize(lambda x: minus_logprob_toys(np.concatenate((np.array([1]), x)), t), np.full(94, 0), method='Powell')['fun']
