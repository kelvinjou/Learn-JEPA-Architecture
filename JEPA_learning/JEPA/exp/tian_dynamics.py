"""
Numerical study of the linear-JEPA gradient flow of
Tian, Chen & Ganguli (arXiv:2102.06810), eqs. (7), (11)-(13), (16)-(17).
Everything here is integrated from the paper's own equations; no fitting.
"""
import numpy as np, json, os
OUT = os.path.dirname(os.path.abspath(__file__)); rng = np.random.default_rng(0)
SIG2, ALPHA_P, DT = 0.5, 1.0, 1e-3

# ============ A. per-mode scalar system, eqs (11)-(13) ====================
def step(p, s, tau, eta, beta, ema):
    d = tau - (1+SIG2)*p
    dp = ALPHA_P*s*d - eta*p
    ds = 2*p*s*d - 2*eta*s
    dtau = (beta*(1-tau) - tau*ds/(2*max(s,1e-14))) if ema else 0.0
    return dp, ds, dtau

def run(p0, s0, eta=0.0, beta=0.0, ema=False, T=200000, tau0=None):
    p, s = float(p0), float(s0)
    tau = (0.0 if ema else 1.0) if tau0 is None else float(tau0)
    hist = np.empty((T,5))
    for t in range(T):
        D  = p*(tau-(1+SIG2)*p) - eta          # Delta_j, eq (16)
        UB = 0.5*(ALPHA_P*(1+SIG2)*s + eta)    # rhs of eq (17)
        hist[t] = (t*DT, p, s, tau, float(D < UB))
        dp, ds, dtau = step(p, s, tau, eta, beta, ema)
        p += DT*dp; s = max(s + DT*ds, 0.0); tau = min(max(tau + DT*dtau, 0.0), 1.0)
        if not np.isfinite(p): break
    return hist[:t+1]

def fp(tau, eta):
    disc = tau*tau - 4*eta*(1+SIG2)
    if disc < 0: return None
    r = np.sqrt(disc); return ((tau-r)/(2*(1+SIG2)), (tau+r)/(2*(1+SIG2)))

R = {'sigma2': SIG2, 'alpha_p': ALPHA_P, 'dt': DT}
R['eta_crit_tau1'] = round(1.0/(4*(1+SIG2)), 6)          # tau^2/(4(1+sig2))

# A1  eta = 0 : collapsed branch is a saddle; escape iff s0 > p0^2/alpha_p
A1 = []
for p0, s0 in [(0.20,0.010),(0.20,0.040),(0.20,0.041),(0.20,0.100),(0.40,0.150)]:
    h = run(p0, s0, eta=0.0, T=120000)
    A1.append(dict(p0=p0, s0=s0, s0_minus_parabola=round(s0-p0**2/ALPHA_P,4),
                   p_inf=round(float(h[-1,1]),4), s_inf=round(float(h[-1,2]),4),
                   c_j=round(p0**2/ALPHA_P - s0, 4)))
R['A1_eta0_parabola'] = A1
R['A1_note'] = 'p_inf should approach tau/(1+sig2)=%.4f from above the parabola' % (1/(1+SIG2))

# A2  eta > 0 : flow lands on s = p^2/alpha_p, then eq (16) gives 3 fixed pts
eta = 0.05; pm, pp = fp(1.0, eta)
R['A2_eta'] = eta; R['A2_p_minus'] = round(pm,5); R['A2_p_plus'] = round(pp,5)
A2 = []
for p0 in [0.020, 0.045, 0.0540, 0.0548, 0.070, 0.200, 0.500]:
    h = run(p0, p0**2/ALPHA_P, eta=eta, T=400000)      # start ON the parabola
    A2.append(dict(p0=p0, predicted_collapse=bool(p0 < pm),
                   p_inf=round(float(h[-1,1]),5), s_inf=round(float(h[-1,2]),6),
                   observed_collapse=bool(h[-1,2] < 1e-5)))
R['A2_basin'] = A2
R['A2_agreement'] = all(d['predicted_collapse']==d['observed_collapse'] for d in A2)

# A3  condition (17) along a whole trajectory: tau==1 vs EMA
A3 = {}
for name, kw in [('tau_fixed_1', dict(ema=False)),
                 ('ema_beta_0.5', dict(ema=True, beta=0.5)),
                 ('ema_beta_2.0', dict(ema=True, beta=2.0))]:
    h = run(0.05, 0.002, eta=0.0, T=200000, **kw)
    A3[name] = dict(frac_steps_eq17_holds = round(float(h[:,4].mean()),4),
                    steps_until_eq17_first_holds = int(np.argmax(h[:,4]>0)),
                    p_inf=round(float(h[-1,1]),4), s_inf=round(float(h[-1,2]),4),
                    tau_inf=round(float(h[-1,3]),4))
R['A3_eq17'] = A3

# ============ B. matrix system, eqs (7)-(10): eigenspace alignment ========
def align_run(d=6, eta=0.02, tau=0.8, T=40000, seed=1):
    g = np.random.default_rng(seed)
    Wp = g.normal(0, .3, (d,d)); Wp = 0.5*(Wp+Wp.T)          # A3: symmetric
    F  = g.normal(0, .3, (d,d)); F  = F@F.T/d + 0.05*np.eye(d)  # PSD
    out = np.empty((T,3))
    for t in range(T):
        comm = F@Wp - Wp@F
        out[t] = (t*DT, np.linalg.norm(comm,'fro'), np.linalg.norm(F,'fro'))
        acom = lambda A,B: A@B + B@A                          # {A,B}
        dWp = -0.5*ALPHA_P*(1+SIG2)*acom(Wp,F) + ALPHA_P*tau*F - eta*Wp
        dF  = -(1+SIG2)*acom(Wp@Wp,F) + tau*acom(Wp,F) - 2*eta*F
        Wp = Wp + DT*dWp; F = F + DT*dF
        Wp = 0.5*(Wp+Wp.T); F = 0.5*(F+F.T)
    return out, Wp, F
al, Wp_f, F_f = align_run()
R['B_alignment'] = dict(
    commutator_t0   = round(float(al[0,1]),5),
    commutator_t25  = round(float(al[len(al)//4,1]),8),
    commutator_tend = round(float(al[-1,1]),10),
    decay_factor    = float(f"{al[0,1]/max(al[-1,1],1e-300):.3e}"),
    # simultaneous diagonalisability check
    max_offdiag_of_Wp_in_F_eigenbasis = float(f"{np.abs((lambda V: V.T@Wp_f@V)(np.linalg.eigh(F_f)[1]) - np.diag(np.diag((lambda V: V.T@Wp_f@V)(np.linalg.eigh(F_f)[1])))).max():.3e}"),
)
def _sub(a, n=4000):
    import numpy as _np
    return a[::max(1, len(a)//n)]
np.savez_compressed(os.path.join(OUT,'tian_traj.npz'),
    esc   = _sub(run(0.20, 0.041, eta=0.0,  T=120000)),
    slow  = _sub(run(0.20, 0.010, eta=0.0,  T=120000)),
    wd_esc= _sub(run(0.070, 0.070**2, eta=0.05, T=400000)),
    wd_col= _sub(run(0.020, 0.020**2, eta=0.05, T=400000)),
    ema   = _sub(run(0.05, 0.002, eta=0.0, ema=True, beta=0.5, T=200000)),
    fixed = _sub(run(0.05, 0.002, eta=0.0, ema=False, T=200000)),
    align = al)
print(json.dumps(R, indent=1)); json.dump(R, open(os.path.join(OUT,'tian_results.json'),'w'), indent=1)
