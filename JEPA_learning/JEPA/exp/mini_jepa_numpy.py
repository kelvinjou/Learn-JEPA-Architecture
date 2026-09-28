"""
A minimal JEPA in pure numpy with hand-written gradients, used to *measure*
representation collapse under three configurations:
   (1) shared    : target branch = online branch, gradients flow through both
   (2) stopgrad  : target branch = online weights, but stop-gradient applied
   (3) ema       : stop-gradient + EMA target encoder  (the I-JEPA recipe)
No attention: the context is the mean of visible patch embeddings.  That keeps
the gradients hand-derivable while preserving every ingredient that matters for
collapse (asymmetry, stop-grad, EMA, latent-space loss, predictor).
"""
import numpy as np, json, os
OUT = os.path.dirname(os.path.abspath(__file__))

# ---------------- synthetic dataset with real structure -------------------
IMG, PS = 24, 4;  G = IMG//PS;  NP = G*G;  PD = PS*PS*3
NCLS = 6
def make_data(n, seed):
    """6 classes = 3 shapes x 2 scales.  Colour is randomised independently of
    the label, so colour is not a shortcut; the label needs shape + extent."""
    g = np.random.default_rng(seed)
    X = np.zeros((n, IMG, IMG, 3), np.float32); y = g.integers(0, NCLS, n)
    for i in range(n):
        shape, big = int(y[i]) % 3, int(y[i]) // 3
        X[i] = 0.20*g.standard_normal((IMG,IMG,3)) + g.uniform(.1,.5)
        side = g.integers(13,18) if big else g.integers(7,11)
        h = w = int(side)
        r, c = g.integers(0, IMG-h+1), g.integers(0, IMG-w+1)
        col = g.uniform(.5,1.0,3)                       # colour independent of y
        if shape == 0:  X[i, r:r+h, c:c+w] = col                       # filled square
        elif shape == 1:                                                # ring
            X[i, r:r+h, c:c+w] = col
            X[i, r+2:r+h-2, c+2:c+w-2] = 0.20*g.standard_normal((max(h-4,0),max(w-4,0),3))+.3
        else:                                                           # triangle
            for k in range(h): X[i, r+k, c:c+max(w-k,1)] = col
        X[i] += 0.10*g.standard_normal((IMG,IMG,3))
    P = X.reshape(n, G, PS, G, PS, 3).transpose(0,1,3,2,4,5).reshape(n, NP, PD)
    return (P - P.mean())/(P.std()+1e-6), y
Xtr, ytr = make_data(4000, 0); Xte, yte = make_data(1500, 1)
NLAB = 600   # small labelled budget for the probe

# ---------------- masks: I-JEPA multi-block, scaled to a 4x4 grid ---------
def sample_masks(g, n_tgt=4, tgt_h=2, tgt_w=2, min_ctx=8):
    # mirrors the real sampler: resample until the context survives min_keep
    for _ in range(200):
        tg, grid = [], np.zeros((G,G), bool)
        for _ in range(n_tgt):
            r, c = g.integers(0, G-tgt_h+1), g.integers(0, G-tgt_w+1)
            m = np.zeros((G,G), bool); m[r:r+tgt_h, c:c+tgt_w] = True
            tg.append(np.flatnonzero(m.ravel())); grid |= m
        ctx = np.flatnonzero(~grid.ravel())            # context = complement
        if len(ctx) >= min_ctx: return tg, ctx
    return tg, ctx

# ---------------- model -------------------------------------------------
D, H, HP = 32, 64, 64
def init(seed=0):
    g = np.random.default_rng(seed); sc = lambda a,b: g.normal(0,(2/a)**.5,(a,b))
    return dict(W1=sc(PD,H), b1=np.zeros(H), W2=sc(H,D), b2=np.zeros(D),
                P1=sc(D+NP,HP), q1=np.zeros(HP), P2=sc(HP,D), q2=np.zeros(D))
def enc(th, x):                    # x:[...,PD] -> z:[...,D]
    a = x@th['W1'] + th['b1']; h = np.tanh(a); return h, a, h@th['W2'] + th['b2']
def ln(z, eps=1e-6):               # stateless LayerNorm over feature dim
    mu = z.mean(-1, keepdims=True); sd = np.sqrt(z.var(-1, keepdims=True)+eps)
    return (z-mu)/sd
def pred(th, c, pe):
    a = np.concatenate([c, pe], -1)@th['P1'] + th['q1']; h = np.tanh(a)
    return a, h, h@th['P2'] + th['q2']

def train(mode, steps=2500, bs=64, lr=3e-3, ema0=0.996, seed=0, log_every=50):
    th = init(seed); tt = {k: v.copy() for k, v in th.items()}
    g  = np.random.default_rng(100+seed); m_ada = {k: np.zeros_like(v) for k,v in th.items()}
    v_ada= {k: np.zeros_like(v) for k,v in th.items()}; log = []
    PEs = np.eye(NP)
    for t in range(steps):
        idx = g.integers(0, len(Xtr), bs); xb = Xtr[idx]
        tgm, ctx = sample_masks(g)
        # ---- target branch
        src = tt if mode == 'ema' else th
        h_t, a_t, z_t = enc(src, xb)                       # [bs,NP,D]
        T = ln(z_t)
        # ---- context branch (online)
        h_c, a_c, z_c = enc(th, xb[:, ctx])                # [bs,|ctx|,D]
        c = z_c.mean(1)                                    # [bs,D]
        gr = {k: np.zeros_like(v) for k, v in th.items()}; loss = 0.0; dc = np.zeros_like(c)
        for m in tgm:
            pe = PEs[m].sum(0)[None].repeat(bs, 0)         # multi-hot block position
            a_p, h_p, zh = pred(th, c, pe)                 # [bs,D]
            tgt = T[:, m].mean(1)                          # block-mean target
            r = zh - tgt; loss += float((r**2).sum(-1).mean())
            d = 2*r/(bs*len(tgm))                          # dL/dzh
            gr['P2'] += h_p.T@d; gr['q2'] += d.sum(0)
            dh = (d@th['P2'].T)*(1-h_p**2)
            inp = np.concatenate([c, pe], -1)
            gr['P1'] += inp.T@dh; gr['q1'] += dh.sum(0)
            dc += (dh@th['P1'].T)[:, :D]
            if mode == 'shared':                           # gradient into target too
                dt = -d/len(m)
                dz = np.zeros_like(z_t); dz[:, m] += dt[:, None]
                # backprop LN
                mu = z_t.mean(-1,keepdims=True); sd = np.sqrt(z_t.var(-1,keepdims=True)+1e-6)
                dz = (dz - dz.mean(-1,keepdims=True) - ((z_t-mu)/sd)*((dz*(z_t-mu)/sd).mean(-1,keepdims=True)))/sd
                gr['W2'] += np.einsum('bnh,bnd->hd', h_t, dz); gr['b2'] += dz.sum((0,1))
                dh2 = np.einsum('bnd,hd->bnh', dz, th['W2'])*(1-h_t**2)
                gr['W1'] += np.einsum('bnp,bnh->ph', xb, dh2); gr['b1'] += dh2.sum((0,1))
        dzc = (dc/len(ctx))[:, None].repeat(len(ctx), 1)
        gr['W2'] += np.einsum('bnh,bnd->hd', h_c, dzc); gr['b2'] += dzc.sum((0,1))
        dh1 = np.einsum('bnd,hd->bnh', dzc, th['W2'])*(1-h_c**2)
        gr['W1'] += np.einsum('bnp,bnh->ph', xb[:, ctx], dh1); gr['b1'] += dh1.sum((0,1))
        # adam
        for k in th:
            m_ada[k] = .9*m_ada[k] + .1*gr[k]; v_ada[k] = .999*v_ada[k] + .001*gr[k]**2
            th[k] -= lr*(m_ada[k]/(1-.9**(t+1)))/(np.sqrt(v_ada[k]/(1-.999**(t+1)))+1e-8)
        if mode == 'ema':
            mm = ema0 + (1.0-ema0)*t/steps
            for k in tt: tt[k] = mm*tt[k] + (1-mm)*th[k]
        elif mode == 'stopgrad':
            tt = {k: v.copy() for k, v in th.items()}
        if t % log_every == 0 or t == steps-1:
            log.append((t, loss/len(tgm), *diagnose(th if mode!='ema' else tt)[:2]))
    return th, tt, log

# ---------------- diagnostics -------------------------------------------
def embed(th, X, bs=500):
    out = []
    for i in range(0, len(X), bs):
        out.append(enc(th, X[i:i+bs])[2].mean(1))         # mean-pool patches
    return np.concatenate(out)
def diagnose(th):
    Z = embed(th, Xte)
    Zn = Z/(np.linalg.norm(Z,axis=1,keepdims=True)+1e-12)
    std = float(Zn.std(0).mean())                          # SimSiam monitor, target 1/sqrt(D)
    s = np.linalg.svd(Z - Z.mean(0), compute_uv=False)
    p = s/(np.abs(s).sum()+1e-12) + 1e-7
    rankme = float(np.exp(-(p*np.log(p)).sum()))           # RankMe, arXiv 2210.02885 eq (1)
    return std, rankme, s
def probe(th):                    # ridge linear probe on NLAB labelled images
    A, B = embed(th, Xtr[:NLAB]), embed(th, Xte)
    mu, sd = A.mean(0), A.std(0)+1e-8
    A = np.hstack([(A-mu)/sd, np.ones((len(A),1))]); B = np.hstack([(B-mu)/sd, np.ones((len(B),1))])
    Y = np.eye(NCLS)[ytr[:NLAB]]
    best = 0.0
    for lam in [1e-3,1e-2,1e-1,1,10]:
        W = np.linalg.solve(A.T@A + lam*np.eye(A.shape[1]), A.T@Y)
        best = max(best, float((np.argmax(B@W,1) == yte).mean()))
    return best

res = {'chance': round(1/NCLS,4), 'D': D, 'one_over_sqrt_D': round(1/np.sqrt(D), 4), 'n_patches': NP}
curves = {}
for mode in ['shared', 'stopgrad', 'ema']:
    th, tt, log = train(mode, seed=0)
    use = tt if mode == 'ema' else th
    std, rankme, s = diagnose(use)
    res[mode] = dict(final_loss=round(log[-1][1], 6), emb_std=round(std, 5),
                     rankme=round(rankme, 3), probe_acc=round(probe(use), 4),
                     top_sv=round(float(s[0]), 4), sv_ratio_1_to_8=round(float(s[0]/max(s[7],1e-12)), 2),
                     n_sv_below_1pct=int((s < 0.01*s[0]).sum()))
    curves[mode] = np.array([(a,b,c,d) for a,b,c,d in log])
    curves[mode+'_sv'] = s
res['random_init_baseline'] = dict(emb_std=round(diagnose(init(0))[0],5),
                                   rankme=round(diagnose(init(0))[1],3),
                                   probe_acc=round(probe(init(0)),4))
np.savez(os.path.join(OUT,'mini_jepa.npz'), **curves)
print(json.dumps(res, indent=1)); json.dump(res, open(os.path.join(OUT,'mini_jepa.json'),'w'), indent=1)
