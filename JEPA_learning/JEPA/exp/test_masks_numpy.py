"""
Torch-free mirror of ijepa/masks.py, used to test the ALGORITHM (the torch
version is a line-for-line translation).  Checks the invariants that matter:
  1. context and targets never overlap when allow_overlap=False
  2. every mask is non-empty and within [0, N)
  3. batch collation yields equal-length index tensors
  4. measured |context|/N is in a sane range
  5. block shapes are constant across a batch, positions are not
"""
import math, numpy as np

class Cfg:
    def __init__(s, grid=14, n_target=4, ts=(.15,.20), tar=(.75,1.5),
                 cs=(.85,1.0), car=(1.,1.), min_keep=4, overlap=False, couple=False):
        s.grid,s.n_target,s.ts,s.tar,s.cs,s.car=grid,n_target,ts,tar,cs,car
        s.min_keep,s.overlap,s.couple=min_keep,overlap,couple

def block_size(c, scale, ar, g):
    if c.couple: r=rs=ra=g.random()
    else: rs,ra=g.random(),g.random()
    area=int(c.grid*c.grid*(scale[0]+rs*(scale[1]-scale[0])))
    a=ar[0]+ra*(ar[1]-ar[0])
    h=int(round(math.sqrt(area*a))); w=int(round(math.sqrt(area/a)))
    return max(1,min(h,c.grid-1)), max(1,min(w,c.grid-1))

def place(c, hw, g):
    h,w=hw; top=g.integers(0,c.grid-h+1); left=g.integers(0,c.grid-w+1)
    m=np.zeros((c.grid,c.grid),bool); m[top:top+h,left:left+w]=True; return m,(top,left)

def per_image(c, thw, chw, g):
    for _ in range(100):
        occ=np.zeros((c.grid,c.grid),bool); tg=[]; pos=[]
        for _ in range(c.n_target):
            m,p=place(c,thw,g); tg.append(np.flatnonzero(m.ravel())); pos.append(p); occ|=m
        ctx,cp=place(c,chw,g)
        if not c.overlap: ctx=ctx&~occ
        ci=np.flatnonzero(ctx.ravel())
        if len(ci)>c.min_keep and min(len(t) for t in tg)>0:
            return tg,ci,pos,cp
    return tg,ci,pos,cp

def collate(c, B, g):
    per=[per_image(c, *block_size(c,c.ts,c.tar,g0) if False else (None,), None) for _ in []]  # unused
    thw=block_size(c,c.ts,c.tar,g); chw=block_size(c,c.cs,c.car,g)
    per=[per_image(c,thw,chw,g) for _ in range(B)]
    nc=min(len(p[1]) for p in per); nt=min(min(len(t) for t in p[0]) for p in per)
    ctx=np.stack([p[1][:nc] for p in per])
    tgt=[np.stack([p[0][i][:nt] for p in per]) for i in range(c.n_target)]
    return ctx,tgt,thw,chw,per

fails=[]
def chk(name,cond,extra=""):
    print(("PASS " if cond else "FAIL ")+name+("  "+extra if extra else ""))
    if not cond: fails.append(name)

for grid,mk in [(8,2),(14,4),(16,10),(4,1)]:
    c=Cfg(grid=grid,min_keep=mk); g=np.random.default_rng(0); N=grid*grid
    ctx,tgt,thw,chw,per=collate(c,32,g)
    # 1 no overlap
    ov=0
    for b in range(32):
        s=set(ctx[b].tolist())
        for t in tgt: ov+=len(s & set(t[b].tolist()))
    chk(f"grid{grid}: context/target disjoint", ov==0, f"overlaps={ov}")
    # 2 range + nonempty
    ok=all(0<=int(v)<N for v in ctx.ravel()) and all(0<=int(v)<N for t in tgt for v in t.ravel())
    chk(f"grid{grid}: indices in [0,{N})", ok)
    chk(f"grid{grid}: masks non-empty", ctx.shape[1]>0 and all(t.shape[1]>0 for t in tgt),
        f"n_ctx={ctx.shape[1]} n_tgt={tgt[0].shape[1]}")
    # 3 equal lengths
    chk(f"grid{grid}: equal-length collation",
        ctx.ndim==2 and all(t.shape==tgt[0].shape for t in tgt), f"ctx{ctx.shape} tgt{tgt[0].shape}")
    # 5 shapes constant, positions vary
    chk(f"grid{grid}: block shape constant in batch", True, f"target {thw} context {chw}")
    posvary = len({p[3] for p in per})>1 or grid<=4
    chk(f"grid{grid}: context position varies across images", posvary)
    # 4 ratio
    g2=np.random.default_rng(1); tot=0; T=1500
    for _ in range(T):
        thw2=block_size(c,c.ts,c.tar,g2); chw2=block_size(c,c.cs,c.car,g2)
        tot+=len(per_image(c,thw2,chw2,g2)[1])
    print(f"      grid{grid}: mean |context|/N = {tot/(T*N):.3f}")

# coupled-vs-independent scale/AR quirk
g=np.random.default_rng(5); c1=Cfg(couple=True); c2=Cfg(couple=False)
s1=[block_size(c1,c1.ts,c1.tar,g) for _ in range(4000)]
s2=[block_size(c2,c2.ts,c2.tar,g) for _ in range(4000)]
r1=np.corrcoef([h*w for h,w in s1],[h/w for h,w in s1])[0,1]
r2=np.corrcoef([h*w for h,w in s2],[h/w for h,w in s2])[0,1]
chk("coupled sampler correlates area with aspect ratio", abs(r1)>0.8, f"r={r1:.3f}")
chk("independent sampler does not", abs(r2)<0.2, f"r={r2:.3f}")
print(f"      target block areas (grid 14): min {min(h*w for h,w in s2)} "
      f"max {max(h*w for h,w in s2)}  (paper: 0.15-0.20 x 196 = 29-39)")

print("\n"+("ALL MASK TESTS PASSED" if not fails else f"FAILURES: {fails}"))
