import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams.update({'font.size':8,'font.family':'serif','axes.grid':True,
    'grid.alpha':.25,'grid.linewidth':.5,'axes.linewidth':.6,'figure.dpi':160,
    'legend.frameon':False,'axes.spines.top':False,'axes.spines.right':False})
E='/sessions/adoring-happy-pascal/mnt/outputs/exp/'; F='/sessions/adoring-happy-pascal/mnt/outputs/fig/'
C={'shared':'#c0392b','stopgrad':'#2980b9','ema':'#27ae60'}
LBL={'shared':'no stop-grad (shared branch)','stopgrad':'stop-grad only','ema':'stop-grad + EMA (I-JEPA)'}

# ---- Fig 1: collapse curves + spectra ----------------------------------
d=np.load(E+'mini_jepa.npz'); fig,ax=plt.subplots(1,3,figsize=(7.2,2.05))
ST={'shared':'-','stopgrad':(0,(4,2)),'ema':'-'}
for m in ['shared','ema','stopgrad']:
    c=d[m]; ax[0].plot(c[:,0],c[:,1],color=C[m],lw=1.2,ls=ST[m],label=LBL[m])
    ax[1].plot(c[:,0],c[:,2],color=C[m],lw=1.2,ls=ST[m])
ax[0].set_yscale('log'); ax[0].set_xlabel('training step'); ax[0].set_ylabel('JEPA loss')
ax[0].set_title('(a) the loss lies',fontsize=8.5,loc='left')
ax[1].axhline(1/np.sqrt(32),ls=':',c='k',lw=.8)
ax[1].text(1250,1/np.sqrt(32)*1.09,r'healthy: $1/\sqrt{D}=0.177$',fontsize=6.4,ha='center')
ax[1].set_yscale('log'); ax[1].set_xlabel('training step')
ax[1].set_ylabel(r'std of $z/\|z\|_2$')
ax[1].set_title('(b) the std tells the truth',fontsize=8.5,loc='left')
for m in ['shared','ema','stopgrad']:
    s=d[m+'_sv']; ax[2].plot(np.arange(1,len(s)+1),s/s[0],color=C[m],lw=1.2,ls=ST[m],marker='o',ms=1.8)
ax[2].set_yscale('log'); ax[2].set_xlabel('singular value index $k$')
ax[2].set_ylabel(r'$\sigma_k/\sigma_1$'); ax[2].set_title('(c) spectrum of $\\mathrm{Cov}(z)$',fontsize=8.5,loc='left')
h_,l_=ax[0].get_legend_handles_labels()
fig.legend([h_[0],h_[2],h_[1]],[l_[0],l_[2],l_[1]],loc='lower center',ncol=3,bbox_to_anchor=(.5,-.05),fontsize=7)
fig.tight_layout(rect=[0,.07,1,1]); fig.savefig(F+'collapse.pdf',bbox_inches='tight')

# ---- Fig 2: Tian phase plane + bifurcation -----------------------------
t=np.load(E+'tian_traj.npz'); SIG2,AP=0.5,1.0
fig,ax=plt.subplots(1,2,figsize=(7.2,2.35))
pp=np.linspace(0,.75,300)
ax[0].plot(pp,pp**2/AP,'k--',lw=.9,label=r'invariant parabola $s=p^2/\alpha_p$')
for k,c,l in [('wd_esc','#27ae60',r'$p_0=0.070>p^*_-$: escapes'),
              ('wd_col','#c0392b',r'$p_0=0.020<p^*_-$: collapses')]:
    h=t[k]; hh=h[np.maximum(h[:,2],0)>0]
    ax[0].plot(hh[:,1],np.maximum(hh[:,2],1e-7),color=c,lw=1.3,label=l)
    ax[0].plot(h[0,1],h[0,2],'o',color=c,ms=3.4,zorder=5)
    ax[0].annotate('',xy=(hh[-1,1],max(hh[-1,2],1.3e-6)),xytext=(hh[-len(hh)//12,1],max(hh[-len(hh)//12,2],2e-6)),
                   arrowprops=dict(arrowstyle='-|>',color=c,lw=1.0,mutation_scale=7))
pm,ppl=0.05445,0.61222
ax[0].axvspan(0,pm,color='#c0392b',alpha=.07)
ax[0].axvline(pm,color='#c0392b',lw=.7,ls=':'); ax[0].axvline(ppl,color='#27ae60',lw=.7,ls=':')
ax[0].text(pm*1.9,2e-6,'collapse\nbasin',fontsize=6.0,ha='left',color='#c0392b')
ax[0].annotate(r'$p^*_-$',(pm*1.15,.35),fontsize=7,color='#c0392b',ha='left')
ax[0].annotate(r'$p^*_+$',(ppl*.97,3e-6),fontsize=7,color='#27ae60',ha='right')
ax[0].set_yscale('log'); ax[0].set_xlim(0,.72); ax[0].set_ylim(1e-6,1.2)
ax[0].set_xlabel(r'predictor eigenvalue $p_j$'); ax[0].set_ylabel(r'online-feature eigenvalue $s_j$')
ax[0].set_title(r'(a) phase plane, $\eta=0.05$, $\tau\!\equiv\!1$',fontsize=8.5,loc='left')
ax[0].legend(fontsize=6.1,loc='lower right')
et=np.linspace(0,.24,600); ec=1/(4*(1+SIG2)); disc=1-4*et*(1+SIG2)
pmv=np.where(disc>=0,(1-np.sqrt(np.clip(disc,0,None)))/(2*(1+SIG2)),np.nan)
ppv=np.where(disc>=0,(1+np.sqrt(np.clip(disc,0,None)))/(2*(1+SIG2)),np.nan)
ax[1].plot(et,ppv,color='#27ae60',lw=1.4,label=r'$p^*_+$ (stable, useful)')
ax[1].plot(et,pmv,color='#c0392b',lw=1.4,ls='--',label=r'$p^*_-$ (unstable)')
ax[1].axhline(0,color='#7f8c8d',lw=1.4,label=r'$p^*_0=0$ (stable, collapsed)')
ax[1].axvline(ec,color='k',lw=.8,ls=':')
ax[1].text(ec,.60,r'  $\eta_{\rm crit}=\tau^2/[4(1+\sigma^2)]=1/6$',fontsize=6.6)
ax[1].axvspan(ec,.24,color='#c0392b',alpha=.07)
ax[1].text((ec+.24)/2,.30,'collapse\nunavoidable',fontsize=6.2,ha='center',color='#c0392b')
ax[1].set_xlim(0,.24); ax[1].set_ylim(-.04,.72)
ax[1].set_xlabel(r'weight decay $\eta$'); ax[1].set_ylabel(r'fixed points $p_j^*$')
ax[1].set_title(r'(b) saddle-node bifurcation',fontsize=8.5,loc='left'); ax[1].legend(fontsize=6.1)
fig.tight_layout(); fig.savefig(F+'tian.pdf',bbox_inches='tight')

# ---- Fig 3: eigenspace alignment + eq.(17) -----------------------------
fig,ax=plt.subplots(1,2,figsize=(7.2,2.1))
a=t['align']; ax[0].semilogy(a[:,0],a[:,1],color='#8e44ad',lw=1.3)
ax[0].set_xlabel(r'flow time $t$'); ax[0].set_ylabel(r'$\|[F,W_p]\|_F$')
ax[0].set_title(r'(a) Thm 3: $W_p$ aligns with $F$',fontsize=8.5,loc='left')
ax[0].text(.97,.92,r'$300\times$ decay, $d=6$',transform=ax[0].transAxes,ha='right',fontsize=6.6)
for k,c,l in [('fixed','#c0392b',r'$\tau\equiv1$ (no EMA)'),('ema','#27ae60',r'EMA, $\tau(0)=0$')]:
    h=t[k]; D=h[:,1]*(h[:,3]-(1+SIG2)*h[:,1]); UB=.5*(AP*(1+SIG2)*h[:,2])
    ax[1].plot(h[:,0],D-UB,color=c,lw=1.2,label=l)
ax[1].axhline(0,color='k',lw=.7)
ax[1].fill_between([0,200],0,.05,color='#c0392b',alpha=.07)
ax[1].text(100,.017,'eq. (17) violated: alignment not guaranteed',fontsize=6.2,ha='center',color='#c0392b')
ax[1].set_xlim(0,200); ax[1].set_ylim(-.03,.05)
ax[1].set_xlabel(r'flow time $t$'); ax[1].set_ylabel(r'$\Delta_j-\frac{1}{2}[\alpha_p(1+\sigma^2)s_j+\eta]$')
ax[1].set_title(r'(b) EMA as automatic curriculum',fontsize=8.5,loc='left'); ax[1].legend(fontsize=6.6)
fig.tight_layout(); fig.savefig(F+'align.pdf',bbox_inches='tight')

# ---- Fig 4: I-JEPA multi-block masking on a 14x14 grid -----------------
def blocks(g,G=14,n=4,sc=(.15,.2),ar=(.75,1.5)):
    tg=[];occ=np.zeros((G,G),bool)
    for _ in range(n):
        r=g.random(); mk=int(G*G*(sc[0]+r*(sc[1]-sc[0]))); a=ar[0]+r*(ar[1]-ar[0])
        h=min(int(round(np.sqrt(mk*a))),G-1); w=min(int(round(np.sqrt(mk/a))),G-1)
        i,j=g.integers(0,G-h+1),g.integers(0,G-w+1)
        m=np.zeros((G,G),bool); m[i:i+h,j:j+w]=True; tg.append(m); occ|=m
    r=g.random(); mk=int(G*G*(.85+r*.15)); h=w=min(int(round(np.sqrt(mk))),G)
    i,j=g.integers(0,G-h+1),g.integers(0,G-w+1)
    cb=np.zeros((G,G),bool); cb[i:i+h,j:j+w]=True
    return tg,cb,cb&~occ
fig,axs=plt.subplots(1,4,figsize=(7.2,2.0)); g=np.random.default_rng(7)
for k,axx in enumerate(axs):
    tg,cb,ctx=blocks(g); G=14
    axx.add_patch(Rectangle((0,0),G,G,fc='#ecf0f1',ec='none'))
    for i,j in zip(*np.where(ctx)): axx.add_patch(Rectangle((j,G-1-i),1,1,fc='#2980b9',ec='w',lw=.35))
    for t_,col in zip(tg,['#c0392b','#e67e22','#8e44ad','#16a085']):
        for i,j in zip(*np.where(t_)): axx.add_patch(Rectangle((j,G-1-i),1,1,fc=col,ec='w',lw=.35,alpha=.9))
    axx.set_xlim(0,G); axx.set_ylim(0,G); axx.set_aspect('equal'); axx.axis('off'); axx.grid(False)
    axx.set_title(f'context {ctx.sum()}/{G*G} = {ctx.sum()/(G*G):.0%}',fontsize=6.8)
fig.suptitle('multi-block masking: 4 target blocks (colours), context = big block minus targets (blue)',fontsize=7.4,y=1.02)
fig.tight_layout(); fig.savefig(F+'masks.pdf',bbox_inches='tight')

# ---- Fig 5: why pixel loss wastes capacity -----------------------------
fig,ax=plt.subplots(1,3,figsize=(7.2,1.95))
gg=np.random.default_rng(3); x=np.linspace(0,1,220)
sig=np.sin(2*np.pi*x*1.4)*.8
for k,(t_,ttl) in enumerate([(sig,'the signal you care about'),
    (sig+.55*gg.standard_normal(220),'what the pixels give you'),
    (sig,'what a latent target keeps')]):
    ax[k].plot(x,t_,lw=1.0,color=['#2c3e50','#c0392b','#27ae60'][k])
    if k==1: ax[k].plot(x,sig,lw=1.0,color='#2c3e50',alpha=.35)
    ax[k].set_title(ttl,fontsize=7.4,loc='left'); ax[k].set_ylim(-2.2,2.2)
    ax[k].set_yticks([]); ax[k].set_xticks([]); ax[k].grid(False)
fig.tight_layout(); fig.savefig(F+'pixelnoise.pdf',bbox_inches='tight')
print('figs:', __import__('os').listdir(F))
