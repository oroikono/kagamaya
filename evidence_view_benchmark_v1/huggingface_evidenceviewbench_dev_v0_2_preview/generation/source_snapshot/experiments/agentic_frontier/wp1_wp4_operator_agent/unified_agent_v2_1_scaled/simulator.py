"""PDE outcome producer with heterogeneous finite-acquisition channels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..maintrack_suite_v41.numerics import dealias_mask, integrate_etdrk4
from ..maintrack_suite_v41.simulator import Family, HiddenTruth, TRUE_SUPPORTS, Trajectory


@dataclass(frozen=True)
class Channel:
    name: str; nx: int; t_final: float; save_dt: float; cost: float


BASELINE=Channel("baseline",64,0.24,0.01,0.0)
CHANNELS=(Channel("reaction_time",32,0.48,0.01,1.0),
          Channel("gradient_space",96,0.12,0.02,1.0),
          Channel("dispersion_long",64,0.36,0.02,1.0))


def _initial(channel: Channel,x: np.ndarray,seed: int)->np.ndarray:
    p=np.random.default_rng(seed).uniform(0,2*np.pi,3)
    if channel.name=="baseline": return .18+.035*np.cos(x+p[0])+.012*np.sin(2*x+p[1])
    if channel.name=="reaction_time": return .48+.24*np.cos(x+p[0])+.10*np.sin(2*x+p[1])
    if channel.name=="gradient_space": return .42*np.sin(2*x+p[0])+.24*np.cos(4*x+p[1])+.10*np.sin(6*x+p[2])
    if channel.name=="dispersion_long": return .30*np.cos(x+p[0])+.20*np.sin(3*x+p[1])+.09*np.cos(5*x+p[2])
    raise ValueError(channel.name)


def _problem(family: Family,k: np.ndarray,rng: np.random.Generator):
    mask=dealias_mask(k.size)
    def ph(v,p):
        z=np.fft.fft(np.fft.ifft(v).real**p); z[~mask]=0; return z
    zero=lambda v:np.zeros_like(v)
    if family is Family.ADVECTION_DIFFUSION:
        c,d=float(rng.uniform(.25,.55)),float(rng.uniform(.025,.050)); return -d*k**2-1j*c*k,zero,(("u_x",-c),("u_xx",d))
    if family is Family.VISCOUS_BURGERS:
        a,nu=float(rng.uniform(.75,1.25)),float(rng.uniform(.035,.070)); return -nu*k**2,lambda v:-.5j*a*k*ph(v,2),(("u_xx",nu),("u u_x",-a))
    if family is Family.FISHER_KPP:
        r,d=float(rng.uniform(.8,1.4)),float(rng.uniform(.020,.045)); return -d*k**2,lambda v:r*(v-ph(v,2)),(("u",r),("u^2",-r),("u_xx",d))
    if family is Family.ALLEN_CAHN:
        r,d=float(rng.uniform(.7,1.2)),float(rng.uniform(.020,.045)); return -d*k**2,lambda v:r*(v-ph(v,3)),(("u",r),("u^3",-r),("u_xx",d))
    if family is Family.KDV:
        a,q=float(rng.uniform(1.5,3.0)),float(rng.uniform(.18,.35)); return 1j*q*k**3,lambda v:-.5j*a*k*ph(v,2),(("u_xxx",-q),("u u_x",-a))
    g,r=float(rng.uniform(.018,.030)),float(rng.uniform(.15,.30)); return -g*k**4,lambda v:r*(v-ph(v,3)),(("u",r),("u^3",-r),("u_xxxx",-g))


def _noise(clean:np.ndarray,level:float,model:str,seed:int)->np.ndarray:
    if level==0:return clean.copy()
    z=np.random.default_rng(seed).normal(size=clean.shape)
    if model=="correlated":
        ft,fx=np.fft.fftfreq(clean.shape[0])[:,None],np.fft.fftfreq(clean.shape[1])[None,:]
        z=np.fft.ifft2(np.fft.fft2(z)*np.exp(-.5*((ft/.20)**2+(fx/.15)**2))).real
    elif model!="iid":raise ValueError(model)
    z/=max(float(np.std(z)),1e-15); return clean+level*max(float(np.std(clean)),1e-12)*z


def simulate(family:Family,channel:Channel,episode_seed:int,public_seed:int,noise:float,noise_model:str):
    internal_dt=.00025; steps=int(round(channel.t_final/internal_dt)); stride=int(round(channel.save_dt/internal_dt))
    if steps%stride or (steps//stride+1)%2==0: raise ValueError("channel needs an odd saved grid")
    length=2*np.pi; dx=length/channel.nx; x=np.arange(channel.nx)*dx; k=2*np.pi*np.fft.fftfreq(channel.nx,d=dx)
    linear,nonlinear,coeff=_problem(family,k,np.random.default_rng(episode_seed))
    clean=integrate_etdrk4(_initial(channel,x,public_seed),linear,nonlinear,internal_dt,steps,stride)
    nseed=int(np.random.SeedSequence([episode_seed,public_seed,991,sum(map(ord,channel.name))]).generate_state(1)[0])
    u=_noise(clean,noise,noise_model,nseed); t=np.linspace(0,channel.t_final,steps//stride+1)
    return Trajectory(f"e{episode_seed}-{channel.name}",channel.name,x,t,u,noise_model,noise),HiddenTruth(family,TRUE_SUPPORTS[family],coeff,episode_seed)

