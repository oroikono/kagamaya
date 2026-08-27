"""Zero-shot external FHN transfer with compact nonperiodic weak tests."""

from __future__ import annotations

import argparse
from itertools import combinations
import hashlib
import json
import math
from pathlib import Path
import pickle
import re

import numpy as np

from .contract import (DEFAULT_DEVELOPMENT_THRESHOLDS, DIAGNOSE_VIEW_ID,
                       NOISE_CELLS, VIEW_BANK)
from .learning import choose


SPATIAL_MAP = {12: 16, 18: 32, 24: 64, 36: 128}
U_ATOMS = ("one", "u", "u3", "v", "lap_u")
V_ATOMS = ("u", "v", "lap_v")


def _block_reduce(values: np.ndarray, desired: int) -> np.ndarray:
    nt, nx, ny = values.shape
    if nx != ny or nx % desired:
        raise ValueError("external grid is not block-nested")
    factor = nx // desired
    return values.reshape(nt, desired, factor, desired, factor).mean(axis=(2, 4))


def _compact_spatial(kind, kx, ky, x, y):
    lx = float(x[-1] - x[0] + (x[1] - x[0]))
    ly = float(y[-1] - y[0] + (y[1] - y[0]))
    xmin = float(x[0] - 0.5 * (x[1] - x[0]))
    ymin = float(y[0] - 0.5 * (y[1] - y[0]))
    zx = (x - xmin) / lx
    zy = (y - ymin) / ly
    ex = zx**2 * (1 - zx)**2
    ey = zy**2 * (1 - zy)**2
    ex1 = (2*zx - 6*zx**2 + 4*zx**3) / lx
    ey1 = (2*zy - 6*zy**2 + 4*zy**3) / ly
    ex2 = (2 - 12*zx + 12*zx**2) / lx**2
    ey2 = (2 - 12*zy + 12*zy**2) / ly**2
    xx, yy = zx[:, None], zy[None, :]
    phase = 2*np.pi*(kx*xx + ky*yy) + (0.173 if kind == "sin" else -0.291)
    if kind == "sin":
        base, first = np.sin(phase), np.cos(phase)
    else:
        base, first = np.cos(phase), -np.sin(phase)
    kxp, kyp = 2*np.pi*kx/lx, 2*np.pi*ky/ly
    envelope = ex[:, None]*ey[None, :]
    spatial = envelope*base
    sxx = (ex2[:, None]*ey[None, :]*base
           + 2*ex1[:, None]*ey[None, :]*kxp*first
           - envelope*kxp**2*base)
    syy = (ex[:, None]*ey2[None, :]*base
           + 2*ex[:, None]*ey1[None, :]*kyp*first
           - envelope*kyp**2*base)
    return spatial, sxx + syy


def _temporal(q, times, start, end):
    s = (times-start)/(end-start)
    b = s**(q+2)*(1-s)**2
    db = ((q+2)*s**(q+1)*(1-s)**2 - 2*s**(q+2)*(1-s))/(end-start)
    return b, db


def _integrate_space(values, x, y):
    return np.trapezoid(np.trapezoid(values, y, axis=2), x, axis=1)


def weak_matrix(u, v, times, x, y, view, target):
    desired = SPATIAL_MAP[view.spatial_nodes]
    u, v = _block_reduce(u, desired), _block_reduce(v, desired)
    x = x.reshape(desired, -1).mean(axis=1)
    y = y.reshape(desired, -1).mean(axis=1)
    normalized_t = (times-times[0])/(times[-1]-times[0])
    eligible = np.flatnonzero((normalized_t >= view.window_start-1e-12)
                              & (normalized_t <= view.window_end+1e-12))
    pos = np.rint(np.linspace(0, len(eligible)-1, view.temporal_samples)).astype(int)
    tidx = eligible[pos]
    t, uu, vv = normalized_t[tidx], u[tidx], v[tidx]
    atoms = U_ATOMS if target == "u" else V_ATOMS
    field = uu if target == "u" else vv
    rows, targets, roles = [], [], []
    for q in range(view.test_mode_count):
        bt, bdt = _temporal(q, t, view.window_start, view.window_end)
        for kx in range(view.test_mode_count+1):
            for ky in range(view.test_mode_count+1):
                if kx == ky == 0:
                    continue
                for kind_index, kind in enumerate(("sin", "cos")):
                    spatial, lap = _compact_spatial(kind, kx, ky, x, y)
                    phi = bt[:, None, None]*spatial[None, :, :]
                    targets.append(float(np.trapezoid(
                        _integrate_space(-field*bdt[:, None, None]*spatial, x, y), t)))
                    row = []
                    for atom in atoms:
                        if atom == "one": integrand = phi
                        elif atom == "u": integrand = uu*phi
                        elif atom == "v": integrand = vv*phi
                        elif atom == "u3": integrand = uu**3*phi
                        elif atom == "lap_u": integrand = uu*bt[:, None, None]*lap
                        elif atom == "lap_v": integrand = vv*bt[:, None, None]*lap
                        else: raise ValueError(atom)
                        row.append(float(np.trapezoid(_integrate_space(integrand, x, y), t)))
                    rows.append(row)
                    roles.append("train" if (q+kx+2*ky+kind_index)%2 == 0 else "validate")
    return np.asarray(rows), np.asarray(targets), np.asarray(roles), atoms


def authority(x, y, roles, atoms):
    candidates = []
    train, valid = roles == "train", roles == "validate"
    for size in range(1, len(atoms)+1):
        for cols in combinations(range(len(atoms)), size):
            xt = x[train][:, cols]
            norms = np.linalg.norm(xt, axis=0)
            if np.any(norms <= 1e-13): continue
            beta_n, _, rank, _ = np.linalg.lstsq(xt/norms, y[train], rcond=None)
            if rank != len(cols): continue
            beta = beta_n/norms
            res = np.linalg.norm(y[valid]-x[valid][:, cols]@beta)/max(np.linalg.norm(y[valid]),1e-15)
            candidates.append((float(res+2e-5*size), size, cols))
    candidates.sort(key=lambda z:(z[0],z[1],tuple(atoms[i] for i in z[2])))
    if not candidates:
        return {"status":"INCONCLUSIVE","support":[],"residual":math.inf,
                "score_gap":0.,"condition":math.inf,"minimum_coefficient":0.,"ambiguity_size":0}
    best=candidates[0]; cols=best[2]; winner=set(cols)
    rivals=[z for z in candidates[1:] if not winner < set(z[2])]
    gap=(rivals[0][0]-best[0]) if rivals else math.inf
    full=x[:,cols]; norms=np.linalg.norm(full,axis=0)
    beta_n,_,rank,_=np.linalg.lstsq(full/norms,y,rcond=None); coef=beta_n/norms
    residual=float(np.linalg.norm(y-full@coef)/max(np.linalg.norm(y),1e-15))
    condition=float(np.linalg.cond(full/norms)) if rank==len(cols) else math.inf
    minimum=float(np.min(np.abs(coef)))
    th=DEFAULT_DEVELOPMENT_THRESHOLDS
    support=tuple(atoms[i] for i in cols)
    if residual >= th.outside_library_residual_min: status="OUTSIDE_LIBRARY"; emitted=()
    elif residual <= th.identified_residual_max and condition <= th.condition_max and minimum >= th.coefficient_floor and gap >= th.support_gap_min:
        status="IDENTIFIED"; emitted=support
    else: status="INCONCLUSIVE"; emitted=()
    return {"status":status,"support":list(emitted),"residual":residual,
            "score_gap":gap,"condition":condition,"minimum_coefficient":minimum,
            "ambiguity_size":1+sum(z[0]<=best[0]+th.support_gap_min for z in rivals)}


def perturb(a, model, level, seed):
    if level == 0: return np.array(a,copy=True)
    rng=np.random.default_rng(seed); e=rng.standard_normal(a.shape)
    if model == "correlated":
        e=(e+np.roll(e,1,1)+np.roll(e,-1,1)+np.roll(e,1,2)+np.roll(e,-1,2))/np.sqrt(5)
        e=.75*e+.25*np.roll(e,1,0)
    return a+level*max(float(np.std(a)),1e-12)*e


def owner_score(record, target):
    truth = U_ATOMS if target == "u" else V_ATOMS
    correct = record["status"] == "IDENTIFIED" and tuple(record["support"]) == truth
    false = record["status"] != "INCONCLUSIVE" and not correct
    return correct, false


def main():
    import h5py
    p=argparse.ArgumentParser(); p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True); p.add_argument("--receipt",type=Path,required=True)
    p.add_argument("--dominance",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); manifest=json.loads(a.manifest.read_text()); receipt=json.loads(a.receipt.read_text())
    receipt_clone=dict(receipt); receipt_clone.pop("receipt_sha256",None)
    if receipt.get("receipt_sha256") != hashlib.sha256(json.dumps(
            receipt_clone,sort_keys=True,separators=(",",":")).encode()).hexdigest():
        raise SystemExit("receipt semantic digest")
    models_path=a.receipt.parent/"models.pkl"
    if hashlib.sha256(models_path.read_bytes()).hexdigest()!=receipt["model_sha256"]: raise SystemExit("model hash")
    mc,mf=pickle.loads(models_path.read_bytes()); fixed=receipt["matched_fixed"]
    rows=[]
    for item in manifest["files"]:
        path=a.data_root/item["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item["sha256"]: raise SystemExit("external hash")
        match=re.search(r"Du=([0-9.]+)_Dv=([0-9.]+)_k=([0-9.]+)",path.name)
        with h5py.File(path,"r") as h:
            for group in sorted(h.keys()):
                data=h[f"{group}/data"][:]; times=h[f"{group}/grid/t"][:]
                x=h[f"{group}/grid/x"][:]; y=h[f"{group}/grid/y"][:]
                for noise_model,noise_level in NOISE_CELLS:
                    base=int.from_bytes(hashlib.sha256(f"{item['path']}|{group}|{noise_model}|{noise_level}".encode()).digest()[:8],"little")
                    u=perturb(data[...,0],noise_model,noise_level,base); v=perturb(data[...,1],noise_model,noise_level,base+1)
                    for target in ("u","v"):
                        records={}
                        for view in VIEW_BANK:
                            xm,ym,roles,atoms=weak_matrix(u,v,times,x,y,view,target)
                            records[view.view_id]=authority(xm,ym,roles,atoms)|{"view_id":view.view_id,
                                "registered_cost":view.registered_cost,"episode_id":"external"}
                        episode={"episode_id":hashlib.sha256(f"{item['path']}|{group}|{noise_model}|{noise_level}|{target}".encode()).hexdigest()[:24],"views":list(records.values())}
                        selected=choose(mc,mf,episode,receipt["selected_risk_threshold"])
                        ar=records[selected] if selected else {"status":"INCONCLUSIVE","support":[]}
                        fr=records[fixed["submit"]]
                        ac,af=owner_score(ar,target); fc,ff=owner_score(fr,target)
                        rows.append({"path":item["path"],"split":item["path"].split('/')[-2],"group":group,
                            "target":target,"noise_model":noise_model,"noise_level":noise_level,
                            "agent_view":selected,"agent_status":ar["status"],
                            "agent_support":ar["support"],
                            "agent_correct":ac,"agent_false":af,
                            "fixed_view":fixed["submit"],"fixed_status":fr["status"],
                            "fixed_support":fr["support"],
                            "fixed_correct":fc,"fixed_false":ff,
                            "parameters":None if match is None else [float(x) for x in match.groups()]})
    n=len(rows); result={"schema_version":"maintrack.external-fhn-transfer.v1","n":n,
        "manifest_sha256":hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
        "model_receipt_sha256":receipt["receipt_sha256"],
        "fixed_policy":{"diagnose":DIAGNOSE_VIEW_ID,"submit":fixed["submit"]},
        "agent":{"correct":sum(r['agent_correct'] for r in rows),"false":sum(r['agent_false'] for r in rows)},
        "fixed":{"correct":sum(r['fixed_correct'] for r in rows),"false":sum(r['fixed_false'] for r in rows)},
        "rows":rows,"claim_ceiling":"independent_numerical_transfer_not_physical_validation"}
    result["content_sha256"]=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(",",":" )).encode()).hexdigest()
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.write_text(json.dumps(result,sort_keys=True,separators=(",",":"))+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},sort_keys=True))


if __name__ == "__main__": main()
