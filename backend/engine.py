from __future__ import annotations
import math, heapq
from typing import Dict, Any, List, Tuple
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve, expm_multiply

KB_EV = 8.617333262e-5
NU0 = 1.0e13
R_INFLUENCE_A = 0.85
R_PAIR_A = 1.25

def unique_rows(a, decimals=8):
    return np.unique(np.round(a, decimals), axis=0) if len(a) else a.reshape(0, 3)

def lattice_points(crystal, nx, ny, nz, a):
    basis = [(0,0,0),(0.5,0.5,0.5)] if crystal == "BCC" else [(0,0,0),(0.5,0.5,0),(0.5,0,0.5),(0,0.5,0.5)]
    pts = [(i+b[0], j+b[1], k+b[2]) for i in range(nx) for j in range(ny) for k in range(nz) for b in basis]
    return np.asarray(pts, float) * a

def bcc_tetra(nx, ny, nz, a):
    basis=np.array([(0.5,0.25,0),(0.5,0.75,0),(0.25,0.5,0),(0.75,0.5,0),(0.5,0,0.25),(0.5,0,0.75),(0.25,0,0.5),(0.75,0,0.5),(0,0.5,0.25),(0,0.5,0.75),(0,0.25,0.5),(0,0.75,0.5)],float)
    pts=[]
    lim=np.array([nx,ny,nz],float)
    for i in range(nx+1):
        for j in range(ny+1):
            for k in range(nz+1):
                p=basis+np.array([i,j,k],float); pts.extend(p[np.all((p>=-1e-9)&(p<=lim+1e-9),axis=1)].tolist())
    return unique_rows(np.asarray(pts)*a)

def bcc_octa(nx, ny, nz, a):
    gx=np.arange(0,nx+.25,.5); gy=np.arange(0,ny+.25,.5); gz=np.arange(0,nz+.25,.5)
    pts=np.asarray([(x,y,z) for x in gx for y in gy for z in gz],float)
    nh=np.sum(np.abs(pts-np.round(pts))>1e-8,axis=1)
    pts=pts[((nh==1)|(nh==2)) & np.all((pts>=-1e-9)&(pts<=np.array([nx,ny,nz])+1e-9),axis=1)]
    return unique_rows(pts*a)

def fcc_tetra(nx, ny, nz, a):
    basis=np.array([(0.25,0.25,0.25),(0.25,0.25,0.75),(0.25,0.75,0.25),(0.75,0.25,0.25),(0.75,0.75,0.25),(0.75,0.25,0.75),(0.25,0.75,0.75),(0.75,0.75,0.75)],float)
    pts=[tuple(p) for i in range(nx) for j in range(ny) for k in range(nz) for p in (basis+np.array([i,j,k],float))]
    return unique_rows(np.asarray(pts)*a)

def fcc_octa(nx, ny, nz, a):
    basis=np.array([(0.5,0.5,0.5),(0.5,0,0),(0,0.5,0),(0,0,0.5),(0.5,0.5,0),(0.5,0,0.5),(0,0.5,0.5)],float)
    pts=[]; lim=np.array([nx,ny,nz],float)
    for i in range(nx+1):
        for j in range(ny+1):
            for k in range(nz+1):
                p=basis+np.array([i,j,k],float); pts.extend(p[np.all((p>=-1e-9)&(p<=lim+1e-9),axis=1)].tolist())
    return unique_rows(np.asarray(pts)*a)

def choose_network(net, species):
    return ("tetrahedral" if species in {"H","D","T"} else "octahedral") if net == "auto" else net

def interstitial_sites(crystal, net, nx, ny, nz, a):
    parts=[]
    if crystal=="BCC":
        if net in {"tetrahedral","tetra_octa"}: parts.append(bcc_tetra(nx,ny,nz,a))
        if net in {"octahedral","tetra_octa"}: parts.append(bcc_octa(nx,ny,nz,a))
    else:
        if net in {"tetrahedral","tetra_octa"}: parts.append(fcc_tetra(nx,ny,nz,a))
        if net in {"octahedral","tetra_octa"}: parts.append(fcc_octa(nx,ny,nz,a))
    if not parts: raise ValueError("No interstitial sites generated.")
    return unique_rows(np.vstack(parts))

def build_edges(sites):
    tree=cKDTree(sites); d,_=tree.query(sites,k=2); nn=float(np.median(d[:,1]))
    pairs=np.array(sorted(tree.query_pairs(r=1.08*nn)),dtype=int)
    if len(pairs)==0: raise ValueError("No interstitial edges found.")
    mids=.5*(sites[pairs[:,0]]+sites[pairs[:,1]]); lens=np.linalg.norm(sites[pairs[:,0]]-sites[pairs[:,1]],axis=1)
    return pairs,mids,lens

def rng_from_setup(setup):
    return np.random.default_rng(abs(hash(str(setup.get("composition_at_percent",{}))+setup.get("crystal_structure","")))%(2**32))

def place_solutes(host_pts, setup):
    rng=rng_from_setup(setup); counts=setup["domain"]["estimated_atom_counts"]; idx=np.arange(len(host_pts)); rng.shuffle(idx); out={}; cur=0
    for el in setup.get("alloying_elements",[]):
        n=min(int(counts.get(el,0)), max(0,len(idx)-cur)); out[el]=host_pts[idx[cur:cur+n]]; cur+=n
    return out

def fields(sites, mids, solutes, e, a):
    U=np.full(len(sites),float(e.get("host_site_energy",0.0))); S=np.zeros(len(mids)); R=R_INFLUENCE_A*a
    for el,pts in solutes.items():
        if len(pts)==0: continue
        tr=cKDTree(pts); ds,_=tr.query(sites,k=1); dm,_=tr.query(mids,k=1)
        Is=np.exp(-(ds/R)**2); Im=np.exp(-(dm/R)**2)
        U += float(e.get(f"{el}_site_energy_shift",0.0))*Is
        U -= abs(float(e.get(f"{el}_cluster_trap_depth",0.0)))*(Is**2)
        S += float(e.get(f"{el}_saddle_energy_shift",0.0))*Im
    els=list(solutes.keys())
    for i in range(len(els)):
        for j in range(i+1,len(els)):
            ael,bel=els[i],els[j]; ps=float(e.get(f"pair_{ael}_{bel}_saddle_shift",e.get(f"pair_{bel}_{ael}_saddle_shift",0.0)))
            if ps==0 or len(solutes[ael])==0 or len(solutes[bel])==0: continue
            da,_=cKDTree(solutes[ael]).query(mids,k=1); db,_=cKDTree(solutes[bel]).query(mids,k=1)
            S += ps*np.exp(-((da+db)/(R_PAIR_A*a))**2)
    return U,S

def rate_system(sites, edges, lens, U, S, setup, e):
    n=len(sites); kBT=KB_EV*float(setup["temperature_K"]); E0=float(e.get("host_migration_barrier",0.1))
    rows=[]; cols=[]; vals=[]; adj=[[] for _ in range(n)]; eb=[]
    for ei,(i,j) in enumerate(edges):
        i=int(i); j=int(j); sad=max(U[i],U[j])+E0+S[ei]; Eij=max(sad-U[i],1e-4); Eji=max(sad-U[j],1e-4)
        kij=NU0*math.exp(-Eij/max(kBT,1e-30)); kji=NU0*math.exp(-Eji/max(kBT,1e-30))
        rows += [i,j]; cols += [j,i]; vals += [kij,kji]
        adj[i].append((j,kij,Eij,ei)); adj[j].append((i,kji,Eji,ei)); eb.append(.5*(Eij+Eji))
    W=csr_matrix((vals,(rows,cols)),shape=(n,n)); sumj=np.asarray(W.sum(axis=1)).ravel(); z=sites[:,2]; a=float(setup["lattice_parameter_A"])
    nearS=z<=z.min()+.35*a; nearB=z>=z.max()-.35*a; ks=np.zeros(n); kb=np.zeros(n)
    Es=float(e.get("surface_exit_barrier",E0)); Eb=float(e.get("bulk_exit_barrier",E0))
    for q in np.where(nearS)[0]: ks[q]=NU0*math.exp(-max(Es-U[q],1e-4)/max(kBT,1e-30))
    for q in np.where(nearB)[0]: kb[q]=NU0*math.exp(-max(Eb-U[q],1e-4)/max(kBT,1e-30))
    total=sumj+ks+kb; A=diags(total,0,format="csr")-W
    return {"W":W,"A":A,"adj":adj,"total":total,"ks":ks,"kb":kb,"edge_activation":np.asarray(eb)}

def source(sites, setup):
    a=float(setup["lattice_parameter_A"]); nx,ny,nz=setup["domain"]["Nx"],setup["domain"]["Ny"],setup["domain"]["Nz"]
    x,y,z=sites[:,0],sites[:,1],sites[:,2]; cx,cy=.5*nx*a,.5*ny*a
    z0=(.25 if setup["boundary"]=="surface_return" else .1 if setup["boundary"]=="bulk_transmission" else .5)*nz*a
    w=np.exp(-((z-z0)/max(.75*a,.06*nz*a))**2)*np.exp(-(((x-cx)**2+(y-cy)**2)/max(1.5*a,.18*min(nx,ny)*a)**2))
    if w.sum()<=0: w[:]=1
    return w/w.sum()

def solve_fp(sys, st):
    A=sys["A"]; qs=np.clip(spsolve(A,sys["ks"]),0,1); qb=1-qs; t=np.maximum(spsolve(A,np.ones(A.shape[0])),0); res=np.maximum(spsolve(A.T,st),0)
    return {"p_surface":float(st@qs),"p_bulk":float(st@qb),"mfpt":float(st@t),"residence":res}

def survival(sys, st, mfpt, n=60):
    Q=sys["W"]-diags(sys["total"],0,format="csr"); tmax=max(5*mfpt,1e-14)
    try:
        P=expm_multiply(Q.T,st,start=0,stop=tmax,num=n); S=np.clip(P.sum(axis=1),0,1); times=np.linspace(0,tmax,n)
    except Exception:
        times=np.linspace(0,tmax,n); S=np.exp(-times/max(mfpt,1e-30))
    return {"time_s":times.tolist(),"S_t":S.tolist()}

def shortest(sites, sys, start, target):
    n=len(sites); dist=np.full(n,np.inf); prev=np.full(n,-1,dtype=int); dist[start]=0; heap=[(0,start)]; rates=sys["ks"] if target=="surface" else sys["kb"]; best=None; bestv=np.inf
    while heap:
        d,u=heapq.heappop(heap)
        if d!=dist[u]: continue
        if rates[u]>0 and d+1/max(rates[u],1e-300)<bestv: best=u; bestv=d+1/max(rates[u],1e-300)
        for v,k,E,ei in sys["adj"][u]:
            nd=d+1/max(k,1e-300)
            if nd<dist[v]: dist[v]=nd; prev[v]=u; heapq.heappush(heap,(nd,v))
    if best is None: return [start], float("inf")
    path=[]; c=int(best)
    while c!=-1: path.append(c); c=int(prev[c]) if prev[c]!=-1 else -1
    return path[::-1], float(bestv)

def path_metrics(path, sites, sys):
    if len(path)<2: return {"tortuosity":1,"barriers":[],"mean_barrier":float("nan"),"bottleneck":float("nan")}
    pts=sites[np.asarray(path)]; length=float(np.sum(np.linalg.norm(np.diff(pts,axis=0),axis=1))); straight=float(np.linalg.norm(pts[-1]-pts[0])); tort=length/max(straight,1e-12)
    lookup={(u,v):float(E) for u,lst in enumerate(sys["adj"]) for v,k,E,ei in lst}; b=[lookup.get((int(u),int(v)),float("nan")) for u,v in zip(path[:-1],path[1:])]; arr=np.asarray(b,float)
    return {"tortuosity":tort,"barriers":b,"mean_barrier":float(np.nanmean(arr)),"bottleneck":float(np.nanmax(arr))}

def thin(points,a,max_points=1000):
    if len(points)>max_points: points=points[np.linspace(0,len(points)-1,max_points).astype(int)]
    return {"x":(points[:,0]/a).tolist(),"z":(points[:,2]/a).tolist()}

def solute_out(solutes,a,max_each=300):
    out=[]
    for el,pts in solutes.items():
        if len(pts)>max_each: pts=pts[np.linspace(0,len(pts)-1,max_each).astype(int)]
        out += [{"element":el,"x":float(p[0]/a),"y":float(p[1]/a),"z":float(p[2]/a)} for p in pts]
    return out

def run_simulation(setup: Dict[str,Any], energies_eV: Dict[str,float]) -> Dict[str,Any]:
    if len(setup.get("alloying_elements",[]))>3: raise ValueError("This revised backend supports up to three alloying elements.")
    nx,ny,nz=int(setup["domain"]["Nx"]),int(setup["domain"]["Ny"]),int(setup["domain"]["Nz"]); a=float(setup["lattice_parameter_A"]); crystal=setup["crystal_structure"]
    net=choose_network(setup.get("site_network","auto"),setup["interstitial_species"]); host=lattice_points(crystal,nx,ny,nz,a); sites=interstitial_sites(crystal,net,nx,ny,nz,a); edges,mids,lens=build_edges(sites)
    sol=place_solutes(host,setup); U,S=fields(sites,mids,sol,energies_eV,a); sys=rate_system(sites,edges,lens,U,S,setup,energies_eV); st=source(sites,setup); fp=solve_fp(sys,st); surv=survival(sys,st,fp["mfpt"])
    start=int(np.argmax(st)); ps,Rs=shortest(sites,sys,start,"surface"); pb,Rb=shortest(sites,sys,start,"bulk")
    target="surface" if fp["p_surface"]>=fp["p_bulk"] else "bulk"; path=ps if target=="surface" else pb; R=Rs if target=="surface" else Rb; pm=path_metrics(path,sites,sys)
    trap_mask=U<=np.percentile(U,20); total_res=float(np.sum(fp["residence"])); trap_frac=float(np.sum(fp["residence"][trap_mask])/max(total_res,1e-300))
    E0=float(energies_eV.get("host_migration_barrier",.1)); acc=float(np.mean(sys["edge_activation"]<=E0+.05)); meanb=pm["mean_barrier"] if not math.isnan(pm["mean_barrier"]) else E0; reten=float(math.exp((meanb-E0)/max(KB_EV*setup["temperature_K"],1e-30))*pm["tortuosity"])
    hi=np.where(sys["edge_activation"]>=np.percentile(sys["edge_activation"],90))[0]; hi=hi[np.linspace(0,len(hi)-1,min(len(hi),500)).astype(int)] if len(hi) else hi; ti=np.where(trap_mask)[0]; ti=ti[np.linspace(0,len(ti)-1,min(len(ti),500)).astype(int)] if len(ti) else ti
    pts=sites[np.asarray(path)]; barriers=pm["barriers"]; marker_barriers=([barriers[0]]+barriers) if len(barriers)==len(path)-1 and barriers else [meanb]*len(path)
    return {"metrics":{"mean_first_passage_time_s":fp["mfpt"],"probability_bulk_transmission":fp["p_bulk"],"probability_surface_return":fp["p_surface"],"bottleneck_barrier_eV":pm["bottleneck"],"minimum_resistance_path_s":R,"path_tortuosity":pm["tortuosity"],"pathway_energy_cost_eV":pm["mean_barrier"],"trap_residence_fraction":trap_frac,"percolation_accessibility_fraction":acc,"retention_factor_vs_host":reten},"survival_probability":surv,"dominant_pathway":{"target":target,"x":(pts[:,0]/a).tolist(),"y":(pts[:,1]/a).tolist(),"z":(pts[:,2]/a).tolist(),"activation_barrier_eV":marker_barriers},"lattice_points":thin(host,a,1400),"interstitial_sites":thin(sites,a,1400),"solute_positions":solute_out(sol,a),"trap_sites":{"x":(sites[ti,0]/a).tolist(),"z":(sites[ti,2]/a).tolist()},"high_barrier_edges":{"x":(mids[hi,0]/a).tolist(),"z":(mids[hi,2]/a).tolist()},"domain":setup["domain"],"site_network_used":net,"notes":{"energy_definition":"Activation barrier is E_act(i->j)=E_saddle(i,j)-U_i. Site energies U_i are explicitly included.","no_energy_estimation":"All energy values are supplied by the user.","dominant_path":"Dominant path is chosen by larger absorption probability and extracted as minimum-resistance path."}}
