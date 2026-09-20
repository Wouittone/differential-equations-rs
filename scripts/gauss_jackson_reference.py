"""Reproduce independent GJ8 fixtures, never imported by the Rust crate.

Requires numpy/scipy. Downloads the pinned BSD-2-Clause reference and its license
into target/gj-reference (not redistributed in the package). The reference uses
centered startup; our crate uses one-sided order-10 startup, so coarse-step
trajectories need not agree to roundoff. Analytic solutions remain the oracle.
"""
from pathlib import Path
import hashlib, json, sys, urllib.request
import numpy as np
REVISION = 'b011576364d77fb606fae6f02415265c33215421'
BASE = f'https://raw.githubusercontent.com/lorcan2440/Gauss-Jackson-Integrator/{REVISION}'
directory = Path('target/gj-reference'); directory.mkdir(parents=True, exist_ok=True)
for relative in ['license.txt', 'python_solver/GJ8.py']:
    target = directory / Path(relative).name
    if not target.exists(): target.write_bytes(urllib.request.urlopen(f'{BASE}/{relative}').read())
sys.path.insert(0, str(directory.resolve()))
expected_hash = '5ac4551f5012a5452c948d2effb068263fcc8b226dc72e01dd15b7b4c64a3d68'
assert hashlib.sha256((directory/'GJ8.py').read_bytes()).hexdigest() == expected_hash, 'reference source hash mismatch'
from GJ8 import gauss_jackson_8
cases=[]
for h in [0.4,0.2,0.1]:
    t,q,v,_=gauss_jackson_8(lambda t,q,v:-q,(0.,20.),np.array([1.]),np.array([0.]),h,conv_max_iter=50,rel_tol=1e-13,abs_tol=1e-14)
    cases.append(dict(case='harmonic',h=h,time=float(t[-1]),q=q[-1].tolist(),v=v[-1].tolist(),analytic_error=float(max(abs(q[-1,0]-np.cos(t[-1])),abs(v[-1,0]+np.sin(t[-1]))))))
t,q,v,_=gauss_jackson_8(lambda t,q,v:-q-2*v,(0.,3.),np.array([1.]),np.array([-1.]),0.1,conv_max_iter=50,rel_tol=1e-13,abs_tol=1e-14)
cases.append(dict(case='damped',h=0.1,time=float(t[-1]),q=q[-1].tolist(),v=v[-1].tolist(),analytic_error=float(max(abs(q[-1,0]-np.exp(-t[-1])),abs(v[-1,0]+np.exp(-t[-1]))))))
result=dict(repository='https://github.com/lorcan2440/Gauss-Jackson-Integrator',revision=REVISION,license='BSD-2-Clause',source_sha256=hashlib.sha256((directory/'GJ8.py').read_bytes()).hexdigest(),cases=cases)
output=Path('tests/fixtures/gauss_jackson_reference.json');output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
