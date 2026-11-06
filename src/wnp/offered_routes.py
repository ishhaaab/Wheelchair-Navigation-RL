"""Gated synthetic offered-route studies; final layouts require a sealed approval."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np
import torch

from wnp.env import GridMDP, _coordinate, _integer
from wnp.features import feature_table
from wnp.route_choice import choice_log_probs, choice_metrics, draw_choices, predict_route, route_features
from wnp.route_metrics import summarize_layouts
from wnp.route_design import audit_choice_design
from wnp.routes import CandidateConstructionError, candidate_pair, tradeoff_candidate_pair
from wnp.soft_dp import plan
from wnp.verification import terminal_margin_diagnostic


ROOT = Path(__file__).resolve().parents[2]
RECOVERY_SHA256 = '815ba19c58ba73e4d58305863af7f0e31f2591c8e81ad5064cbce1491f2551c5'
CHECKPOINT = '29051aebf4f3a784f2b4ec9e7deffa6c557ee729'


@dataclass(frozen=True)
class OfferedRouteConfig:
    shape: tuple[int,int] = (20,20)
    start: tuple[int,int,int] = (0,0,0)
    goal: tuple[int,int] = (19,19)
    headings: int = 4
    obstacle_probability: float = .20
    max_attempts: int = 1000
    d_safe: float = 2.
    horizon: int = 160
    terminal_penalty: float = 200.
    development_seeds: tuple[int,...] = tuple(range(10000,10010))
    final_seeds: tuple[int,...] = tuple(range(20000,20100))
    choices_per_layout: int = 64
    bootstrap_refits: int = 10000
    bootstrap_seed: int = 920001
    candidate_rule: str = 'feature_tradeoffs_v2'

    def __post_init__(self):
        if self.candidate_rule not in ('feature_tradeoffs_v2','edge_alternatives_v1'):
            raise ValueError('unsupported candidate rule')
        # JSON arrays must not reach NumPy indexing as lists: [r,c] selects
        # entire rows, while (r,c) selects the intended single endpoint cell.
        for name in ('shape','development_seeds','final_seeds'):
            try:
                object.__setattr__(self,name,tuple(getattr(self,name)))
            except TypeError as exc:
                raise ValueError(f'{name} must be a sequence') from exc
        if (len(self.shape)!=2 or any(not _integer(x) or x<1 for x in self.shape)
                or self.headings != 4 or isinstance(self.headings,bool)):
            raise ValueError('study needs a positive 2D shape and four headings')
        object.__setattr__(self,'start',_coordinate(self.start,(*self.shape,4),'start'))
        object.__setattr__(self,'goal',_coordinate(self.goal,self.shape,'goal'))
        for name in ('max_attempts','horizon','choices_per_layout','bootstrap_refits'):
            value=getattr(self,name)
            if not _integer(value) or value<1:
                raise ValueError(f'{name} must be a positive integer')
        if not _integer(self.bootstrap_seed) or self.bootstrap_seed<0:
            raise ValueError('bootstrap seed must be nonnegative integer')
        if (not math.isfinite(self.obstacle_probability) or not 0<=self.obstacle_probability<=1
                or not math.isfinite(self.terminal_penalty) or self.terminal_penalty<0
                or not math.isfinite(self.d_safe) or self.d_safe<0):
            raise ValueError('invalid probability, terminal penalty or clearance threshold')
        for seeds in (self.development_seeds,self.final_seeds):
            if (not seeds or any(not _integer(s) or s<0 for s in seeds)
                    or len(set(seeds))!=len(seeds)):
                raise ValueError('stage seeds must be distinct nonnegative integers')
        if set(self.development_seeds)&set(self.final_seeds):
            raise ValueError('development and final seeds must be disjoint')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _text(value):
    return json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n'


def _write(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(_text(value),encoding='utf-8')
    temporary.replace(path)


def load_frozen_models(recovery_path: Path) -> list[dict]:
    if _sha(recovery_path)!=RECOVERY_SHA256:
        raise ValueError('recovery checkpoint artifact hash mismatch')
    report=json.loads(Path(recovery_path).read_text(encoding='utf-8'))
    models=[]
    for seed in (11,29,47,83,101):
        cells=[c for c in report['cells'] if c['seed']==seed and c['n']==2048]
        if len(cells)!=1:
            raise ValueError('checkpoint needs exactly one primary model per training seed')
        fit=cells[0]['point_fit']
        if not fit['success'] or fit['l2']!=0 or fit['weights'][0]!=1.:
            raise ValueError('checkpoint models must be successful lambda-zero primary fits')
        models.append({'name':f'learned_{seed}','training_seed':seed,'weights':fit['weights'],
                       'n':2048,'l2':0.,'checkpoint':CHECKPOINT})
    return models


def _all_models():
    return [{'name':'fixed','weights':[1.,1.,1.]},
            {'name':'soft_distance','weights':[1.,0.,0.]},
            {'name':'oracle','weights':[1.,.7,1.2]},
            *load_frozen_models(ROOT/'results/recovery.json')]


def generate_recorded_layout(seed: int, config: OfferedRouteConfig) -> tuple[GridMDP,int]:
    """Same RNG sequence and connectivity-only rejection as generate_layout."""
    if not _integer(seed) or seed<0:
        raise ValueError('layout seed must be a nonnegative integer')
    rng=np.random.default_rng(int(seed))
    for attempt in range(1,config.max_attempts+1):
        occupied=rng.random(config.shape)<config.obstacle_probability
        occupied[config.start[:2]]=False
        occupied[config.goal]=False
        env=GridMDP(occupied,config.start,config.goal,config.headings)
        if env.goal_reachable():
            return env,attempt
    raise ValueError(f'connectivity rejection exhausted {config.max_attempts} attempts for seed {seed}')


def _manifest(config,stage):
    source=Path(__file__).resolve().parent
    names=('env','features','demos','soft_dp','verification','routes','route_choice','route_metrics','route_design','offered_routes')
    hashes={name:_sha(source/f'{name}.py') for name in names}
    hashes['cli']=_sha(ROOT/'scripts/run_offered_routes.py')
    value={'protocol_version':2,'stage':stage,'configuration':asdict(config),
           'selected_models':_all_models(),'checkpoint':CHECKPOINT,
           'recovery_sha256':RECOVERY_SHA256,'source_hashes':hashes,
           'versions':{'python':platform.python_version(),'numpy':np.__version__,'torch':str(torch.__version__)},
           'protocol':{'task':'binary completed offered-route choice','temperature':1.,
                        'candidate_order':['distance','action_count','action_tuple'],
                        'candidate_rule':config.candidate_rule,
                        'candidate_library':'six lexicographic feature-priority extrema',
                        'pair_selection':'nondominated balanced range-normalized opposing features',
                        'shortest_baseline':'shortest offered route, then action count and action tuple',
                        'development_design_gate':{'minimum_varying_fraction':.2,'normalized_min_eigenvalue':.01,
                                                   'normalized_max_condition':100.,'minimum_profile_disagreement':.2},
                       'candidate_ids':'lexicographic unpadded action sequence',
                       'prediction_tie_tolerance':1e-12,'prediction_tie_choice':'shortest route A',
                       'choice_seed_sequence_prefix':420001,'bootstrap_percentiles':[1.25,98.75],
                       'bootstrap_unit':'layout','bootstrap_percentile_method':'linear',
                       'oracle_completion_threshold':.90,'torch_threads':1}}
    return json.loads(_text(value))


def _prepare_output(output,identity):
    """Only rerun verified matching studies; never overwrite unrelated work."""
    allowed={'identity.json','preflight.json','study.json','integrity.json','final_manifest.json'}
    if output.exists() and any(p.name not in allowed for p in output.iterdir()):
        raise ValueError('output directory contains unrelated files')
    existing=output/'identity.json'
    if existing.exists():
        if json.loads(existing.read_text(encoding='utf-8'))!=identity:
            raise ValueError('incompatible study identity/configuration/source versions')
        integrity=output/'integrity.json'
        if not integrity.exists():
            raise ValueError('corrupt or interrupted output: integrity record missing; use a new directory')
        recorded=json.loads(integrity.read_text(encoding='utf-8'))
        if set(recorded)!={'preflight.json','study.json'}:
            raise ValueError('corrupt output integrity record')
        for name,digest in recorded.items():
            if not (output/name).is_file() or _sha(output/name)!=digest:
                raise ValueError('corrupt study artifact; use a new output directory')
    elif output.exists() and any(output.iterdir()):
        raise ValueError('output directory is not an empty study directory')
    output.mkdir(parents=True,exist_ok=True)


def _completion(env,solution):
    mass=torch.zeros(len(env.states),dtype=torch.float64)
    mass[env.start_state]=1.
    max_error=0.
    for log_policy in solution.log_policy:
        edge_mass=mass[:,None]*log_policy.exp()
        mass=torch.zeros_like(mass).scatter_add(0,env.next_state.reshape(-1),edge_mass.reshape(-1))
        max_error=max(max_error,abs(mass.sum().item()-1.))
    if max_error>1e-10:
        raise ValueError('unrestricted distribution mass error exceeds tolerance')
    return mass[env.is_goal].sum().item(),max_error


def _preflight(seed,config,models):
    row={'seed':int(seed),'status':'failed'}
    context=None
    try:
        env,attempts=generate_recorded_layout(seed,config)
        row.update(occupied=env.occupied.tolist(),generation_attempts=attempts)
        features=feature_table(env,config.d_safe)
        if config.candidate_rule=='feature_tradeoffs_v2':
            pair,construction=tradeoff_candidate_pair(env,features,config.horizon)
            row['candidate_construction']=construction
        else:
            pair=candidate_pair(env,config.horizon)
        totals=route_features(env,features,pair)
        row.update(routes=[asdict(r) for r in pair.routes],shortest_index=pair.shortest_index,
                   feature_totals=totals.tolist(),feature_difference=(totals[1]-totals[0]).tolist(),
                   feature_identical=bool(torch.equal(totals[0],totals[1])))
        diagnostics={'completion':{},'mass_error':{},'model_errors':{}}
        row['diagnostics']=diagnostics
        with torch.no_grad():
            for model in models:
                try:
                    w=torch.tensor(model['weights'],dtype=torch.float64)
                    solution=plan(env,features,w,config.horizon,config.terminal_penalty)
                    probability,error=_completion(env,solution)
                    diagnostics['completion'][model['name']]=probability
                    diagnostics['mass_error'][model['name']]=error
                except ValueError as exc:
                    diagnostics['model_errors'][model['name']]=str(exc)
            margin=terminal_margin_diagnostic(env,features,torch.tensor([1.,.7,1.2],dtype=torch.float64),
                                              config.horizon,config.terminal_penalty)
        diagnostics['terminal_margin']=margin
        if diagnostics['model_errors']:
            raise ValueError('one or more models failed numerical validation')
        if not margin['passed']:
            raise ValueError('reference terminal-margin gate failed')
        if diagnostics['completion']['oracle']<.90:
            raise ValueError('oracle completion gate failed')
        row['status']='preflight_passed'
        context=(env,pair,totals)
    except ValueError as exc:
        row['reason']=str(exc)
        if isinstance(exc,CandidateConstructionError):
            row['candidate_construction']=exc.metadata
        if 'generation_attempts' not in row:
            row['generation_attempts']=config.max_attempts
    return row,context


def _evaluate(row,context,models,config):
    _,pair,totals=context
    oracle_costs=totals@torch.tensor([1.,.7,1.2],dtype=torch.float64)
    oracle_lp=(-oracle_costs).log_softmax(0)
    labels=draw_choices(oracle_lp,row['seed'],config.choices_per_layout)
    row['labels']=labels.tolist()
    row['models']={}
    row['model_metrics']={}
    choices=[{'name':'shortest','weights':None},*models]
    for model in choices:
        name=model['name']
        if name=='shortest':
            lp=None; prediction=pair.shortest_index; tied=False
        else:
            w=torch.tensor(model['weights'],dtype=torch.float64)
            lp=choice_log_probs(totals,w)
            costs=totals@w
            prediction=predict_route(costs,pair.shortest_index)
            tied=abs((costs[0]-costs[1]).item())<=1e-12
        row['models'][name]={'prediction':prediction,'tie':tied,
                             'log_probabilities':lp.tolist() if lp is not None else None,
                             'probabilities':lp.exp().tolist() if lp is not None else None}
        row['model_metrics'][name]=choice_metrics(lp,prediction,labels,oracle_lp,oracle_costs)
    row['status']='passed'


def _run(config,output,stage,identity):
    output=Path(output)
    _prepare_output(output,identity)
    torch.set_num_threads(1)
    models=identity['selected_models']
    seeds=config.development_seeds if stage=='development' else config.final_seeds
    rows=[]; contexts=[]
    for seed in seeds:
        row,context=_preflight(seed,config,models)
        rows.append(row); contexts.append(context)
        print(f'{stage} seed={seed}: {row["status"]}' +
              (f' ({row["reason"]})' if 'reason' in row else ''),flush=True)
    design_audit=(audit_choice_design(rows,len(seeds)) if config.candidate_rule=='feature_tradeoffs_v2'
                  else {'passed':True,'scope':'legacy reference, not revised trade-off study'})
    design_ok=(stage!='development' or design_audit['passed'])
    preflight={'identity':identity,'layouts':rows,'design_audit':design_audit}
    # Serialize before mutating rows during evaluation; this evidence precedes labels.
    if (output/'preflight.json').exists():
        previous=json.loads((output/'preflight.json').read_text(encoding='utf-8'))
        if previous!=json.loads(_text(preflight)):
            raise ValueError('deterministic preflight rerun disagrees with recorded evidence')
    _write(output/'identity.json',identity)
    _write(output/'preflight.json',preflight)
    passed=sum(r['status']=='preflight_passed' for r in rows)
    if passed==len(seeds) and design_ok:
        for row,context in zip(rows,contexts):
            _evaluate(row,context,models,config)
    else:
        for row in rows:
            if row['status']=='preflight_passed':
                row.update(status='not_evaluated',reason='study halted before labels: layout feasibility or development design audit failed')
    report={'identity':identity,'stage':stage,'status':'passed' if passed==len(seeds) and design_ok else 'halted',
            'design_audit':design_audit,
            'preflight_passed_layouts':passed,'preflight_failed_layouts':len(seeds)-passed,
            'layouts':rows,'summary':summarize_layouts(rows,seeds,config.bootstrap_refits,config.bootstrap_seed)}
    # Normalize tuple/list representation so returned and persisted records agree.
    report=json.loads(_text(report))
    if (output/'study.json').exists():
        previous=json.loads((output/'study.json').read_text(encoding='utf-8'))
        if previous!=report:
            raise ValueError('deterministic study rerun disagrees with recorded results')
    _write(output/'study.json',report)
    _write(output/'integrity.json',{name:_sha(output/name) for name in ('preflight.json','study.json')})
    return report


def run_development(config: OfferedRouteConfig, output: Path) -> dict:
    return _run(config,output,'development',_manifest(config,'development'))


def seal_manifest(config: OfferedRouteConfig, output: Path) -> dict:
    output=Path(output)
    identity=_manifest(config,'development')
    _prepare_output(output,identity)
    path=output/'study.json'
    if not path.exists() or json.loads(path.read_text(encoding='utf-8'))['status']!='passed':
        raise ValueError('all development layouts must pass before sealing final manifest')
    manifest=_manifest(config,'final')
    destination=output/'final_manifest.json'
    if destination.exists() and json.loads(destination.read_text(encoding='utf-8'))!=manifest:
        raise ValueError('existing final manifest differs; never overwrite approved identities')
    _write(destination,manifest)
    return manifest


def run_final(config: OfferedRouteConfig, output: Path,
              approved_manifest: Path, approved_hash: str) -> dict:
    if (not approved_manifest or not approved_hash or not Path(approved_manifest).is_file()
            or _sha(approved_manifest)!=approved_hash.lower()):
        raise ValueError('explicit approved manifest and matching SHA256 are required')
    manifest=json.loads(Path(approved_manifest).read_text(encoding='utf-8'))
    if manifest!=_manifest(config,'final'):
        raise ValueError('approved manifest does not match current code/configuration/models/versions')
    return _run(config,output,'final',manifest)
