from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from wnp.env import generate_layout
from wnp.offered_routes import (
    OfferedRouteConfig, load_frozen_models, generate_recorded_layout,
    run_development, run_final, seal_manifest,
)


RECOVERY = Path(__file__).resolve().parents[1] / 'results/recovery.json'


def _small():
    return replace(OfferedRouteConfig(), shape=(2,2), start=(0,0,0), goal=(1,1),
                   obstacle_probability=0., horizon=8, terminal_penalty=30.,
                   development_seeds=(1,2), final_seeds=(3,4), bootstrap_refits=20,
                   candidate_rule='edge_alternatives_v1')


def test_frozen_checkpoint_models_are_exactly_the_five_declared_primary_fits(tmp_path):
    models=load_frozen_models(RECOVERY)
    assert [m['training_seed'] for m in models]==[11,29,47,83,101]
    saved=json.loads(RECOVERY.read_text())
    for model in models:
        cell=next(c for c in saved['cells'] if c['seed']==model['training_seed'] and c['n']==2048)
        assert model['weights']==cell['point_fit']['weights']
    bad=tmp_path/'bad.json'
    bad.write_text(RECOVERY.read_text()+' ')
    with pytest.raises(ValueError, match='checkpoint'):
        load_frozen_models(bad)


def test_generator_matches_existing_draws_and_records_attempts():
    cfg=replace(_small(),shape=(4,4),goal=(3,3),obstacle_probability=.35)
    got,attempts=generate_recorded_layout(17,cfg)
    expected=generate_layout(17,cfg.shape,cfg.obstacle_probability,4,cfg.max_attempts,
                             start=cfg.start,goal=cfg.goal)
    np.testing.assert_array_equal(got.occupied,expected.occupied)
    rng=np.random.default_rng(17)
    for _ in range(attempts):
        occupied=rng.random(cfg.shape)<.35
        occupied[cfg.start[:2]]=occupied[cfg.goal]=False
    np.testing.assert_array_equal(got.occupied,occupied)
    assert attempts>=1
    with pytest.raises(ValueError, match='connect'):
        generate_recorded_layout(17,replace(cfg,obstacle_probability=1.,max_attempts=2))


def test_json_round_trip_keeps_generator_and_approved_identity_identical():
    from wnp.offered_routes import _manifest
    original=OfferedRouteConfig()
    restored=OfferedRouteConfig(**json.loads(json.dumps(asdict(original))))
    assert _manifest(original,'final')==_manifest(restored,'final')
    first,first_attempts=generate_recorded_layout(10000,original)
    second,second_attempts=generate_recorded_layout(10000,restored)
    assert first_attempts==second_attempts==3
    np.testing.assert_array_equal(first.occupied,second.occupied)
    assert original==restored
    for name in ('shape','start','goal','development_seeds','final_seeds'):
        assert isinstance(getattr(restored,name),tuple)


def test_development_persists_all_models_shared_labels_and_reproduces(tmp_path,monkeypatch):
    import wnp.irl
    def forbidden(*args,**kwargs): raise AssertionError('evaluation must not fit')
    monkeypatch.setattr(wnp.irl,'fit',forbidden)
    cfg=_small()
    report=run_development(cfg,tmp_path)
    assert report['status']=='passed' and report['stage']=='development'
    assert report['summary']['requested_layouts']==2
    for row in report['layouts']:
        assert row['seed'] in (1,2) and len(row['labels'])==64
        assert set(row['models'])=={'fixed','shortest','soft_distance','oracle',
                                    'learned_11','learned_29','learned_47','learned_83','learned_101'}
        assert row['models']['shortest']['log_probabilities'] is None
        assert row['diagnostics']['completion']['oracle']>=.9
        assert row['diagnostics']['terminal_margin']['passed']
        assert len(row['routes'][0]['actions'])==8
    before=(tmp_path/'study.json').read_bytes()
    assert run_development(cfg,tmp_path)==report
    assert (tmp_path/'study.json').read_bytes()==before
    manifest=seal_manifest(cfg,tmp_path)
    assert manifest['stage']=='final'
    assert manifest['configuration']['final_seeds']==[3,4]
    assert not any(row['seed'] in (3,4) for row in report['layouts'])
    assert set(manifest['source_hashes']) >= {'env','soft_dp','routes','route_choice','offered_routes'}


@pytest.mark.parametrize('changes',[
    {'shape':(1,3),'goal':(0,2)}, {'horizon':2}, {'terminal_penalty':0.},
    {'terminal_penalty':1000.}, {'obstacle_probability':1.},
])
def test_failed_preflight_halts_all_labels_and_retains_denominators(tmp_path,monkeypatch,changes):
    import wnp.offered_routes as runner
    def forbidden(*args,**kwargs): raise AssertionError('no labels before ALL gates pass')
    monkeypatch.setattr(runner,'draw_choices',forbidden)
    cfg=replace(_small(),**changes)
    report=run_development(cfg,tmp_path)
    assert report['status']=='halted'
    assert report['summary']['requested_layouts']==2
    assert report['summary']['failed_layouts']==2
    assert report['summary']['comparisons'] is None
    assert all('labels' not in row for row in report['layouts'])
    assert (tmp_path/'preflight.json').is_file()
    with pytest.raises(ValueError,match='development'):
        seal_manifest(cfg,tmp_path)


def test_one_failure_prevents_choices_even_on_other_passing_layout(tmp_path,monkeypatch):
    import wnp.offered_routes as runner
    original=runner.generate_recorded_layout
    def generator(seed,cfg):
        if seed==2: raise ValueError('injected connectivity exhaustion')
        return original(seed,cfg)
    monkeypatch.setattr(runner,'generate_recorded_layout',generator)
    monkeypatch.setattr(runner,'draw_choices',lambda *a,**k: pytest.fail('premature labels'))
    report=run_development(_small(),tmp_path)
    assert report['status']=='halted'
    assert report['summary']['requested_layouts']==2
    assert not any('labels' in row for row in report['layouts'])


def test_final_gate_requires_approved_unchanged_manifest_before_generation(tmp_path,monkeypatch):
    import wnp.offered_routes as runner
    cfg=_small(); dev=tmp_path/'dev'; final=tmp_path/'final'
    run_development(cfg,dev); seal_manifest(cfg,dev)
    path=dev/'final_manifest.json'; digest=hashlib.sha256(path.read_bytes()).hexdigest()
    def forbidden(*a,**k): raise AssertionError('final generation before valid approval')
    monkeypatch.setattr(runner,'generate_recorded_layout',forbidden)
    for approved in ('','wrong'):
        with pytest.raises(ValueError,match='approv'):
            run_final(cfg,final,path,approved)
    with pytest.raises(ValueError,match='manifest'):
        run_final(replace(cfg,horizon=9),final,path,digest)
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='approv'):
        run_final(cfg,final,path,digest)
    assert not final.exists()


def test_output_collisions_changed_config_and_corruption_rejected(tmp_path):
    cfg=_small(); run_development(cfg,tmp_path)
    with pytest.raises(ValueError,match='incompatible'):
        run_development(replace(cfg,horizon=9),tmp_path)
    report=json.loads((tmp_path/'study.json').read_text())
    report['layouts'][0]['labels'][0]=7
    (tmp_path/'study.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='corrupt'):
        run_development(cfg,tmp_path)
    foreign=tmp_path/'foreign'; foreign.mkdir(); (foreign/'keep.txt').write_text('user work')
    with pytest.raises(ValueError,match='output'):
        run_development(cfg,foreign)
    assert (foreign/'keep.txt').read_text()=='user work'


def test_manifest_rejects_changed_dependency_or_source_identity(tmp_path,monkeypatch):
    import wnp.offered_routes as runner
    cfg=_small(); dev=tmp_path/'dev'; run_development(cfg,dev); seal_manifest(cfg,dev)
    manifest=dev/'final_manifest.json'; digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
    monkeypatch.setattr(runner.platform,'python_version',lambda:'changed dependency')
    with pytest.raises(ValueError,match='manifest'):
        run_final(cfg,tmp_path/'final',manifest,digest)
    assert not (tmp_path/'final').exists()


def test_valid_approved_manifest_uses_final_seeds_only(tmp_path):
    cfg=_small(); dev=tmp_path/'dev'; run_development(cfg,dev); seal_manifest(cfg,dev)
    manifest=dev/'final_manifest.json'; digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
    report=run_final(cfg,tmp_path/'final',manifest,digest)
    assert report['stage']=='final' and report['status']=='passed'
    assert [row['seed'] for row in report['layouts']]==[3,4]


def test_preflight_drift_does_not_overwrite_existing_evidence(tmp_path,monkeypatch):
    import wnp.offered_routes as runner
    cfg=_small(); run_development(cfg,tmp_path)
    before={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    original=runner._preflight
    def drift(seed,config,models):
        row,context=original(seed,config,models)
        row['generation_attempts']+=1
        return row,context
    monkeypatch.setattr(runner,'_preflight',drift)
    with pytest.raises(ValueError,match='rerun'):
        run_development(cfg,tmp_path)
    assert {p.name:p.read_bytes() for p in tmp_path.iterdir()}==before


@pytest.mark.parametrize('changes',[
    {'development_seeds':(True,)},{'development_seeds':(1,1)}, {'horizon':True},
    {'terminal_penalty':float('inf')},{'obstacle_probability':-1.},
    {'development_seeds':(20000,)},{'final_seeds':(10000,)},
])
def test_invalid_or_overlapping_stage_configuration_rejected(changes):
    with pytest.raises(ValueError): replace(OfferedRouteConfig(),**changes)


def test_default_revision_rejects_clearance_only_design_before_labels(tmp_path,monkeypatch):
    import wnp.offered_routes as runner
    cfg=replace(OfferedRouteConfig(),development_seeds=(1,2))
    assert cfg.candidate_rule=='feature_tradeoffs_v2'
    def degenerate(seed,config,models):
        return dict(seed=seed,status='preflight_passed',feature_difference=[0.,0.,float(seed)]),None
    monkeypatch.setattr(runner,'_preflight',degenerate)
    monkeypatch.setattr(runner,'draw_choices',lambda *a,**k: pytest.fail('uninformative study sampled labels'))
    report=run_development(cfg,tmp_path)
    assert report['status']=='halted' and not report['design_audit']['passed']
    assert report['preflight_failed_layouts']==0
    assert report['summary']['comparisons'] is None
    assert not any('labels' in row for row in report['layouts'])
    with pytest.raises(ValueError,match='development'):
        seal_manifest(cfg,tmp_path)


def test_revised_manifest_cannot_reuse_withdrawn_old_approval(tmp_path):
    old=Path(__file__).resolve().parents[1]/'results/offered_routes_dev/final_manifest.json'
    if not old.exists(): pytest.skip('historical manifest unavailable')
    digest=hashlib.sha256(old.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='manifest'):
        run_final(OfferedRouteConfig(),tmp_path/'no_final',old,digest)
    assert not (tmp_path/'no_final').exists()


def test_failed_revised_preflight_saves_every_candidate_recipe(tmp_path):
    cfg=replace(_small(),candidate_rule='feature_tradeoffs_v2')
    report=run_development(cfg,tmp_path)
    assert report['status']=='halted'
    for row in report['layouts']:
        assert len(row['candidate_construction']['recipes'])==6
        assert row['candidate_construction']['library']
        assert 'labels' not in row
    saved=json.loads((tmp_path/'preflight.json').read_text())
    assert len(saved['layouts'][0]['candidate_construction']['recipes'])==6
