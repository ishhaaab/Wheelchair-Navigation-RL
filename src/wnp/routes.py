"""Preference-independent legal shortest-distance routes and offered pairs."""

from dataclasses import dataclass
import heapq
import math
from itertools import permutations

import numpy as np

from wnp.demos import Trajectory, _validate_horizon, validate_trajectories
from wnp.env import GridMDP
from wnp.soft_dp import _validate_features


@dataclass(frozen=True)
class RoutePair:
    routes: tuple[Trajectory, Trajectory]
    shortest_index: int


class CandidateConstructionError(ValueError):
    """Failed selection with the completed construction record still available."""

    def __init__(self, message, metadata):
        super().__init__(message)
        self.metadata={**metadata,'failure_reason':message}


def shortest_route(env: GridMDP, blocked_edges: frozenset = frozenset()) -> Trajectory | None:
    """Minimize distance, action count, then action tuple; stop at first arrival.

    Edge exclusions are undirected frozensets of two spatial coordinates. They
    affect only this search, not the environment used to score the routes.
    """
    successors, legal = env.next_state.numpy(), env.legal.numpy()
    best = {env.start_state: (0., 0, ())}
    pending = [(0., 0, (), env.start_state, (env.start_state,))]
    while pending:
        distance, count, actions, state, states = heapq.heappop(pending)
        if best[state] != (distance, count, actions):
            continue
        if bool(env.is_goal[state]):
            return Trajectory(states, actions, count)
        for action in range(3):
            if not legal[state, action]:
                continue
            successor = int(successors[state, action])
            source, target = env.states[state][:2], env.states[successor][:2]
            if source != target and frozenset((source, target)) in blocked_edges:
                continue
            key = (distance + math.dist(source, target), count + 1, actions + (action,))
            if successor not in best or key < best[successor]:
                best[successor] = key
                heapq.heappush(pending, (*key, successor, states + (successor,)))
    return None


def _route_key(env, route):
    return (sum(math.dist(env.states[s][:2], env.states[t][:2])
                for s, t in zip(route.states, route.states[1:])),
            len(route.actions), route.actions)


def candidate_pair(env: GridMDP, horizon: int) -> RoutePair:
    horizon = _validate_horizon(horizon)
    first = shortest_route(env)
    if first is None:
        raise ValueError('no route reaches the goal')
    edges = {frozenset((env.states[s][:2], env.states[t][:2]))
             for s, t in zip(first.states, first.states[1:])
             if env.states[s][:2] != env.states[t][:2]}
    alternatives = [route for edge in sorted(edges, key=lambda e: sorted(e))
                    if (route := shortest_route(env, frozenset((edge,)))) is not None
                    and route.actions != first.actions]
    if not alternatives:
        raise ValueError('no distinct edge-block alternative route')
    second = min(alternatives, key=lambda r: _route_key(env, r))
    ordered = sorted((first, second), key=lambda r: r.actions)
    if any(len(r.actions) > horizon for r in ordered):
        raise ValueError('candidate route exceeds horizon')
    padded = tuple(Trajectory(r.states + (r.states[-1],) * (horizon - len(r.actions)),
                              r.actions + (0,) * (horizon - len(r.actions)), r.arrival_step)
                   for r in ordered)
    validate_trajectories(env, padded)
    return RoutePair(padded, ordered.index(first))


def feature_priority_route(env, features, priority) -> Trajectory | None:
    """Shortest path under a lexicographic ordering of physical cost features.

    Nonnegative costs plus final action count eliminate zero-cost turn cycles.
    No scalar reward weights or demonstration data enter this search.
    """
    _validate_features(env,features)
    if tuple(sorted(priority))!=(0,1,2):
        raise ValueError('priority must permute the three feature indices')
    # Sum the stored float64 feature values EXACTLY as binary rationals. Rounded
    # float prefix ordering is not preserved by extension, which can cause a
    # label-setting search to discard a better secondary-objective path.
    values=features.detach().numpy()
    exact=np.empty(values.shape,dtype=object)
    for j in range(3):
        ratios={float(v):float(v).as_integer_ratio() for v in np.unique(values[:,:,j])}
        denominator=max(d for _,d in ratios.values())
        for value,(numerator,divisor) in ratios.items():
            exact[:,:,j][values[:,:,j]==value]=numerator*(denominator//divisor)
    costs=exact[:,:,list(priority)]
    successors,legal=env.next_state.numpy(),env.legal.numpy()
    initial=(0,0,0,0,())
    best={env.start_state:initial}
    pending=[(initial,env.start_state,(env.start_state,))]
    while pending:
        key,state,states=heapq.heappop(pending)
        if best[state]!=key:
            continue
        if bool(env.is_goal[state]):
            return Trajectory(states,key[-1],key[-2])
        for action in range(3):
            if not legal[state,action]:
                continue
            successor=int(successors[state,action])
            next_key=tuple(key[j]+int(costs[state,action,j]) for j in range(3))+(key[3]+1,key[4]+(action,))
            if successor not in best or next_key<best[successor]:
                best[successor]=next_key
                heapq.heappush(pending,(next_key,successor,states+(successor,)))
    return None


def select_tradeoff_pair(totals, action_sequences) -> tuple[int,int]:
    """Pick balanced opposing feature contrasts, never a winner under weights."""
    values=np.asarray(totals,dtype=np.float64)
    if (values.ndim!=2 or values.shape[1]!=3 or len(values)!=len(action_sequences)
            or not np.isfinite(values).all() or (values<0).any()):
        raise ValueError('invalid trade-off library feature totals')
    if len(values)<2:
        raise ValueError('no competing-cost trade-off pair')
    tolerance=1e-10
    ordered=sorted(range(len(values)),key=lambda i:action_sequences[i])
    nondominated=[]
    for i in ordered:
        if any(np.all(np.abs(values[i]-values[j])<=tolerance) for j in nondominated):
            continue
        if any(np.all(values[j]<=values[i]+tolerance) and np.any(values[j]<values[i]-tolerance)
               for j in range(len(values)) if j!=i):
            continue
        nondominated.append(i)
    spans=np.ptp(values,axis=0)
    spans[spans<=tolerance]=1.
    candidates=[]
    for offset,i in enumerate(nondominated):
        for j in nondominated[offset+1:]:
            difference=values[j]-values[i]
            if not ((difference>tolerance).any() and (difference< -tolerance).any()):
                continue
            scaled=difference/spans
            balance=min(scaled[scaled>0].sum(),-scaled[scaled<0].sum())
            candidates.append((-float(balance),-float(scaled@scaled),
                               action_sequences[i],action_sequences[j],i,j))
    if not candidates:
        raise ValueError('no competing-cost trade-off pair')
    chosen=min(candidates)
    return chosen[-2],chosen[-1]


def tradeoff_candidate_pair(env, features, horizon):
    """Select from six feature-priority extrema, not a full Pareto frontier."""
    horizon=_validate_horizon(horizon)
    routes={}
    recipes=[]
    for priority in permutations(range(3)):
        route=feature_priority_route(env,features,priority)
        status='unreachable' if route is None else 'over_horizon' if len(route.actions)>horizon else 'eligible'
        recipes.append({'priority':list(priority),'status':status,
                        'arrival_step':None if route is None else route.arrival_step,
                        'actions':None if route is None else list(route.actions)})
        if status=='eligible':
            routes[route.actions]=route
    library=[routes[actions] for actions in sorted(routes)]
    totals=np.array([features[list(r.states[:-1]),list(r.actions)].sum(0).tolist() for r in library])
    metadata={'rule':'feature_tradeoffs_v2','uses_preference_weights':False,'recipes':recipes,
              'library':[{'actions':list(r.actions),'feature_totals':total.tolist()}
                         for r,total in zip(library,totals)],
              'selected_library_indices':None,'selected_feature_totals':None}
    try:
        if len(library)<2:
            raise ValueError('no competing-cost trade-off pair within horizon')
        selected=select_tradeoff_pair(totals,[r.actions for r in library])
    except ValueError as exc:
        raise CandidateConstructionError(str(exc),metadata) from exc
    chosen=[library[i] for i in selected]
    ordered=sorted(chosen,key=lambda r:r.actions)
    shortest=min(range(2),key=lambda i:(_route_key(env,ordered[i])))
    padded=tuple(Trajectory(r.states+(r.states[-1],)*(horizon-len(r.actions)),
                            r.actions+(0,)*(horizon-len(r.actions)),r.arrival_step) for r in ordered)
    validate_trajectories(env,padded)
    metadata.update(selected_library_indices=list(selected),
                    selected_feature_totals=[totals[i].tolist() for i in selected])
    return RoutePair(padded,shortest),metadata
