"""WC2026 final model — market-anchored Dixon-Coles, EXPECTED-POINTS-maximizing picks.

Pipeline: de-vig DraftKings 3-way moneylines -> fit Dixon-Coles (lambda_home, lambda_away) to the
de-vigged 3-way + over/under + Asian-handicap spread -> pick the scoreline maximizing expected
points under the pool's 12/5/2 rule (evpick).

PICK RULE (evpick): maximize expected points. GENUINE EV ties (EV_EPS=0.0) are broken toward
higher exact-probability ONLY within the EV-max result class — a truly free tiebreak that raises
expected exact-count (the pool tiebreaker) without sacrificing any EV, and NEVER punts a clear
favorite to a draw/upset.

DESIGN HISTORY (adv R1-R8): an earlier variance-chasing layer (cross-class "coinflip" flips,
hybrid draw-lottery, P(win) simulation) was REJECTED — R6/R7 proved its apparent P(win) edge was a
self-consistency artifact (it loses to simpler rules under an independent DGP and collapses if any
rival also chases), and R8 showed its residual flip still punted a favorite. Conclusion: maximize
EXPECTED POINTS, full stop. No P(win) claim is made; the sim files are retained only as the
disproof of the variance thesis. EV_EPS=0.0 so the within-class exact-prob tiebreak fires only on
genuine EV ties — it gives up exactly 0 EV (a loose EV_EPS=0.2 was found 2026-06-15 to sacrifice
0.024 pts/run by picking higher-exact-prob-but-lower-EV scorelines, a within-class variance trade
that contradicts the EV-max objective; tightened to 0.0).
"""
import json, math
from itertools import product
RHO=-0.13; GRID_MAX=11; LAM_MAX=5.5
TOTAL_W=0.20; SUP_W=0.15; EV_EPS=0.0
# ulp-noise absorbers for evpick's deterministic tiebreak. At integer lambda Poisson P(k)=P(k-1)
# exactly, so two scorelines (e.g. (2,0)/(3,0)) can be EV- AND exact-prob-identical to within float
# rounding; scalar-sum (wc_model) vs einsum (fast_model) then round the last ulp differently and the
# old strict `>=mx` filter let that ulp luck pick the winner -> twin divergence. These tolerances make
# genuinely-equal candidates co-eligible so a deterministic GEOMETRY rule (fewest total goals, then
# lowest (hp,ap)) decides, identically in both models. They are FAR below any real EV/prob gap (integer
# points, probs ~O(0.1)), so they never trade away meaningful EV — unlike a loose EV_EPS (kept 0.0).
_EV_TOL=1e-9; _PROB_TOL=1e-12
def pois(n,L): return math.exp(-L)*L**n/math.factorial(n)
def tau(x,y,lh,la):
    if x==0 and y==0: return 1-lh*la*RHO
    if x==0 and y==1: return 1+lh*RHO
    if x==1 and y==0: return 1+la*RHO
    if x==1 and y==1: return 1-RHO
    return 1.0
def grid(lh,la):
    g={}; s=0.0
    for x,y in product(range(GRID_MAX),range(GRID_MAX)):
        p=max(0.0,tau(x,y,lh,la))*pois(x,lh)*pois(y,la); g[(x,y)]=p; s+=p
    return {k:v/s for k,v in g.items()}
def outcome(g):
    return (sum(p for (x,y),p in g.items() if x>y),
            sum(p for (x,y),p in g.items() if x==y),
            sum(p for (x,y),p in g.items() if x<y))
def rcls(h,a): return 'H' if h>a else ('D' if h==a else 'A')
def dec(ml): ml=float(ml); return 1+(ml/100 if ml>0 else 100/(-ml))
def devig(o):
    rh,rd,ra=1/dec(o['hml']),1/dec(o['dml']),1/dec(o['aml']); s=rh+rd+ra; return rh/s,rd/s,ra/s
def fit(o):
    tw,td,tl=devig(o); ou=o.get('ou') or 2.5
    sp=o.get('spread'); tgt=(-float(sp)) if sp is not None else None
    best=None; steps=[i/20 for i in range(2,int(LAM_MAX*20)+1)]
    for lh in steps:
        for la in steps:
            pw,pd,pl=outcome(grid(lh,la))
            e=(pw-tw)**2+(pd-td)**2+(pl-tl)**2+TOTAL_W*((lh+la-ou)/ou)**2
            if tgt is not None: e+=SUP_W*((lh-la-tgt)/2.0)**2
            if best is None or e<best[0]: best=(e,lh,la)
    return best[1],best[2]
def pts(hp,ap,H,A):
    # Scores the 90/120-minute result under 12/5/2 (works for group AND knockout regulation scores).
    if hp==H and ap==A: return 12
    return (5 if (hp>ap)==(H>A) and (hp<ap)==(H<A) else 0)+(2 if (hp==H or ap==A) else 0)
def pts_shootout(pred_winner, actual_winner):
    """Knockout ONLY: a SEPARATE +5 for the predicted penalty-shootout winner, which in Felix is only
    offered when you predicted a DRAW in regulation. Inert during the group stage (no shootouts).
    pred_winner/actual_winner are team identifiers (or None). Returns 5 if matched else 0."""
    return 5 if (pred_winner is not None and pred_winner == actual_winner) else 0
def evpick(g, eps=None):
    """Guarded EV-max: pick the expected-points-maximizing scoreline; break GENUINE EV ties
    (EV_EPS=0.0) toward higher exact-probability, but ONLY within the EV-max result class — so the
    exact-prob tiebreak is truly free (0 EV) and a clear favorite is NEVER punted to a draw/upset
    (adv R8 fix: the cross-class flip branch is removed; we maximize expected points, full stop).
    Residual EV/prob ties (e.g. (2,0) vs (3,0) at integer lambda) are resolved by a DETERMINISTIC,
    ulp-independent geometry rule — fewest total goals, then lowest (hp,ap) — so fast_model's einsum
    twin picks bit-identically (see _EV_TOL/_PROB_TOL note above)."""
    eps=EV_EPS if eps is None else eps
    cands=[]
    for hp,ap in product(range(7),range(7)):
        ev=sum(p*pts(hp,ap,H,A) for (H,A),p in g.items()); cands.append((ev,g.get((hp,ap),0.0),(hp,ap)))
    mx=max(c[0] for c in cands)
    ev_cls=rcls(*max(cands,key=lambda c:c[0])[2])
    allowed=[c for c in cands if c[0]>=mx-max(eps,_EV_TOL) and rcls(*c[2])==ev_cls]
    pmax=max(c[1] for c in allowed)
    tied=[c for c in allowed if c[1]>=pmax-_PROB_TOL]
    return min(tied,key=lambda c:(c[2][0]+c[2][1],c[2]))[2]
def fit_all(path=None):
    import os
    path = path or os.path.expanduser('~/wc-pool/data/fixtures.json')
    fixtures=json.load(open(path)); pre=[]
    for f in fixtures:
        o=f.get('odds')
        # require a PRE game with a complete 3-way line (partial None lines crash devig)
        if f['state']!='pre' or not o or any(o.get(k) is None for k in ('hml','dml','aml')): continue
        lh,la=fit(o); pre.append((f,grid(lh,la)))
    return pre

# ============================================================================
# FINAL DECISION (adversarial rounds R1-R8) — pick rule = GUARDED EV-MAX (evpick).
# The review converged AWAY from variance-chasing:
#   R3 argued a chaser should add variance.  R6 showed the supporting sim used a
#   homogeneous (clone) field.  R7 showed every variance strategy (guard-as-edge, hybrid,
#   modal-draws) is a SELF-CONSISTENCY ARTIFACT: under a DGP independent of the fitted grids
#   it LOSES to simpler rules, COLLAPSES if any rival also chases, and forfeits ~5.2 real EV
#   points.  R8 found the residual cross-class flip still punted one favorite (Egypt/Iran) and
#   removed it.
# Robust conclusion: maximize EXPECTED POINTS, full stop. evpick() does exactly this and breaks
# ties to higher exact-probability ONLY within the EV-max result class (free tiebreaker; never
# punts a favorite).  Result: 69 picks, exactly the EV-max scoreline per game (a draw appears
# only where the draw is itself the expected-points-maximizing result, e.g. a low-total coinflip).
# ============================================================================
