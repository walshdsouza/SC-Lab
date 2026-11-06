import numpy as np
 
# ---------------- Objective and its derivatives ----------------
def f(x):
    return x * np.sin(10 * np.pi * x) + 2
 
def df(x):    # f'(x) = sin(10πx) + 10πx cos(10πx)
    return np.sin(10 * np.pi * x) + 10 * np.pi * x * np.cos(10 * np.pi * x)
 
def d2f(x):   # f''(x) = 20π cos(10πx) - 100π² x sin(10πx)
    return (20 * np.pi * np.cos(10 * np.pi * x)
            - 100 * np.pi ** 2 * x * np.sin(10 * np.pi * x))
 
 
# ---------------- True local extrema of f on [lo, hi] ----------------
def true_extrema(lo, hi, n_grid=200001):
    """Roots of f'(x)=0 (bracket on a fine grid, refine by bisection),
    typed by the sign of f''(x)."""
    xs = np.linspace(lo, hi, n_grid)
    g = df(xs)
    idx = np.where(g[:-1] * g[1:] < 0)[0]
    a, b = xs[idx], xs[idx + 1]
    for _ in range(60):                       # vectorised bisection
        m = (a + b) / 2
        left = df(a) * df(m) <= 0
        b = np.where(left, m, b)
        a = np.where(left, a, m)
    roots = (a + b) / 2
    kind = np.where(d2f(roots) < 0, "Local Maximum", "Local Minimum")
    return roots, kind
 
 
def classify(x, ext_x, ext_kind, tol=1e-2):
    """Local max/min status of a particle, using the derivative test:
    a particle is *at* an extremum if it lies within `tol` of a root of f'.
    Otherwise it is 'Neither' and we report the nearest extremum it is heading to."""
    j = np.argmin(np.abs(ext_x - x))
    if abs(ext_x[j] - x) <= tol:
        return ext_kind[j], j
    return "Neither", j
 
 
# ---------------- PSO ----------------
def pso(n_particles=10, max_iter=10, lo=-1.0, hi=2.0,
        w=0.7, c1=1.5, c2=1.5, seed=42, tol=1e-2):
    rng = np.random.default_rng(seed)
    vmax = 0.2 * (hi - lo)
    ext_x, ext_kind = true_extrema(lo, hi)
 
    x = rng.uniform(lo, hi, n_particles)
    v = rng.uniform(-0.1, 0.1, n_particles)
    fit = f(x)
 
    pbest_x, pbest_f = x.copy(), fit.copy()
    g = np.argmax(pbest_f)
    gbest_x, gbest_f = pbest_x[g], pbest_f[g]
 
    hdr = (f"{'P':>2} {'x':>10} {'v':>10} {'f(x)':>10} {'pbest_x':>10} "
           f"{'pbest_f':>10} {'f\'(x)':>10}  Status")
 
    def show(title, it):
        print(f"\n{'=' * 98}\n{title}\n{'=' * 98}\n{hdr}")
        for i in range(n_particles):
            status, j = classify(x[i], ext_x, ext_kind, tol)
            if status == "Neither":
                status += (f"  (nearest: {ext_kind[j].split()[1]} "
                           f"at x={ext_x[j]:.4f})")
            print(f"{i + 1:>2} {x[i]:>10.6f} {v[i]:>10.6f} {fit[i]:>10.6f} "
                  f"{pbest_x[i]:>10.6f} {pbest_f[i]:>10.6f} "
                  f"{df(x[i]):>10.4f}  {status}")
        print(f"-> gbest x = {gbest_x:.8f}, gbest f = {gbest_f:.8f}")
 
    show("Iteration 0 (initial swarm)", 0)
    history = [gbest_f]
 
    for it in range(1, max_iter + 1):
        r1, r2 = rng.random(n_particles), rng.random(n_particles)
        v = w * v + c1 * r1 * (pbest_x - x) + c2 * r2 * (gbest_x - x)
        v = np.clip(v, -vmax, vmax)
        x = x + v
 
        # boundary handling: clamp and kill velocity so particles don't stick
        out = (x < lo) | (x > hi)
        x = np.clip(x, lo, hi)
        v[out] = 0.0
        fit = f(x)
 
        better = fit > pbest_f
        pbest_x[better], pbest_f[better] = x[better], fit[better]
        g = np.argmax(pbest_f)
        if pbest_f[g] > gbest_f:
            gbest_x, gbest_f = pbest_x[g], pbest_f[g]
        history.append(gbest_f)
 
        show(f"Iteration {it}", it)
 
    # ---------------- Summary ----------------
    gm = ext_kind == "Local Maximum"
    true_best = ext_x[gm][np.argmax(f(ext_x[gm]))]
    print(f"\n{'=' * 98}\nFINAL RESULT\n{'=' * 98}")
    print(f"PSO best x        = {gbest_x:.10f}")
    print(f"PSO max fitness   = {gbest_f:.10f}")
    print(f"True global max   : x = {true_best:.10f}, f = {f(true_best):.10f}")
    print(f"Error in fitness  = {abs(f(true_best) - gbest_f):.3e}")
    print(f"Local maxima in [{lo}, {hi}] = {gm.sum()}, "
          f"local minima = {(~gm).sum()}")
    print("gbest fitness per iteration:",
          ", ".join(f"{h:.4f}" for h in history))
    return gbest_x, gbest_f
 
 
if __name__ == "__main__":
    pso(n_particles=10, max_iter=10, lo=-1, hi=2,
        w=0.7, c1=1.5, c2=1.5, seed=42, tol=1e-2)