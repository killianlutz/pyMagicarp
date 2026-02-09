from src.ode import *
from src.optimize import *

def final_state(g, hp):
    U0, H, nt = hp
    args = (g, H)
    t0, t1 = (0.0, 1.0)
    return odesolve(schrodinger, U0, t0, t1, nt, args)

def model(v, args):
    hp, su_basis = args[:2]
    return final_state(su_matrix(v, su_basis), hp)

def model_coeffs(v, args):
    mat_basis = args[2]
    return su_coeffs(model(v, args), mat_basis)

def cost(v, q, args):
    return infidelity(model(v, args), q)

def metrics(v, q, args):
    hp, su_basis = args[:2]
    control_subspace = hp[1]

    loss = cost(v, q, args)
    time = gatetime(v, su_basis, control_subspace)
    return jnp.asarray([loss, time])

##########################
# GENERIC ROUTINE SETUP
##########################
def setup_routine(method, method_args, args, lsearch_args):
    @jax.jit
    def cost_fn(v, p):
        return cost(v, p, args)

    @jax.jit
    def step_fn(v, p):
        return method(v, p, args, method_args)

    @jax.jit
    def merit_fn(s, x, y, p):
        step_size = jnp.pow(10.0, s)
        return cost(x + step_size*y, p, args)

    @jax.jit
    def line_search(point, direction, ref_val, p):
        f = lambda s: merit_fn(s, point, direction, p)
        s, merit_val = golden_section(f, *lsearch_args, ref_val)
        step_size = jnp.pow(10.0, s)
        return step_size, merit_val

    return cost_fn, step_fn, merit_fn, line_search

####################################################
# NATURAL GRADIENT
####################################################

def natgrad(v, q, args, method_args):
    # min l(a(v))
    su_basis, mat_basis = args[-2:]
    reg = method_args[0]

    a = model(v, args)
    A = jax.jacfwd(model_coeffs, argnums=0)(v, args)
    Adag = dagger(A)

    M = Adag @ A + reg(v, q, args)*jnp.identity(len(su_basis))
    dl = su_coeffs(a - q, mat_basis)
    b = Adag @ (-dl)
    dv = jnp.linalg.solve(M, b)

    return dv/jnp.linalg.norm(dv)

####################################################
# NULL SPACE NATURAL GRADIENT
####################################################

def nullnat(v, q, args, method_args):
    # min l(a(v)) subject to c(v) = 0
    hp, su_basis, mat_basis = args
    control_subspace = hp[1]
    reg_a, reg_c = method_args

    #### linearize model v -> P(g(v))
    a = project(v, su_basis, control_subspace)
    A = jax.jacfwd(project, argnums=0)(v, su_basis, control_subspace)
    Adag = dagger(A)

    M = Adag @ A + reg_a(v, q, args)*jnp.identity(len(su_basis))
    dl = a # jax.grad(l)(a)
    b = Adag @ (-dl)
    du = jnp.linalg.solve(M, b)

    #### linearize constraint v -> u(1; v)
    c = su_coeffs(model(v, args) - q, mat_basis)
    C = jax.jacfwd(model_coeffs, argnums=0)(v, args)
    Cdag = dagger(C)

    M = C @ Cdag + reg_c(v, q, args)*jnp.identity(len(mat_basis))
    b = C @ du + c
    x = jnp.linalg.solve(M, b)
    dv = du - Cdag @ x

    return dv/jnp.linalg.norm(dv)


##### MUST IMPLEMENT MERIT FUNCTION BASED ON DUALITY
def setup_nullnat(method_args, args, lsearch_args):
    @jax.jit
    def cost_fn(v, p):
        return cost(v, p, args)

    @jax.jit
    def step_fn(v, p):
        return nullnat(v, p, args, method_args)

    @jax.jit
    def merit_fn(s, x, y, p):
        step_size = jnp.pow(10.0, s)
        return cost(x + step_size*y, p, args)

    ##### TO DO ////////////
    @jax.jit
    def line_search(point, direction, ref_val, p):
        #f = lambda s: merit_fn(s, point, direction, p)
        #s, merit_val = golden_section(f, *lsearch_args, ref_val)
        #step_size = jnp.pow(10.0, s)
        step_size = 1e-1
        merit_val = merit_fn(0.0, point, direction, p)
        return step_size, merit_val

    return cost_fn, step_fn, merit_fn, line_search