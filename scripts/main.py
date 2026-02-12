from src.methods import *

key = jax.random.PRNGKey(0)

##########################
# PARAMETERS
##########################
dim = 4
graph = linear_graph(dim)  # coupling btw control hamiltonians
su_basis = subasis(dim)  # traceless hermitian
mat_basis = basis(dim)  # matrices as real vector space
h = ctrlbasis(dim, graph, sigma_z=False)  # control hamiltonians
u0 = jnp.eye(dim, dtype=jnp.complex64)  # initial gate

nt = 100  # time grid points
nmax = 10_000  # gradient iterations
atol = 1e-4  # infidelity goal

# log-linesearch of stepsize in [10^a, 10^b]
a = -4.0
b = 1.0
ls_atol = 1e-1

hp = (u0, h, nt)
args = (hp, su_basis, mat_basis)
lsearch_args = (a, b, ls_atol)

##########################
# OPTIMIZER
##########################
# # least-squares Tikhonov regularization
# reg = lambda v, q, args: cost(v, q, args)
reg = lambda v, q, args: 1e-3
method_args = (reg,)
method = natgrad
routine = setup_routine(method, method_args, args, lsearch_args)


@jax.jit
def optimize(v, q):
    v1, loss, i = descent(v, q, routine, nmax, atol)
    return v1


@jax.jit
def map_optimize(v, q):
    return jax.vmap(optimize, (0, 2), 0)(v, q)


####################################################
# CONTROL PROBLEM // INITIALIZATION
####################################################
key1, key2, key3, key4 = jax.random.split(key, num=4)
su_dim = len(su_basis)
q = sampleSU(dim, key1)  # target
v0 = jax.random.normal(key2, su_dim)  # coefficients of g0

####################################################
# SOLVE
####################################################
v1 = optimize(v0, q)
g1 = su_matrix(v1, su_basis)
m = metrics(v1, q, args)
print(f"Infidelity: {m[0]:.1e} |||| Gate time: {m[1]:.2f}")

# save target and solution to a file
jnp.savez("./sims/example.npz", target=q, control=g1)

# solve simultaneously over multiple gates
# batch_size = 10
# keys = jax.random.split(key3, num=batch_size)
# qs = jnp.stack([sampleSU(dim, keys[i]) for i in range(batch_size)], axis=2)
# keys = jax.random.split(key4, num=batch_size)
# v0s = jnp.stack([jax.random.normal(keys[i], su_dim) for i in range(batch_size)], axis=0)
#
# v1s = map_optimize(v0s, qs)
# ms = jax.vmap(metrics, (0, 2, None), 0)(v1s, qs, args)
# print(ms)

#
# gs = jax.vmap(su_matrix, (0, None), 0)(v1s, su_basis)
# g0s = jax.vmap(su_matrix, (1, None), 0)(v0s, su_basis)
# jnp.savez("./sims/targets.npz", target=qs)
# jnp.savez("./sims/controls.npz", control=gs)
# jnp.savez("./sims/metrics.npz", metrics=ms)
# jnp.savez("./sims/initcontrols.npz", initcontrols=g0s)


