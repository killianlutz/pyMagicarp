import jax.numpy as jnp
import jax
import numpy as np
import scipy.sparse as sparse

### quantum
def linear_graph(dim):
    return [(i, i + 1) for i in range(dim - 1)]

def TDgraph():
    graph = [
        [1, 2], [2, 3], [3, 4],
        [5, 6], [6, 7], [7, 8],
        [9, 10], [10, 11], [11, 12],
        [13, 14], [14, 15], [15, 16],  # rows
        [1, 5], [2, 6], [3, 7], [4, 8],
        [5, 9], [6, 10], [7, 11], [8, 12],
        [9, 13], [10, 14], [11, 15], [12, 16]  # cols
    ]
    return [[a - 1, b - 1] for (a, b) in graph]

def basis(dim):
    B = []
    for i in range(dim):
        for j in range(dim):
            b = np.zeros((dim, dim), dtype=np.complex64)
            b[i, j] = 1
            B.append(b)
            B.append(1j*b)

    return jnp.array(B)

def subasis(dim):
    # orthonormal
    basis = []

    for k in np.arange(dim - 1):
        Dk = np.zeros((dim, dim), dtype=jnp.complex64)
        for i in np.arange(k + 1):
            Dk[i, i] = 1
        Dk[k + 1, k + 1] = -(k + 1)
        Dk /= np.linalg.norm(Dk)
        basis.append(Dk)

    k = 0
    for i in np.arange(dim):
        for j in np.arange(i):
            Sk = np.zeros((dim, dim), dtype=jnp.complex64)
            Ak = np.zeros((dim, dim), dtype=jnp.complex64)

            Sk[i, j] = 1
            Sk[j, i] = 1
            Sk /= np.linalg.norm(Sk)

            Ak[i, j] = 1j
            Ak[j, i] = -1j
            Ak /= np.linalg.norm(Ak)

            basis.append(Sk)
            basis.append(Ak)

        k += 1

    return jnp.array(basis)

def ctrlbasis(dim, graph, sigma_z=False):
    # orthonormal
    H = []

    data_x = np.array([1, 1], dtype=np.complex64) / np.sqrt(2)
    data_y = np.array([-1j, 1j], dtype=np.complex64) / np.sqrt(2)
    data_z = np.array([1, -1], dtype=np.complex64) / np.sqrt(2)

    for idx_pair in graph:
        i, j = idx_pair

        for datum in (data_x, data_y):
            rows = np.array([i, j])
            cols = np.array([j, i])
            h = sparse.coo_array((datum, (rows, cols)), shape=(dim, dim)).toarray()
            H.append(h)

        if sigma_z:
            rows = np.array([i, j])
            cols = np.array([i, j])
            h = sparse.coo_array((data_z, (rows, cols)), shape=(dim, dim)).toarray()
            H.append(h)

    return jnp.array(H)

@jax.jit
def su_coeff(g, basis_vector):
    return jnp.real(trdot(g, basis_vector))

@jax.jit
def su_coeffs(g, basis):
    return jax.vmap(su_coeff, (None, 0))(g, basis)

@jax.jit
def su_matrix(v, basis):
    z = jax.vmap(lambda x, y: x*y, 0, 0)(v, basis)
    return jnp.sum(z, axis=0)

def sampleSU(dim, key):
    A = jax.random.normal(key, (dim, dim), dtype=jnp.complex64)
    Q, R = np.linalg.qr(A)

    r = jnp.diag(R)
    q = Q @ jnp.diag(r/jnp.abs(r))
    return jnp.array(toSU(q))

def toSU(q):
    dim = jnp.size(q, 0)
    phase = jnp.angle(jnp.linalg.det(q))
    return q * jnp.exp(-1j * phase / dim)

def dagger(a):
    return a.T.conj()

def trdot(a, b):
    return jnp.einsum('ij, ij -> ', a, b.conj())

@jax.jit
def infidelity(x, y):
    dim = jnp.size(x, 1)
    fidelity = jnp.abs(trdot(x, y)) / dim
    return jnp.abs(1 - fidelity)

def gatetime(v, su_basis, projector_range):
    return jnp.linalg.norm(project(v, su_basis, projector_range))

def project_(g, projector_range):
    return su_matrix(su_coeffs(g, projector_range), projector_range)

def project(v, su_basis, projector_range):
    g = su_matrix(v, su_basis)
    return su_coeffs(project_(g, projector_range), su_basis)