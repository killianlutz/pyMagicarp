from quantum import *

@jax.jit
def schrodinger(U, args):
    g, H = args
    cg = jnp.einsum('ij, jk, kl -> il', U, g, dagger(U))
    dU = jnp.zeros_like(U)
    init = (dU, cg)

    def sum_hamiltonians(carry, h):
        dU, cg = carry
        dotp = su_coeff(cg, h)
        new_dU = dU - 1j * dotp * h

        new_carry = (new_dU, cg)
        return new_carry, None

    carry, _ = jax.lax.scan(sum_hamiltonians, init, H)
    dU = jnp.einsum('ij, jk -> ik', carry[0], U)
    return dU

def odesolve(f, y0, t0, t1, nt, args):
    dt = (t1 - t0)/nt
    def rungekutta(_, y):
        k1 = f(y            , args)
        k2 = f(y + 0.5*dt*k1, args)
        k3 = f(y + 0.5*dt*k2, args)
        k4 = f(y +     dt*k3, args)
        return y + dt*(k1 + 2*k2 + 2*k3 + k4)/6

    return jax.lax.fori_loop(0, nt, rungekutta, y0)