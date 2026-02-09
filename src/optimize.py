import jax
import jax.numpy as jnp

def descent(v0, p, routine, nmax=1_000, atol=1e-4):
    cost_fn, step_fn, merit_fn, line_search = routine

    def cond_fn(state):
        _, loss, i = state
        return jnp.logical_and(atol < loss, i < nmax)

    def body_fn(state):
        v, loss_ref, i = state

        dv = step_fn(v, p)
        step_size, merit = line_search(v, dv, loss_ref, p)
        v = v + step_size*dv
        loss = cost_fn(v, p)

        return v, loss, i+1

    loss = cost_fn(v0, p)
    init_state = (v0, loss, 0)
    return jax.lax.while_loop(cond_fn, body_fn, init_state)

def golden_section(f, a, b, atol=1e-1, nmax=100, ref_val=jnp.inf):
    g = (1 + jnp.sqrt(5)) / 2  # golden number
    c = b - (g - 1)*(b - a)
    d = a + (g - 1)*(b - a)
    fc = f(c)
    fd = f(d)

    def cond_fun(state):
        i, a, b = state[:3]
        return jnp.logical_and(i <= nmax, atol < b - a)

    def true_fun(state):
        i, a, b, c, d, left_value, _ = state

        b = d
        d = c
        c = b - (g - 1)*(b - a)
        right_value = left_value
        left_value = f(c)

        return i+1, a, b, c, d, left_value, right_value

    def false_fun(state):
        i, a, b, c, d, _, right_value = state

        a = c
        c = d
        d = a + (g - 1)*(b - a)
        left_value = right_value
        right_value = f(d)

        return i+1, a, b, c, d, left_value, right_value

    def body_fun(state):
        _, _, _, _, _, left_value, right_value = state
        return jax.lax.cond(left_value <= right_value, true_fun, false_fun, state)

    init_state = (1, a, b, c, d, fc, fd)
    state = jax.lax.while_loop(cond_fun, body_fun, init_state)
    a, b = state[1:3]

    # estimated minimizer
    midpoint = (a + b) / 2
    midpoint_val = f(midpoint)
    x = (midpoint, midpoint_val)

    def true_fn(x):
       return x

    def false_fn(_):
       midpoint = 1e-12
       return midpoint, f(midpoint)

    return jax.lax.cond(ref_val > midpoint_val, true_fn, false_fn, x)

