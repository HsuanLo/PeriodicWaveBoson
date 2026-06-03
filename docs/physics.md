# Physics And NN-VMC Method

This document describes the current bilayer boson system in this repository
and how the neural-network variational Monte Carlo (NN-VMC) code solves it.
It is written against the active code path:

```text
periodicwave/configs/bilayer_bosons.py
periodicwave/boson_network.py
periodicwave/pbc/bilayer_hamiltonian.py
periodicwave/mcmc.py
periodicwave/loss.py
periodicwave/train.py
```

The short version is: the code studies identical bosons confined to two
parallel two-dimensional layers. The bosons move continuously in the `x,y`
plane, carry fixed discrete layer labels, and interact through a z-polarized
dipole-dipole potential in a cell that is periodic in `x,y` and open along
`z`. A symmetric neural network represents the many-body wavefunction, and VMC
optimizes its parameters by sampling configurations from $|\psi|^2$ and
minimizing the sampled energy.

## 1. Current Physical System

The current run script sets:

```python
num_bosons = 14
layer_occupations = (7, 7)
layer_separation = 10.0
dipole_strength = 20.0
supercell_shape = "sq"
density_rs = 1.0
```

So the simulated system is a balanced bilayer:

- `N = 14` total bosons.
- `N_top = 7` bosons in the upper layer.
- `N_bottom = 7` bosons in the lower layer.
- The layer separation is `d = 10.0`.
- The dipole interaction strength is `D = 20.0`.
- The xy simulation cell is square.
- The density is parameterized by `r_s = 1.0`.

The layer assignment is stored as a fixed array:

```python
layer_assignment = [1, ..., 1, -1, ..., -1]
```

The code stores this array in the `spins` field of each walker, but in this
bosonic path `spins` does not mean fermion spin. It means layer label:

- $+1$ for the upper layer.
- $-1$ for the lower layer.

Only `x,y` coordinates are sampled. The `z` coordinate is reconstructed from
the layer label,

$$
z_i = \frac{d}{2}s_i, \qquad s_i \in \{+1, -1\}.
$$

Thus an upper-layer particle has $z = +d/2$, and a lower-layer particle has
$z = -d/2$. The layer populations do not fluctuate during a run.

## 2. Geometry And Density

The walker coordinates are two-dimensional:

$$
R = (\mathbf r_1, \ldots, \mathbf r_N), \qquad
\mathbf r_i = (x_i, y_i).
$$

For the current square cell, the run script constructs

$$
L = r_s \sqrt{\pi N}, \qquad \mathbf L = L I_2.
$$

so the cell area is

$$
A = L^2 = \pi r_s^2 N.
$$

This matches the two-dimensional Wigner-Seitz convention

$$
\frac{A}{N} = \pi r_s^2.
$$

For the current `N = 14` and `r_s = 1`, the square side length is

$$
L = \sqrt{14\pi}.
$$

The code also supports a triangular supercell option in the run script, but the
current system uses `supercell_shape = "sq"`.

The periodic boundary conditions are applied only in `x,y`. When a pair
displacement is needed, it is folded back into the first periodic cell with a
minimum-image operation:

$$
\Delta\mathbf r_{ij}
\longrightarrow
\text{minimum-image xy displacement}.
$$

The `z` direction is not periodic. The two layers sit at fixed height
$+d/2$ and $-d/2$.

## 3. Hamiltonian

The Hamiltonian has the standard first-quantized VMC form:

$$
H = T + V.
$$

In the units used by the code, the kinetic term is

$$
T = -\frac{1}{2}\sum_i
\left(
\frac{\partial^2}{\partial x_i^2}
+ \frac{\partial^2}{\partial y_i^2}
\right).
$$

There is no derivative with respect to `z` because the particles are not
sampled continuously in `z`; their layer labels are fixed.

The interaction is the dipole-dipole interaction for dipoles polarized along
the z axis:

$$
V_{\mathrm{dd}}(\boldsymbol\rho, z)
= D\,\frac{1 - 3z^2/r^2}{r^3},
\qquad
r^2 = |\boldsymbol\rho|^2 + z^2.
$$

Here:

- $\boldsymbol\rho$ is the xy pair displacement after applying periodic boundary
  conditions.
- $z = z_i - z_j$ is the layer separation for the pair.
- `D` is `dipole_strength`.

This has two important physical regimes:

- Same-layer pairs have $z = 0$, so

  $$
  V_{\mathrm{same}}(\boldsymbol\rho)
  = \frac{D}{|\boldsymbol\rho|^3},
  $$

  which is repulsive for positive `D`.

- Opposite-layer pairs have $|z| = d$, so

  $$
  V_{\mathrm{opposite}}(\boldsymbol\rho, d)
  =
  D\,\frac{1 - 3d^2/(|\boldsymbol\rho|^2 + d^2)}
  {(|\boldsymbol\rho|^2 + d^2)^{3/2}}.
  $$

  This can be attractive at short xy separation because the $-3z^2/r^2$
  term dominates when $\boldsymbol\rho$ is small.

The code includes a small softening parameter in the potential configuration:

```python
"softening": 0.1
```

With softening, the local pair distance used by the direct softened piece is
effectively

$$
r_{\mathrm{soft}}^2 = |\boldsymbol\rho|^2 + z^2 + \epsilon^2,
$$

where $\epsilon$ is the configured softening.

This regularizes the short-distance singularity and makes optimization more
stable.

## 4. Periodic Dipolar Potential

The current configuration uses the Ewald path:

```python
"use_ewald": True
"ewald_geometry": "xy_periodic_open_z"
"ewald_alpha": 10.0 / sqrt(cell_area)
"ewald_real_cut": 2
"ewald_kmax": 12
```

This means the pair interaction is not just the nearest minimum-image direct
interaction. Instead, the potential includes the contribution of periodic
copies in the xy plane while keeping the z direction open.

The implemented Ewald construction uses the identity

$$
\frac{1 - 3z^2/r^2}{r^3}
=
-\frac{\partial^2}{\partial z^2}\left(\frac{1}{r}\right).
$$

The code first builds a two-dimensional-periodic/open-z Coulomb Ewald pair
kernel, then applies $-\partial^2/\partial z^2$ to obtain the dipolar pair
kernel. The Ewald sum has:

- a real-space image sum, controlled by `ewald_real_cut`,
- a reciprocal-space sum over xy reciprocal lattice vectors, controlled by
  `ewald_kmax`,
- a `G = 0` term for the open-z two-dimensional periodic geometry.

When `softening > 0`, the code separates the zero real-space image from the
smooth Ewald remainder. The singular local zero-image part is replaced by a
softened direct dipole plus a regular correction. This avoids subtracting very
large nearly singular same-layer terms.

There is also a direct minimum-image potential builder in
`bilayer_hamiltonian.py`. That path is used only when `use_ewald = False`.

## 5. Wavefunction Ansatz

The neural network represents a real positive bosonic wavefunction:

$$
\psi_\theta(R, s) = \exp(\log |\psi_\theta(R, s)|).
$$

In code, `BosonNet.apply(...)` returns

```python
phase = 1.0
log_abs = neural_network_output
```

The current path does not implement a complex phase. It is appropriate for a
bosonic ground-state search where the lowest-energy state can be represented by
a nonnegative real wavefunction.

### Periodic Input Features

The network does not feed raw `x,y` coordinates directly into the first layer.
For each particle it maps positions into periodic features:

$$
\mathbf u_i = \mathbf L^{-1}\mathbf r_i,
\qquad
\mathrm{features}_i =
\left[
\sin(2\pi\mathbf u_i),
\cos(2\pi\mathbf u_i),
s_i
\right].
$$

For `ndim = 2`, this gives:

$$
\left[
\sin(2\pi u_x),\,
\sin(2\pi u_y),\,
\cos(2\pi u_x),\,
\cos(2\pi u_y),\,
s_i
\right].
$$

These features make the neural network periodic by construction in the xy
cell. Translating a particle by a lattice vector leaves its sine/cosine
features unchanged.

### Bosonic Symmetry

The wavefunction must be symmetric under exchange of identical bosons. BosonNet
enforces this through a permutation-equivariant per-particle representation
followed by symmetric pooling.

The current run uses:

```python
cfg.network.BosonNet.architecture = "Transformer"
cfg.network.BosonNet.num_layers = 3
cfg.network.BosonNet.mlp_dim = 64
cfg.network.BosonNet.num_heads = 4
cfg.network.BosonNet.attn_dim = 16
cfg.network.BosonNet.value_dim = 16
cfg.network.BosonNet.num_perceptrons_per_layer = 2
cfg.network.BosonNet.use_layer_norm = True
cfg.network.BosonNet.mlp_activation_fct = "GELU"
```

The processing pipeline is:

1. Build one feature vector per boson.
2. Embed each feature vector into a learned `mlp_dim`-dimensional token.
3. Apply several shared permutation-equivariant blocks.
4. Average the final tokens over particles.
5. Pass the pooled vector through a final MLP to produce one scalar,
   `log_abs`.

The average pooling step is what makes the final scalar invariant to a
permutation of particle order. The Transformer attention layers are
equivariant: if the input particles are permuted, the output tokens are
permuted in the same way. The mean over tokens removes that ordering.

Because the layer label is part of the per-particle feature, the network can
learn different correlations for particles in the same layer and opposite
layers while still producing a symmetric bosonic scalar for the full labelled
configuration.

## 6. What VMC Minimizes

For a trial wavefunction $\psi_\theta$, the variational energy is

$$
E(\theta)
=
\frac{\langle \psi_\theta | H | \psi_\theta \rangle}
{\langle \psi_\theta | \psi_\theta \rangle}.
$$

VMC rewrites this expectation as an average over the probability distribution

$$
p_\theta(R)
=
\frac{|\psi_\theta(R)|^2}
{\int |\psi_\theta(R)|^2\,dR}.
$$

The key object is the local energy:

$$
E_L(R)
=
\frac{(H\psi_\theta)(R)}{\psi_\theta(R)}.
$$

Then

$$
E(\theta)
=
\mathbb E_{R \sim |\psi_\theta|^2}\!\left[E_L(R)\right].
$$

The code estimates this average with a batch of Monte Carlo walkers. Each
walker is one complete bilayer configuration: all `N` xy positions plus the
fixed layer labels.

## 7. Local Energy Evaluation

The local energy function is built by
`periodicwave.pbc.bilayer_hamiltonian.local_energy`.

For each walker it computes:

$$
E_L(R) = T_L(R) + V(R).
$$

The potential part is the pair sum:

$$
V(R)
=
\sum_{i<j}
V_{\mathrm{dd}}(\boldsymbol\rho_{ij}, z_i - z_j).
$$

using either Ewald or direct minimum-image evaluation, depending on the config.

The kinetic part uses the logarithmic form of the kinetic energy. For a real
positive wavefunction,

$$
\psi = \exp(f), \qquad f = \log|\psi|.
$$

so

$$
\frac{\nabla^2\psi}{\psi}
=
\nabla^2 f + |\nabla f|^2.
$$

Therefore the local kinetic energy is

$$
T_L(R)
=
-\frac{1}{2}
\left[
\nabla_R^2 f(R) + |\nabla_R f(R)|^2
\right].
$$

The derivatives are only with respect to the flattened xy coordinate vector:

$$
R = (x_1, y_1, \ldots, x_N, y_N).
$$

The current config chooses:

```python
"kinetic_kwargs": {"laplacian_method": "folx"}
```

so the code uses `folx.forward_laplacian` to obtain the Jacobian and Laplacian
of the network output more efficiently than explicitly looping over every
coordinate direction.

## 8. Monte Carlo Sampling

The MCMC code samples configurations from $|\psi_\theta|^2$ using
Metropolis-Hastings random walks.

For each walker, the proposal is an all-particle Gaussian displacement:

$$
R' = R + \sigma\eta,
$$

where `eta` is standard normal noise with the same shape as the flattened xy
position vector. The layer labels are copied unchanged.

The acceptance rule uses:

$$
\log p(R) = 2\log|\psi_\theta(R)|.
$$

A proposed move is accepted with probability

$$
\min\left(1, \exp[\log p(R') - \log p(R)]\right).
$$

The current MCMC settings are:

```python
cfg.mcmc.burn_in = 20
cfg.mcmc.steps = 10
cfg.mcmc.init_width = 0.5
cfg.mcmc.move_width = 0.1
cfg.mcmc.move_width_updater = "adaptive"
cfg.mcmc.adapt_frequency = 10
cfg.mcmc.adaptive_steps = 100
```

At the beginning of a fresh run, the xy coordinates are initialized from a
Gaussian of width `init_width`. The chain is then burned in for `burn_in`
optimization-style sampling steps with no parameter update.

During training, each optimization iteration first advances the walkers by
`cfg.mcmc.steps` Metropolis steps. The average acceptance probability is logged
as `pmove`. During the adaptive phase, the proposal width is increased if
`pmove > 0.55` and decreased if `pmove < 0.50`. After
`cfg.mcmc.adaptive_steps`, the proposal width is frozen.

## 9. Optimization

After MCMC updates the walker batch, the code evaluates the sampled energy and
updates the network parameters.

The current optimizer configuration is:

```python
cfg.optim.optimizer = "adam_kfac"
cfg.optim.iterations = 5000
cfg.optim.lr.rate = 1e-4
cfg.optim.lr.delay = 1000
cfg.optim.lr.decay = 1.0
cfg.optim.adam_kfac.switch_iteration = 2000
cfg.optim.adam_kfac.kfac_lr_rate = 0.01
cfg.optim.adam_kfac.kfac_lr_delay = 10000.0
cfg.optim.adam_kfac.kfac_lr_decay = 1.0
```

This means:

1. Use Adam for the first 2000 iterations.
2. Reinitialize a KFAC optimizer from the current parameters and walkers.
3. Continue with KFAC until the final iteration.

Adam is a robust first stage when the network and walkers are still far from a
good variational distribution. KFAC then uses an approximate curvature matrix
to take more geometry-aware steps in parameter space.

The learning-rate schedule has the form:

$$
\mathrm{lr}(t)
=
\mathrm{rate}
\left[
\frac{1}{1 + t/\mathrm{delay}}
\right]^{\mathrm{decay}}.
$$

The KFAC phase uses the analogous KFAC-specific rate parameters.

## 10. Energy Gradient Used By VMC

The loss function in `periodicwave/loss.py` implements the standard VMC energy
gradient. For a real wavefunction,

$$
\frac{dE}{d\theta}
=
2\left\langle
\left(E_L(R) - E\right)
\frac{d\log|\psi_\theta(R)|}{d\theta}
\right\rangle_{|\psi|^2}.
$$

The code realizes this through a custom JAX JVP. Operationally:

1. Evaluate local energies over the batch.
2. Compute the mean energy `E`.
3. Form the centered local-energy differences `E_L - E`.
4. Backpropagate through `log_abs` weighted by those centered differences.

The centering is important: it removes the component of the gradient
corresponding to a global normalization change of the wavefunction. A constant
shift in $\log|\psi|$ changes the normalization but not the physical state.

The code can also clip local energies before constructing the gradient. The
base config enables:

```python
cfg.optim.clip_local_energy = 5.0
cfg.optim.clip_median = False
cfg.optim.center_at_clip = True
```

Clipping is used only to stabilize gradients from rare high-energy walker
configurations. The reported energy remains the unclipped batch mean.

## 11. Data Layout

The central container is `networks.WalkerData`:

```python
WalkerData(
    positions=...,
    spins=...,
    atoms=...,
    charges=...,
)
```

For this bilayer boson code path:

- `positions` contains flattened xy coordinates.
- `spins` contains fixed layer labels.
- `atoms` and `charges` are placeholders kept for compatibility with the
  shared network and Hamiltonian call signatures.

The training code shards walkers over local JAX devices. On each device, the
per-device batch is:

$$
\mathrm{device\_batch\_size}
=
\frac{\mathrm{cfg.batch\_size}}{\mathrm{number\_of\_devices}}.
$$

The current config uses:

```python
cfg.batch_size = 2048
```

The code checks that the batch size is divisible by the total number of JAX
devices across hosts.

## 12. Logging, Checkpoints, And Diagnostics

The current run writes to:

```text
results/bilayer-bosons/BosonNet/N14_layers7_7_rs1.0_d10.0_D20.0_sq
```

The important output files are:

```text
config.json
device_info.log
train_stats.csv
qmcjax_ckpt_*.npz
```

`train_stats.csv` records:

- `step`: optimization iteration,
- `energy`: sampled variational energy,
- `ewmean`: exponentially weighted mean of the energy,
- `ewvar`: exponentially weighted energy variance estimate,
- `pmove`: Metropolis acceptance probability,
- `locstd`: standard deviation of local energy in the current batch.

`scripts/evaluate/evaluate_energies.py` plots the energy convergence from
`train_stats.csv`.
`scripts/evaluate/evaluate_observables.py` loads checkpoints and plots:

- total xy density,
- layer-resolved xy density,
- the discrete z-density,
- the static structure factor

  $$
  S(\mathbf k)
  =
  \frac{1}{N}
  \left\langle
  \left|
  \sum_j e^{i\mathbf k\cdot\mathbf r_j}
  \right|^2
  \right\rangle.
  $$

The evaluation scripts contain their own parameter block at the top. If the run
parameters in `bilayer_bosons.py` change, those scripts must be updated to
point at the matching result folder and lattice.

## 13. End-To-End Algorithm

A fresh training run follows this sequence:

1. Build the square or triangular xy cell from `N`, `r_s`, and
   `supercell_shape`.
2. Create fixed layer labels with the requested layer occupations.
3. Build the `ConfigDict`, including Hamiltonian, potential, network, MCMC,
   optimizer, and logging settings.
4. Initialize BosonNet parameters.
5. Initialize walker positions from a Gaussian distribution.
6. Replace the initial generated labels with the explicit `layer_assignment`.
7. Burn in the MCMC chains without changing network parameters.
8. For each optimization step:

   - Move walkers with Metropolis-Hastings using current $|\psi|^2$.
   - Evaluate $E_L = T_L + V$ on the walker batch.
   - Estimate mean energy and local-energy variance.
   - Compute the VMC gradient of the energy.
   - Update network parameters with Adam or KFAC.
   - Adapt MCMC proposal width during the early adaptive window.
   - Log statistics and periodically save checkpoints.

9. Use checkpoint and CSV outputs for convergence and structural diagnostics.

## 14. What The Method Is Solving

The code is not solving the Schrodinger equation by diagonalizing a Hamiltonian
matrix. Instead, it searches over a flexible family of wavefunctions
$\psi_\theta$. The variational principle says:

$$
E(\theta) \ge E_0.
$$

for the exact ground-state energy `E_0`, assuming the trial wavefunction has
the right symmetry and boundary conditions. By minimizing $E(\theta)$, the
network approximates the bosonic ground-state wavefunction and energy.

The approximation quality depends on:

- whether BosonNet can represent the relevant correlations,
- whether MCMC samples $|\psi_\theta|^2$ well,
- whether the local energy is evaluated accurately,
- whether optimization reaches a stable minimum,
- whether the finite simulation cell and Ewald cutoffs are large enough for
  the intended physics.

For this bilayer system, the network must learn a balance between same-layer
repulsion, opposite-layer attraction or correlation, kinetic delocalization,
and finite-density ordering effects under periodic boundary conditions.
