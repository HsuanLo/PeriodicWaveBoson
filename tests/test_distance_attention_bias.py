import jax
import jax.numpy as jnp

from periodicwave import boson_network


def test_pair_distance_bias_shape_translation_and_permutation():
  lattice = jnp.eye(2) * 4.0
  init, apply = boson_network.make_pair_distance_bias(
      num_heads=3, num_rbf=5, lattice=lattice, eps=1.0e-6)
  params = init(jax.random.PRNGKey(0))
  params = {
      "intra_w": jnp.reshape(jnp.arange(15, dtype=jnp.float32), (5, 3)),
      "inter_w": jnp.reshape(jnp.arange(15, 30, dtype=jnp.float32), (5, 3)),
  }
  pos = jnp.array([
      [0.1, 0.2],
      [1.1, 0.4],
      [0.3, 1.4],
      [2.2, 2.0],
  ])
  layers = jnp.array([1.0, 1.0, -1.0, -1.0])

  bias = apply(params, pos, layers)
  assert bias.shape == (3, 4, 4)
  assert jnp.allclose(jnp.diagonal(bias, axis1=1, axis2=2), 0.0)

  translated = apply(params, pos + jnp.array([3.7, -2.3]), layers)
  assert jnp.allclose(translated, bias, atol=1.0e-5)

  perm = jnp.array([2, 0, 3, 1])
  permuted = apply(params, pos[perm], layers[perm])
  expected = bias[:, perm, :][:, :, perm]
  assert jnp.allclose(permuted, expected, atol=1.0e-5)


def test_distance_biased_transformer_output_and_xy_gradients():
  lattice = jnp.eye(2) * 4.0
  network = boson_network.make_boson_net(
      nspins=(2, 2),
      charges=jnp.zeros((0,)),
      ndim=2,
      complex_output=False,
      pbc_lattice=lattice,
      architecture="Transformer",
      num_layers=1,
      mlp_dim=16,
      num_heads=2,
      attn_dim=4,
      value_dim=4,
      num_perceptrons_per_layer=1,
      use_layer_norm=True,
      mlp_activation_fct="GELU",
      use_distance_attention_bias=True,
      distance_bias_num_rbf=4,
      distance_bias_eps=1.0e-6,
  )
  params = network.init(jax.random.PRNGKey(1))
  pos = jnp.array([
      [0.1, 0.2],
      [1.1, 0.4],
      [0.3, 1.4],
      [2.2, 2.0],
  ]).reshape(-1)
  layers = jnp.array([1.0, 1.0, -1.0, -1.0])

  _, log_abs = network.apply(params, pos, layers, None, None)
  assert log_abs.shape == ()

  grad = jax.grad(
      lambda xy: network.apply(params, xy, layers, None, None)[1])(pos)
  assert grad.shape == pos.shape
  assert jnp.all(jnp.isfinite(grad))

