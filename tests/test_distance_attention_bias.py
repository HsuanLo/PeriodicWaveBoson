import jax
import jax.numpy as jnp

from periodicwave import boson_network


def test_pair_distance_bias_shape_translation_and_permutation():
  lattice = jnp.eye(2) * 4.0
  init, apply = boson_network.make_pair_distance_bias(
      num_heads=3,
      distance_attention_bias_num_rbf=5,
      lattice=lattice,
      distance_attention_bias_eps=1.0e-6,
      distance_attention_bias_scale=1.0,
      layer_separation=1.5,
      use_dipole_attention_bias=False,
      dipole_attention_bias_scale=1.0,
      dipole_strength=1.0,
      density_rs=1.0)
  params = init(jax.random.PRNGKey(0))
  params = {
      "distance_intra_w": jnp.reshape(
          jnp.arange(15, dtype=jnp.float32), (5, 3)),
      "distance_inter_w": jnp.reshape(
          jnp.arange(15, 30, dtype=jnp.float32), (5, 3)),
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


def test_dipole_attention_bias_uses_strength_weighted_shape():
  lattice = jnp.eye(2) * 4.0
  pos = jnp.array([
      [0.0, 0.0],
      [0.0, 0.0],
  ])
  layers = jnp.array([1.0, -1.0])
  dipole_strength = 3.0
  density_rs = 2.0
  dipole_attention_bias_scale = 2.5

  weighted_shape_values = []
  for layer_separation in [1.0, 2.0]:
    init, apply = boson_network.make_pair_distance_bias(
        num_heads=1,
        distance_attention_bias_num_rbf=2,
        lattice=lattice,
        distance_attention_bias_eps=1.0e-6,
        distance_attention_bias_scale=1.0,
        layer_separation=layer_separation,
        use_dipole_attention_bias=True,
        dipole_attention_bias_scale=dipole_attention_bias_scale,
        dipole_strength=dipole_strength,
        density_rs=density_rs)
    params = init(jax.random.PRNGKey(0))
    params = {
        "distance_intra_w": jnp.zeros((2, 1)),
        "distance_inter_w": jnp.zeros((2, 1)),
        "dipole_intra_w": jnp.zeros((1,)),
        "dipole_inter_w": jnp.ones((1,)),
    }
    weighted_shape_values.append(apply(params, pos, layers)[0, 0, 1])

  expected_weighted_shape = []
  for layer_separation in [1.0, 2.0]:
    dz2 = layer_separation ** 2
    typical_kernel = 1.0 / (
        (density_rs ** 2 + dz2 + 1.0e-6) ** 1.5)
    direct_dipole = -2.0 * dz2 / ((dz2 + 1.0e-6) ** 2.5)
    dipole_shape = direct_dipole / typical_kernel
    bounded_shape = dipole_shape / jnp.hypot(1.0, dipole_shape)
    coupling = dipole_strength * typical_kernel * (density_rs ** 2 + 1.0e-6)
    expected_weighted_shape.append(
        dipole_attention_bias_scale * coupling * bounded_shape)

  assert jnp.allclose(
      jnp.asarray(weighted_shape_values),
      jnp.asarray(expected_weighted_shape),
      atol=1.0e-5)


def test_dipole_attention_bias_masks_self_pairs_before_projection():
  lattice = jnp.eye(2) * 4.0
  init, apply = boson_network.make_pair_distance_bias(
      num_heads=1,
      distance_attention_bias_num_rbf=2,
      lattice=lattice,
      distance_attention_bias_eps=1.0e-6,
      distance_attention_bias_scale=0.0,
      layer_separation=1.0,
      use_dipole_attention_bias=True,
      dipole_attention_bias_scale=100.0,
      dipole_strength=20.0,
      density_rs=1.0)
  params = init(jax.random.PRNGKey(0))
  params = {
      "distance_intra_w": jnp.zeros((2, 1)),
      "distance_inter_w": jnp.zeros((2, 1)),
      "dipole_intra_w": jnp.ones((1,)),
      "dipole_inter_w": jnp.ones((1,)),
  }
  pos = jnp.array([
      [0.0, 0.0],
      [0.0, 0.0],
  ])
  layers = jnp.array([1.0, 1.0])

  bias = apply(params, pos, layers)
  assert jnp.all(jnp.isfinite(bias))
  assert jnp.allclose(jnp.diagonal(bias, axis1=1, axis2=2), 0.0)


def test_dipole_attention_strength_weighted_shape_scales_with_dipole_strength():
  lattice = jnp.eye(2) * 4.0
  pos = jnp.array([
      [0.0, 0.0],
      [0.0, 0.0],
  ])
  layers = jnp.array([1.0, -1.0])
  values = []
  for dipole_strength in [3.0, 6.0]:
    init, apply = boson_network.make_pair_distance_bias(
        num_heads=1,
        distance_attention_bias_num_rbf=2,
        lattice=lattice,
        distance_attention_bias_eps=1.0e-6,
        distance_attention_bias_scale=1.0,
        layer_separation=1.0,
        use_dipole_attention_bias=True,
        dipole_attention_bias_scale=1.0,
        dipole_strength=dipole_strength,
        density_rs=2.0)
    params = init(jax.random.PRNGKey(0))
    params = {
        "distance_intra_w": jnp.zeros((2, 1)),
        "distance_inter_w": jnp.zeros((2, 1)),
        "dipole_intra_w": jnp.zeros((1,)),
        "dipole_inter_w": jnp.ones((1,)),
    }
    values.append(apply(params, pos, layers)[0, 0, 1])

  assert jnp.allclose(values[1], 2.0 * values[0], atol=1.0e-5)


def test_distance_and_dipole_scales_are_independent():
  lattice = jnp.eye(2) * 4.0
  pos = jnp.array([
      [0.0, 0.0],
      [0.0, 0.0],
  ])
  layers = jnp.array([1.0, -1.0])

  def pair_bias(distance_attention_bias_scale, dipole_attention_bias_scale,
                distance_inter_w, dipole_inter_w):
    init, apply = boson_network.make_pair_distance_bias(
        num_heads=1,
        distance_attention_bias_num_rbf=2,
        lattice=lattice,
        distance_attention_bias_eps=1.0e-6,
        distance_attention_bias_scale=distance_attention_bias_scale,
        layer_separation=1.0,
        use_dipole_attention_bias=True,
        dipole_attention_bias_scale=dipole_attention_bias_scale,
        dipole_strength=3.0,
        density_rs=2.0)
    params = init(jax.random.PRNGKey(0))
    params = {
        "distance_intra_w": jnp.zeros((2, 1)),
        "distance_inter_w": distance_inter_w,
        "dipole_intra_w": jnp.zeros((1,)),
        "dipole_inter_w": dipole_inter_w,
    }
    return apply(params, pos, layers)[0, 0, 1]

  distance_weights = jnp.ones((2, 1))
  zero_distance_weights = jnp.zeros((2, 1))
  dipole_weights = jnp.ones((1,))
  zero_dipole_weights = jnp.zeros((1,))

  dipole_only = pair_bias(0.0, 2.0, zero_distance_weights, dipole_weights)
  assert jnp.allclose(
      dipole_only, pair_bias(5.0, 2.0, zero_distance_weights, dipole_weights))

  distance_only = pair_bias(3.0, 0.0, distance_weights, zero_dipole_weights)
  assert jnp.allclose(
      distance_only, pair_bias(3.0, 9.0, distance_weights,
                               zero_dipole_weights))

  assert dipole_only != 0.0
  assert distance_only != 0.0
  assert jnp.allclose(
      pair_bias(0.0, 0.0, distance_weights, dipole_weights), 0.0, atol=1.0e-6)


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
      distance_attention_bias_num_rbf=4,
      distance_attention_bias_eps=1.0e-6,
      distance_attention_bias_scale=0.5,
      use_dipole_attention_bias=True,
      dipole_attention_bias_scale=1.0,
      dipole_strength=6.0,
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
