"""Run the sustained training and GEMM fixtures on the phone."""

from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np


INPUT_SIZE = 32
CANDIDATES_PER_USER = 32


def make_examples(rng, user_count):
    """Generate one labeled candidate list per user."""
    users = rng.standard_normal((user_count, 1, 16), dtype=np.float32)
    items = rng.standard_normal((user_count, CANDIDATES_PER_USER, 16), dtype=np.float32)
    user_features = np.broadcast_to(users, items.shape)
    features = np.concatenate((user_features, items), axis=-1)
    # Positive labels depend on both user-item similarity and item features.
    affinity = (user_features[..., :4] * items[..., :4]).sum(axis=-1)
    labels = (affinity + 0.7 * items[..., 4] - 0.5 * user_features[..., 4] > 0)
    return features, labels.astype(np.float32)


def initialize_params(rng, layer_sizes):
    layers = []
    for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:], strict=True):
        weights = rng.standard_normal((fan_in, fan_out), dtype=np.float32)
        weights *= np.float32(np.sqrt(2.0 / fan_in))
        layers.append((weights, np.zeros(fan_out, dtype=np.float32)))
    return tuple(layers)


def matmul_fp16_to_fp32(left, right):
    return jnp.matmul(left, right, preferred_element_type=jnp.float32)


@jax.custom_vjp
def mixed_matmul(left, right):
    return matmul_fp16_to_fp32(left.astype(jnp.float16), right.astype(jnp.float16))


def mixed_matmul_forward(left, right):
    left = left.astype(jnp.float16)
    right = right.astype(jnp.float16)
    return matmul_fp16_to_fp32(left, right), (left, right)


def mixed_matmul_backward(inputs, grad):
    left, right = inputs
    grad = grad.astype(jnp.float16)
    return matmul_fp16_to_fp32(grad, right.T), matmul_fp16_to_fp32(left.T, grad)


mixed_matmul.defvjp(mixed_matmul_forward, mixed_matmul_backward)


def predict(params, features, *, mixed_precision=False):
    matmul = mixed_matmul if mixed_precision else jnp.matmul
    for weights, bias in params[:-1]:
        features = jax.nn.relu(matmul(features, weights) + bias)
    weights, bias = params[-1]
    return (matmul(features, weights) + bias).squeeze(-1)


def numpy_predict(params, features):
    params = jax.tree.map(np.asarray, params)
    for weights, bias in params[:-1]:
        features = np.maximum(features @ weights + bias, 0)
    weights, bias = params[-1]
    return (features @ weights + bias).squeeze(-1)


def train_epoch(state, batches, labels, *, learning_rate=0.04, mixed_precision=False):
    loss_scale = 128.0 if mixed_precision else 1.0

    def loss(params, batch, target):
        logits = predict(params, batch, mixed_precision=mixed_precision)
        return jnp.mean(jax.nn.softplus(logits) - target * logits) * loss_scale

    def step(state, batch):
        params, velocity = state
        gradients = jax.grad(loss)(params, *batch)
        velocity = jax.tree.map(lambda v, g: 0.9 * v + g / loss_scale, velocity, gradients)
        params = jax.tree.map(lambda p, v: p - learning_rate * v, params, velocity)
        return (params, velocity), None

    state, _ = jax.lax.scan(step, state, (batches, labels))
    return state


def evaluate(logits, labels):
    loss = np.mean(np.logaddexp(0, logits) - labels * logits)
    accuracy = np.mean((logits > 0) == labels)
    k = 5
    order = np.argsort(-logits, axis=1, kind="stable")[:, :k]
    ranked = np.take_along_axis(labels, order, axis=1)
    discount = 1 / np.log2(np.arange(2, k + 2))
    ideal = (np.arange(k) < labels.sum(axis=1, keepdims=True)) @ discount
    dcg = ranked @ discount
    ndcg = np.divide(dcg, ideal, out=np.zeros_like(dcg), where=ideal > 0)
    return loss, accuracy, ndcg.mean()


def benchmark_sustained_training(platform, width, blocks, precision):
    device, = jax.devices(platform)
    rng = np.random.default_rng(42)
    train_features, train_labels = make_examples(rng, 1024)
    eval_features, eval_labels = make_examples(rng, 64)
    batch_size = 4096
    pairs_per_block = train_labels.size
    updates_per_block = pairs_per_block // batch_size
    input_size = INPUT_SIZE
    permutation = rng.permutation(pairs_per_block)
    batches = train_features.reshape(-1, input_size)[permutation].reshape(-1, batch_size, input_size)
    labels = train_labels.ravel()[permutation].reshape(-1, batch_size)
    layer_sizes = (input_size, width, width, 1)
    initial = initialize_params(np.random.default_rng(17), layer_sizes)
    initial_loss, _, _ = evaluate(numpy_predict(initial, eval_features), eval_labels)
    initial_state, batches, labels = jax.device_put((
        (initial, jax.tree.map(np.zeros_like, initial)), batches, labels
    ), device)
    jax.block_until_ready((initial_state, batches, labels))

    learning_rate = 0.04 * min(1, 256 / width)
    train_jit = jax.jit(train_epoch, static_argnames=("learning_rate", "mixed_precision"))
    start = time.perf_counter()
    train = train_jit.lower(initial_state, batches, labels, learning_rate=learning_rate,
                            mixed_precision=precision == "mixed").compile()
    compile_seconds = time.perf_counter() - start
    parameter_count = sum(param.size for param in jax.tree.leaves(initial))
    print(f"platform={platform}, device={device.device_kind}, layers={layer_sizes}, "
          f"params={parameter_count}, precision={precision}, batch={batch_size}, "
          f"updates={blocks * updates_per_block}, pairs={blocks * pairs_per_block}, "
          f"compile_and_load_s={compile_seconds:.3f}", flush=True)
    jax.block_until_ready(train(initial_state, batches, labels))

    durations = []
    state = initial_state
    for block in range(1, blocks + 1):
        start = time.perf_counter()
        state = train(state, batches, labels)
        jax.block_until_ready(state)
        elapsed = time.perf_counter() - start
        durations.append(elapsed)
        print(f"block={block}/{blocks}, train_s={elapsed:.3f}, "
              f"pairs_per_s={pairs_per_block / elapsed:.0f}", flush=True)

    if any(leaf.device != device or leaf.dtype != jnp.float32 for leaf in jax.tree.leaves(state)):
        raise RuntimeError(f"Expected FP32 optimizer state on {device}")
    final_loss, accuracy, ndcg = evaluate(numpy_predict(state[0], eval_features), eval_labels)
    total_seconds = sum(durations)
    window = max(1, blocks // 3)
    first_rate = window * pairs_per_block / sum(durations[:window])
    last_rate = window * pairs_per_block / sum(durations[-window:])
    print(f"total_train_s={total_seconds:.3f}, pairs_per_s={blocks * pairs_per_block / total_seconds:.0f}, "
          f"first_window_pairs_per_s={first_rate:.0f}, last_window_pairs_per_s={last_rate:.0f}, "
          f"eval_loss={initial_loss:.4f}->{final_loss:.4f}, accuracy={accuracy:.4f}, ndcg@5={ndcg:.4f}",
          flush=True)
    if not (np.isfinite(final_loss) and final_loss < initial_loss):
        raise RuntimeError("Sustained training did not improve the held-out loss")


def benchmark_sustained_gemm(platform):
    device, = jax.devices(platform)
    rng = np.random.default_rng(42)
    rows, depth, columns = 4096, 1024, 1024
    left = rng.standard_normal((rows, depth), dtype=np.float32).astype(np.float16)
    right = rng.standard_normal((depth, columns), dtype=np.float32).astype(np.float16)
    inputs = jax.device_put((left, right), device)
    jax.block_until_ready(inputs)
    matmul = jax.jit(matmul_fp16_to_fp32).lower(*inputs).compile()
    for _ in range(2):
        matmul(*inputs).block_until_ready()

    blocks, calls_per_block = 8, 64
    durations = []
    print(f"platform={platform}, device={device.device_kind}, M={rows}, K={depth}, "
          f"N={columns}, precision=fp16->fp32, gemms={blocks * calls_per_block}", flush=True)
    for block in range(1, blocks + 1):
        start = time.perf_counter()
        for _ in range(calls_per_block):
            result = matmul(*inputs).block_until_ready()
            del result
        elapsed = time.perf_counter() - start
        durations.append(elapsed)
        print(f"gemm_block={block}/{blocks}, seconds={elapsed:.3f}", flush=True)

    result = matmul(*inputs).block_until_ready()
    if result.device != device or result.dtype != jnp.float32:
        raise RuntimeError(f"Expected FP32 result on {device}")
    reference = left[:8].astype(np.float32) @ right[:, :8].astype(np.float32)
    np.testing.assert_allclose(np.asarray(result)[:8, :8], reference, rtol=1e-4, atol=5e-4)
    operations = 2 * rows * depth * columns * blocks * calls_per_block
    total_seconds = sum(durations)
    print(f"total_gemm_s={total_seconds:.3f}, effective_gflops={operations / total_seconds / 1e9:.2f}",
          flush=True)


def main():
    platform, *arguments = sys.argv[1:]
    client = arguments[:1] == ["client"]
    if client:
        arguments.pop(0)
    case, = arguments or ["mixed"]
    if case not in ("mixed", "fp32", "gemm"):
        raise ValueError(f"Unknown benchmark: {case}")
    if platform == "mali" and not client:
        from pjrt_smoke import run_worker

        package = Path(__file__).resolve().parent
        command = ["/system/bin/sh", str(package / "run_on_phone.sh"), platform, "client", case]
        run_worker(package, command, timeout=900)
        return

    jax.config.update("jax_platforms", platform)
    if case == "gemm":
        benchmark_sustained_gemm(platform)
    else:
        benchmark_sustained_training(platform, 1024, 12, case)


if __name__ == "__main__":
    main()
