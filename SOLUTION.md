# Solution: Zero-Order Fine-Tuning of ResNet18 on CIFAR-100

## Reproducibility

### Environment

Install the required packages:

```bash
pip install -r requirements.txt
```

Package versions:

```text
torch==2.10.0
torchvision==0.25.0
tqdm==4.67.1
```

CUDA is recommended for runtime, but the solution does not require a specific GPU. The first run downloads CIFAR-100 to `./data` and downloads pretrained ResNet18 ImageNet weights through torchvision.

### Evaluation command

Run the official evaluation script:

```bash
python validate.py \
    --data_dir ./data \
    --batch_size 64 \
    --n_batches 128 \
    --output results.json \
    --seed 42
```

This uses the full sample budget:

```text
64 × 128 = 8192 samples
```

The run also uses `4096` black-box scalar loss queries inside the optimizer:

```text
128 steps × 16 SPSA directions × 2 evaluations = 4096 loss queries
```

The official sample budget is still satisfied: the same fixed mini-batch is reused for all SPSA probes inside one optimizer step. No gradients or backward passes are used.

### Reported result

```json
{
  "val_accuracy_top1_imagenet_head": 0.0037,
  "val_accuracy_top1_init_head": 0.2396,
  "val_accuracy_top1_finetuned": 0.3069,
  "n_batches": 128,
  "batch_size": 64,
  "layers_tuned": [
    "fc.weight"
  ],
  "total_samples": 10000
}
```

Main metric:

```text
val_accuracy_top1_finetuned = 30.69%
```

The initialized head gives `23.96%`; zero-order fine-tuning improves it to `30.69%`, a gain of `+6.73` percentage points.

---

## Final approach

The solution modifies only the allowed files:

* `head_init.py`
* `zo_optimizer.py`
* `train_data.py`
* `augmentation.py`

`validate.py` and `model.py` are not modified.

### 1. `head_init.py`: semantic ImageNet-head initialization

The CIFAR-100 head is initialized from the pretrained ImageNet classifier of ResNet18. For each CIFAR-100 class, I copy or average semantically related ImageNet-1K classifier rows. Examples include close pairs such as `bear`, `bus`, `butterfly`, `shark`, `tiger`, and `tractor`.

Before copying ImageNet rows, the head uses a deterministic fallback:

```python
nn.init.orthogonal_(layer.weight)
layer.weight.mul_(0.02)
nn.init.zeros_(layer.bias)
```

The ImageNet classifier bias is transferred and centered by subtracting its mean. This initialization uses no validation data, no gradients, and no fixed-file changes. Its role is to place the new linear head in a useful region of the pretrained feature space before ZO optimization starts.

### 2. `zo_optimizer.py`: weight-only MeZO with SignSGD momentum

The optimizer uses a simultaneous perturbation / MeZO-style estimator. Each direction is evaluated with a symmetric two-point estimate:

```text
(f(x + eps z) - f(x - eps z)) / (2 eps)
```

All selected parameters are perturbed together, so the number of loss evaluations does not depend on the number of weights.

The final optimizer tunes only:

```python
self.layer_names = ["fc.weight"]
```

I did not tune `fc.bias`, because bias updates mostly captured noisy mini-batch class priors. Updating only `fc.weight` focuses the optimizer on rotating the class hyperplanes in the pretrained feature space.

Final ZO settings:

* `n_samples = 16`
* `eps = 1e-3`
* `lr = 6e-4`
* `beta1 = 0.9`
* 4 warmup steps
* cosine decay to `min_lr_ratio = 0.15`

The update rule is momentum SignSGD. This worked better than Adam-style ZO updates because SPSA pseudo-gradients are very noisy: the magnitude of a coordinate-wise estimate is less reliable than the averaged direction. SignSGD discards unstable magnitude information and keeps only the sign of the momentum direction.

### 3. `train_data.py`: deterministic class-balanced order

The training loader uses a deterministic class-balanced subset of `8192` samples. With `batch_size=64`, each mini-batch contains 64 distinct CIFAR-100 classes, and the class window shifts between batches.

This reduces class-prior noise in scalar loss differences, which is especially important for zero-order optimization. The loader uses:

```python
shuffle=False
num_workers=0
drop_last=False
```

### 4. `augmentation.py`: mild CIFAR-style augmentation

The final training transform is intentionally simple:

```python
T.RandomCrop(32, padding=4)
T.RandomHorizontalFlip(p=0.5)
T.Resize(224, antialias=True)
T.ToTensor()
T.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD)
```

The validation transform is deterministic and unchanged except for resize, tensor conversion, and normalization. Stronger augmentations increased loss-estimation noise and did not reliably improve the ZO metric.

---

## What contributed most

1. **Semantic head initialization**
   The initialized-head checkpoint reached `23.96%`, giving the ZO optimizer a strong starting point.

2. **Weight-only MeZO / SPSA optimization**
   SPSA avoids per-parameter finite differences. Momentum SignSGD made the noisy black-box updates stable and improved the initialized head by `+6.73` percentage points.

3. **Removing `fc.bias` from tuning**
   Bias updates were sensitive to mini-batch class-prior noise. Tuning only `fc.weight` was more stable.

4. **Class-balanced training order**
   Batch-level balancing reduced scalar loss noise. The deterministic balanced loader improved over the plain shuffled loader (`30.69%` vs `29.35%`).

5. **Mild augmentation**
   Crop + flip was more stable than stronger policies. Aggressive augmentations changed the result only slightly and were less reliable.

The main ZO-specific lesson was that estimator noise was the limiting factor. More adaptive coordinate-wise updates did not help; the best strategy was to reduce noisy degrees of freedom and use a robust sign-based update.

---

## Experiments and failed attempts

### Standard random initialization

I tested Kaiming, Xavier, orthogonal, and small-scale orthogonal initialization. These were clean baselines, but the initialized-head accuracy was too low for the limited ZO budget. Starting from an almost random 100-class head made recovery difficult.

### Prototype / class-mean initialization

I tested a prototype-style initialization using class-mean pretrained ResNet18 features. It reached about `48.15%` before fine-tuning, but ZO did not improve it (`48.14%` after fine-tuning). This variant changed the nature of the solution: most of the score came from supervised prototype construction rather than from the zero-order optimizer. It also required an extra feature-extraction stage, while the final submission keeps `head_init.py` self-contained.

A weak blended prototype version was also tested, but it was too weak: about `6.02%` before fine-tuning and `7.19%` after ZO.

### Adam-style ZO optimizers

I tested several Adam-like MeZO variants with different learning rates, perturbation sizes, and numbers of SPSA samples. They were less stable than momentum SignSGD. The likely reason is that Adam uses coordinate-wise magnitude information, while SPSA coordinate magnitudes are dominated by random perturbation noise.

### Bias tuning

I tested tuning `fc.bias` together with `fc.weight`, but did not keep it. CIFAR-100 is globally balanced, while individual mini-batches are not. Bias perturbations therefore often reflected accidental mini-batch class composition rather than useful validation-improving directions.

### Train-data alternatives

I compared several training-data strategies:

* standard shuffled full CIFAR-100 loader: `29.35%`
* deterministic class-balanced round-robin loader: `30.69%`
* randomized class-balanced batches: `30.84%`
* small balanced replay subset: `28.74%`
* globally balanced 8192-sample subset with shuffle: `29.29%`

The deterministic class-balanced loader was selected because it was simple, reproducible, and close to the best randomized result.

### Stronger augmentation

I tested stronger policies such as RandomResizedCrop, ColorJitter, RandomErasing, and light RandAugment-style transforms. They did not reliably beat mild crop + flip. The best augmentation-only variants were around `30.79%–30.80%`, but the gain over the final simple transform was only about `+0.10` percentage points, so I kept the simpler version.

---

## Final repository contents

The repository contains:

```text
README.md
SOLUTION.md
requirements.txt
results.json
run.sh
validate.py
model.py
zo_optimizer.py
head_init.py
augmentation.py
train_data.py
```

The solution is self-contained and runs with the official `validate.py`.
