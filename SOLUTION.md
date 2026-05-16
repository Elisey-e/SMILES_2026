# Solution: Zero-Order Fine-Tuning of ResNet18 on CIFAR-100

## Reproducibility

### Environment

The solution was tested with the package versions from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Required packages:

```text
torch==2.10.0
torchvision==0.25.0
tqdm==4.67.1
```

CUDA is recommended for runtime, but the code does not require a specific GPU id. The first run downloads CIFAR-100 to `./data` and downloads the pretrained ResNet18 ImageNet weights through torchvision.

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

The submitted `results.json` was produced with this configuration.

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

The main metric is:

```text
val_accuracy_top1_finetuned = 30.69%
```

---

## Final approach

The solution modifies only the files allowed by the assignment:

- `zo_optimizer.py`
- `head_init.py`
- `augmentation.py`
- `train_data.py`

`validate.py` and `model.py` are used as fixed infrastructure.

### 1. `head_init.py`: semantic ImageNet-head initialization

The new CIFAR-100 classification head is initialized from the pretrained ImageNet classifier of ResNet18.

For each CIFAR-100 class, I select one or several semantically related ImageNet-1K classifier rows and average them into the corresponding CIFAR-100 row. For example, CIFAR classes such as `bear`, `bus`, `butterfly`, `shark`, `tiger`, and `tractor` have close ImageNet counterparts. If a class has several related ImageNet categories, their classifier weights are averaged.

Before copying the ImageNet rows, the head is initialized with a deterministic conservative fallback:

```python
nn.init.orthogonal_(layer.weight)
layer.weight.mul_(0.02)
nn.init.zeros_(layer.bias)
```

The ImageNet classifier bias is also transferred and centered by subtracting its mean. This keeps relative class-prior information while removing a global logit shift.

This initialization uses:

- no validation data;
- no gradients;
- no fixed-file modifications;
- only pretrained weights already used by the assignment model.

The main purpose is to avoid starting ZO from a random 100-way head, where the initial loss is too high and the limited budget is not enough for recovery.

### 2. `zo_optimizer.py`: weight-only MeZO with SignSGD momentum

The optimizer uses a simultaneous perturbation / MeZO-style estimator. Each random direction is evaluated with a symmetric two-point estimate:

```text
(f(x + eps z) - f(x - eps z)) / (2 eps)
```

Instead of perturbing each parameter separately, all selected parameters are perturbed together. This keeps the number of loss evaluations independent of the number of weights.

The final optimizer tunes only:

```python
self.layer_names = ["fc.weight"]
```

Bias is not tuned. In experiments, bias updates were noisy because mini-batches may have non-uniform class priors. Most useful adaptation came from rotating the class hyperplanes in `fc.weight`.

The update rule is SignSGD with momentum:

- `n_samples = 16` random SPSA directions per optimizer step;
- `eps = 1e-3`;
- `lr = 6e-4`;
- `beta1 = 0.9`;
- linear warmup for the first 4 steps;
- cosine decay down to `min_lr_ratio = 0.15`.

### 3. `train_data.py`: deterministic class-balanced training order

The train loader uses a deterministic class-balanced subset of 8192 samples.

For each mini-batch, classes are selected in a round-robin order. With `batch_size=64`, each batch contains 64 distinct CIFAR-100 classes, and the class window shifts after every batch. This reduces class-prior noise in the scalar loss values used by the zero-order estimator.

The loader uses:

```python
shuffle=False
num_workers=0
drop_last=False
```

This makes the training order deterministic and reproducible.

### 4. `augmentation.py`: mild CIFAR-style augmentation

The final training transform is intentionally simple:

```python
T.RandomCrop(32, padding=4)
T.RandomHorizontalFlip(p=0.5)
T.Resize(224, antialias=True)
T.ToTensor()
T.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD)
```

The validation transform is unchanged except for deterministic resize, tensor conversion, and normalization.

Stronger augmentations were tested, but they increased loss-estimation noise and did not improve the final ZO metric reliably.

---

## What contributed most

The main contributions were:

1. **Semantic head initialization**  
   This increased the initialized-head checkpoint to about 24%, giving ZO a useful starting point.

2. **Weight-only MeZO / SPSA optimization**  
   Simultaneous perturbation avoids the cost of per-parameter finite differences and makes the optimization feasible.

3. **Class-balanced training order**  
   Batch-level class balance reduced loss noise for the scalar zero-order estimator.

4. **Mild augmentation**  
   Simple crop + flip was more stable than stronger augmentation policies under ZO.

---

## Experiments and failed attempts

### Random and standard initialization

I tested standard head initializations such as Kaiming, Xavier, orthogonal, and small-scale orthogonal initialization. These are clean and close to the skeleton, but the initialized-head accuracy was too low for the limited ZO budget.

### Prototype / class-mean initialization

I tested a stronger initialization based on class prototypes: compute average pretrained ResNet18 features for CIFAR-100 train images and use them as classifier weights. This produced very high initialized-head accuracy, but ZO no longer improved the result meaningfully. I discarded it because the assignment should still demonstrate the zero-order optimizer.

### Automatic weak prototype blending

I tested blending random orthogonal weights with weak class prototypes. This was cleaner than hard-coded ImageNet mapping, but the initialized-head accuracy dropped too much and the ZO optimizer could not recover enough.

### Stronger augmentation

I tested RandomResizedCrop, ColorJitter, RandomErasing, and light RandAugment-style policies. They did not reliably beat the mild CIFAR crop + flip transform. For this optimizer, strong image perturbations increase noise in the scalar loss estimates.

### Train-data alternatives

I compared several train loader strategies:

- standard shuffled full CIFAR-100 train loader;
- deterministic class-balanced round-robin order;
- randomized class-balanced batches;
- small balanced replay subset;
- globally balanced 8192-sample subset with shuffle.

The deterministic class-balanced round-robin loader was selected because it was simple, stable, and almost tied with the best randomized variant while being easier to reproduce and explain.

### Bias tuning

I considered tuning `fc.bias` together with `fc.weight`, but did not keep it in the final solution. Bias updates were more sensitive to mini-batch class-prior noise, while tuning only `fc.weight` was more stable.

---

## Final files

The final repository should contain:

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

The official grading infrastructure replaces `validate.py` and `model.py`, so the solution does not rely on editing them.
