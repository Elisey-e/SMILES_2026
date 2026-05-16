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

CUDA is recommended for runtime, but the code does not require a specific GPU id. My local runs were done on CUDA; CPU also works but is much slower because the final optimizer uses 4096 black-box loss queries in one full `128`-step run (`128 steps × 16 directions × 2 evaluations`). The first run downloads CIFAR-100 to `./data` and downloads the pretrained ResNet18 ImageNet weights through torchvision.

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

I used this split because `batch_size=64` gave noticeably more stable scalar loss estimates than smaller batches, while `128` optimizer steps were still enough for the SignSGD momentum to move the final head. In earlier tests, using more stochastic batches gave worse results even when the total sample budget stayed the same.

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

The initialized head already gives `23.96%`, and ZO fine-tuning improves it to `30.69%`, i.e. a gain of `+6.73` percentage points. The ImageNet-head sanity checkpoint is only `0.37%`, so almost all useful transfer comes from replacing and adapting the head.

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

The main purpose is to avoid starting ZO from a random 100-way head, where the initial loss is too high and the limited budget is not enough for recovery. In my runs, this semantic initialization reached `23.96%` at checkpoint 2. A much stronger prototype-style initialization reached about `48.15%` before fine-tuning, but then ZO did not improve it (`48.14%` after fine-tuning), so I did not use that version. A weak blended prototype version went in the opposite direction and produced only about `6.02%` before fine-tuning and `7.19%` after ZO, which was too weak for the budget.

### 2. `zo_optimizer.py`: weight-only MeZO with SignSGD momentum

The optimizer uses a simultaneous perturbation / MeZO-style estimator. Each random direction is evaluated with a symmetric two-point estimate:

```text
(f(x + eps z) - f(x - eps z)) / (2 eps)
```

Instead of perturbing each parameter separately, all selected parameters are perturbed together. This keeps the number of loss evaluations independent of the number of weights. In the final setup, one optimizer step uses `16` SPSA directions and therefore `32` scalar loss evaluations. A full run uses:

```text
128 steps × 16 directions × 2 evaluations = 4096 loss queries
```

Each query is made on the fixed mini-batch for the current step, so the perturbation pair compares the same images and labels.

The final optimizer tunes only:

```python
self.layer_names = ["fc.weight"]
```

Bias is not tuned. In experiments, bias updates were noisy because mini-batches may have non-uniform class priors. Most useful adaptation came from rotating the class hyperplanes in `fc.weight`. The tuned tensor has shape `100 × 512`, i.e. `51200` trainable scalar weights; this is small enough for repeated black-box perturbations but still large enough to adapt the classifier.

The update rule is SignSGD with momentum:

- `n_samples = 16` random SPSA directions per optimizer step;
- `eps = 1e-3`;
- `lr = 6e-4`;
- `beta1 = 0.9`;
- linear warmup for the first 4 steps;
- cosine decay down to `min_lr_ratio = 0.15`.

The learning-rate warmup matters because the first few SPSA estimates are very noisy. I kept a conservative final schedule instead of trying to unfreeze ResNet blocks: tuning deeper layers added too many parameters relative to the 8192-sample budget.

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

I compared several loader variants. The plain author-style shuffled loader gave `29.35%`. The deterministic class-balanced round-robin loader gave `30.69%`. A randomized balanced variant reached `30.84%` in one run, but it was slightly more complex and only `+0.15` percentage points above the deterministic version. A small balanced replay subset gave `28.74%`, and a globally balanced 8192-sample subset with normal shuffle gave `29.29%`. This suggests that batch-level class balance is more important than only balancing the whole subset.

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

Stronger augmentations were tested, but they increased loss-estimation noise and did not improve the final ZO metric reliably. The augmentation sweep was small but useful:

```text
baseline / mild crop+flip: 30.69%
1_aug:                    30.65%
2_aug:                    30.26%
3_aug:                    30.80%
4_aug:                    30.49%
5_aug:                    30.79%
6_aug:                    30.34%
```

The best two augmentation-only variants were `30.80%` and `30.79%`, but the gain over the simple crop+flip baseline was only about `+0.10` percentage points. I therefore kept the simpler transform, because it is easier to reproduce and less likely to depend on one lucky seed.

---

## What contributed most

The main contributions were:

1. **Semantic head initialization**  
   This increased the initialized-head checkpoint to `23.96%`, giving ZO a useful starting point. Random or almost-random heads were too weak, while the full prototype version was too strong and hid the optimizer contribution.

2. **Weight-only MeZO / SPSA optimization**  
   Simultaneous perturbation avoids the cost of per-parameter finite differences and makes the optimization feasible. With `16` directions per step and `128` steps, the final run uses `4096` loss queries and improves the initialized head by `+6.73` percentage points.

3. **Class-balanced training order**  
   Batch-level class balance reduced loss noise for the scalar zero-order estimator. The deterministic balanced loader improved over the plain shuffled loader by about `+1.34` percentage points (`30.69%` vs `29.35%`).

4. **Mild augmentation**  
   Simple crop + flip was more stable than stronger augmentation policies under ZO. More aggressive variants moved results within roughly `30.26%–30.80%`, so the final choice was based on stability and simplicity rather than the single best augmentation number.

---

## Experiments and failed attempts

### Random and standard initialization

I tested standard head initializations such as Kaiming, Xavier, orthogonal, and small-scale orthogonal initialization. These are clean and close to the skeleton, but the initialized-head accuracy was too low for the limited ZO budget. In practice, a near-random 100-class head starts close to chance-level behavior, and 128 ZO steps were not enough to recover to the 30% range.

### Prototype / class-mean initialization

I tested a stronger initialization based on class prototypes: compute average pretrained ResNet18 features for CIFAR-100 train images and use them as classifier weights. This produced very high initialized-head accuracy, about `48.15%`, but ZO no longer improved the result meaningfully: the fine-tuned checkpoint was about `48.14%`. I discarded it because the assignment should still demonstrate the zero-order optimizer.

### Automatic weak prototype blending

I tested blending random orthogonal weights with weak class prototypes. This was cleaner than hard-coded ImageNet mapping, but the initialized-head accuracy dropped too much and the ZO optimizer could not recover enough. One representative run gave `6.02%` initialized-head accuracy and `7.19%` after ZO fine-tuning. That confirmed that the initialization must be stronger than a mostly random head.

### Stronger augmentation

I tested RandomResizedCrop, ColorJitter, RandomErasing, and light RandAugment-style policies. They did not reliably beat the mild CIFAR crop + flip transform. For this optimizer, strong image perturbations increase noise in the scalar loss estimates. The best stronger variants were only about `+0.10` percentage points above the simple baseline, while several variants were worse (`30.26%`, `30.34%`, `30.49%`).

### Train-data alternatives

I compared several train loader strategies:

- standard shuffled full CIFAR-100 train loader: `29.35%`;
- deterministic class-balanced round-robin order: `30.69%`;
- randomized class-balanced batches: `30.84%`;
- small balanced replay subset: `28.74%`;
- globally balanced 8192-sample subset with shuffle: `29.29%`.

The deterministic class-balanced round-robin loader was selected because it was simple, stable, and almost tied with the best randomized variant while being easier to reproduce and explain. The main observation was that balancing only the whole 8192-sample subset was not enough; balancing each mini-batch mattered more for the zero-order loss differences.

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
