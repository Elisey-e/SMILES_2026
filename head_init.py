"""
head_init.py — CIFAR-100 head initialization for pretrained ResNet18.

Idea:
    Use the pretrained ImageNet ResNet18 classifier as a semantic prior.
    Many CIFAR-100 classes are identical or close to ImageNet classes
    (bear, butterfly, bus, clock, shark, tiger, tractor, ...).

    For each CIFAR-100 class, we copy or average the corresponding ImageNet
    classifier rows into the new 100-way head.

No gradients, no validation data, no fixed-file changes.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# CIFAR-100 class order used by torchvision.datasets.CIFAR100:
# apple, aquarium_fish, baby, bear, beaver, bed, bee, beetle, bicycle, ...
# Values are ImageNet-1K class ids from torchvision's standard class order.
# Multiple ids are averaged into one CIFAR row.
_CIFAR100_TO_IMAGENET: tuple[tuple[int, ...], ...] = (
    (948, 956),                              # apple
    (0, 1, 391, 392, 393, 394, 396, 397),    # aquarium_fish
    (431, 516, 520, 529, 850),               # baby-related proxy
    (294, 295, 296, 297),                    # bear
    (337,),                                  # beaver
    (564, 721, 797),                         # bed-related
    (309, 410, 599),                         # bee
    (300, 301, 302, 303, 304, 305, 306, 307),# beetle
    (444, 671, 870, 880),                    # bicycle
    (440, 720, 737, 898, 907),               # bottle
    (659, 809, 647),                         # bowl
    (981, 982, 724),                         # boy/person proxy
    (821, 839, 888, 460),                    # bridge
    (654, 779, 874),                         # bus
    (321, 322, 323, 324, 325, 326),          # butterfly
    (354,),                                  # camel
    (653, 473, 412),                         # can-related
    (483, 698, 663),                         # castle
    (78, 79, 310, 311, 312, 313, 315, 316, 317, 319, 320),  # caterpillar proxy
    (345, 346, 347, 351),                    # cattle
    (423, 559, 703, 765, 857),               # chair
    (367,),                                  # chimpanzee
    (409, 530, 531, 604, 826, 892),          # clock
    (970, 971, 979, 980),                    # cloud/sky proxy
    (314,),                                  # cockroach
    (831,),                                  # couch
    (118, 119, 120, 121, 125),               # crab
    (49, 50),                                # crocodile/alligator
    (504, 647, 968),                         # cup
    (51, 69),                                # dinosaur proxy
    (149, 150, 147, 148),                    # dolphin/marine mammal proxy
    (101, 385, 386),                         # elephant
    (389, 391, 394, 0),                      # flatfish/fish proxy
    (970, 972, 975, 979, 958),               # forest/nature proxy
    (277, 278, 279, 280),                    # fox
    (981, 982, 578),                         # girl/person proxy
    (333, 338),                              # hamster
    (425, 449, 497, 580, 660),               # house/building proxy
    (104,),                                  # kangaroo/wallaby
    (508, 810, 878),                         # keyboard
    (619, 846),                              # lamp
    (621,),                                  # lawn_mower
    (288, 289, 290),                         # leopard
    (291,),                                  # lion
    (38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48),  # lizard
    (122, 123, 124),                         # lobster
    (981, 982, 724, 652),                    # man/person proxy
    (988, 990),                              # maple_tree proxy
    (665, 670),                              # motorcycle
    (970, 972, 976, 979, 980),               # mountain
    (673, 674),                              # mouse
    (947, 991, 992, 993, 994, 995, 996, 997),# mushroom/fungi
    (988,),                                  # oak_tree proxy
    (950,),                                  # orange
    (986,),                                  # orchid proxy
    (360, 175),                              # otter
    (953, 975, 978),                         # palm_tree/tropical proxy
    (952, 954, 956, 948),                    # pear/fruit proxy
    (717, 864, 867, 569),                    # pickup_truck
    (496, 958, 970),                         # pine_tree proxy
    (958, 970, 979),                         # plain
    (923, 729, 868),                         # plate
    (984, 985, 986),                         # poppy/flower proxy
    (334,),                                  # porcupine
    (106, 363, 383),                         # possum proxy
    (330, 331, 332),                         # rabbit
    (362, 298, 387),                         # raccoon proxy
    (5, 6),                                  # ray
    (919, 920, 640),                         # road/street proxy
    (657, 744, 812),                         # rocket
    (985, 986, 883),                         # rose/flower proxy
    (973, 975, 976, 977, 978),               # sea
    (150, 147, 148),                         # seal/sea mammal proxy
    (2, 3, 4),                               # shark
    (356, 357, 358, 359),                    # shrew proxy
    (361,),                                  # skunk
    (538, 682, 663, 698, 500),               # skyscraper proxy
    (112, 113, 114, 115),                    # snail
    (52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68),  # snake
    (70, 72, 73, 74, 75, 76, 77),            # spider
    (335, 336, 382),                         # squirrel
    (829, 874, 547, 466, 820),               # streetcar/train proxy
    (984, 985, 986, 599),                    # sunflower/flower proxy
    (945,),                                  # sweet_pepper
    (526, 532, 736),                         # table
    (847, 586),                              # tank
    (487, 528, 707),                         # telephone
    (851, 664, 598, 782),                    # television
    (292, 282),                              # tiger
    (866,),                                  # tractor
    (466, 547, 565, 705, 820),               # train
    (391, 0, 394, 389),                      # trout/fish proxy
    (984, 985, 986),                         # tulip/flower proxy
    (33, 34, 35, 36, 37),                    # turtle
    (894, 492, 493, 495, 648),               # wardrobe/cabinet proxy
    (147, 148),                              # whale
    (988, 990, 958),                         # willow_tree proxy
    (269, 270, 271, 170),                    # wolf
    (982, 578, 399),                         # woman/person proxy
    (110, 111),                              # worm
)


def _safe_fallback_init(layer: nn.Linear) -> None:
    """Deterministic conservative init used before transfer and as fallback."""
    with torch.no_grad():
        nn.init.orthogonal_(layer.weight)
        layer.weight.mul_(0.02)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


def _load_imagenet_fc() -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return pretrained ResNet18 ImageNet fc weights, or None if unavailable."""
    try:
        from torchvision import models

        weights = models.ResNet18_Weights.IMAGENET1K_V1
        src_model = models.resnet18(weights=weights)
        src_fc = src_model.fc
        return src_fc.weight.detach().cpu(), src_fc.bias.detach().cpu()
    except Exception:
        return None


def init_last_layer(layer: nn.Linear) -> None:
    """Initialize the 100-way CIFAR-100 classification head in-place."""
    if not isinstance(layer, nn.Linear):
        raise TypeError(f"init_last_layer expects nn.Linear, got {type(layer)!r}")

    # Keep validate.py's global RNG stream reproducible.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12345)
        _safe_fallback_init(layer)

        loaded = _load_imagenet_fc()
        if loaded is None:
            return

        src_w, src_b = loaded
        src_w = src_w.to(device=layer.weight.device, dtype=layer.weight.dtype)
        src_b = src_b.to(device=layer.weight.device, dtype=layer.weight.dtype)

        if layer.out_features != len(_CIFAR100_TO_IMAGENET):
            return
        if layer.in_features != src_w.shape[1]:
            return

        with torch.no_grad():
            for cifar_idx, imagenet_ids in enumerate(_CIFAR100_TO_IMAGENET):
                valid_ids = [i for i in imagenet_ids if 0 <= i < src_w.shape[0]]
                if not valid_ids:
                    continue

                ids = torch.tensor(valid_ids, device=layer.weight.device)
                layer.weight[cifar_idx].copy_(src_w.index_select(0, ids).mean(dim=0))

                if layer.bias is not None:
                    layer.bias[cifar_idx].copy_(src_b.index_select(0, ids).mean())

            if layer.bias is not None:
                layer.bias.sub_(layer.bias.mean())