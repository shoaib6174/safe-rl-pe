# Gonultas & Isler (IROS 2025) Reproduction

Standalone PyTorch reproduction of MADDPG + BiMDN for asymmetric partially-observable pursuit-evasion.

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
cd repro_gonultas_isler_2024
python training/train.py
```

## Evaluation

```bash
python evaluation/evaluate.py
```

## Tests

```bash
python -m pytest tests/ -v
```

## Key Parameters

- Arena: 16x16m
- Pursuer: bicycle model (wheelbase 0.15m)
- Evader: point-mass double integrator
- Episodes: 20,000 with 8 parallel envs
- MADDPG: 2x128 actor/critic, batch size 512
- BiMDN: 1-layer BiLSTM (128 hidden), 3 Gaussian components
