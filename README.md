# guided-diffusion

This is the codebase for [Removing Anomalies as Noises for Industrial Defect Localization](https://openaccess.thecvf.com/content/ICCV2023/html/Lu_Removing_Anomalies_as_Noises_for_Industrial_Defect_Localization_ICCV_2023_paper.html).

# Installation

This repository is based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion).
Please refer to the vanilla installation.

# Prepare models for training

 * Pretrained 256x256 guided-diffusion model: [256x256_diffusion.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt)

# Prepare MVTecAD dataset

Put your MVTecAD dataset in the ./data/MVTecAD folder.

# Training

For MVTecAD examples, run

```
bash train.sh MVTecAD $catogory 64
```

# Testing

To test the AUROC/AUPRO performance of trained models, run 

```
bash test.sh $category
```

