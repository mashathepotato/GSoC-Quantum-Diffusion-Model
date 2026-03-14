# GSOC Quantum Diffusion Model

This is a research project for ML4Sci with Google Summer of Code aimed at building a quantum denoising diffusion model.

## Description

The mid-term update blog for this project can be accessed here: [QDM Part I](https://medium.com/@mashapotatoes/gsoc-2024-quantum-diffusion-models-for-high-energy-physics-892e59ddcd3e).

The final blog can be accessed here: [QDM Part II](https://medium.com/@mashapotatoes/6e693d625931).

More details can be found at the [MLNCP NeurIPS 2024 Workshop](https://openreview.net/pdf?id=vUQLzDAdqt) website. The scaled version of that work is published in the [AAAI Symposium Series 2025](https://ojs.aaai.org/index.php/AAAI-SS/article/view/36901).

If you use or build on this work, please cite our paper:

* [Quantum diffusion model for quark and gluon jet generation (AAAI Symposium Series 2025)](https://ojs.aaai.org/index.php/AAAI-SS/article/view/36901)

```bibtex
@inproceedings{baidachna2025quantum,
  title={Quantum diffusion model for quark and gluon jet generation},
  author={Baidachna, Mariia and Guadarrama, Rey and Dahale, Gopal Ramesh and Magorsch, Tom and Pedraza, Isabel and Matchev, Konstantin T and Matcheva, Katia and Kong, Kyoungchul and Gleyzer, Sergei},
  booktitle={Proceedings of the AAAI Symposium Series},
  volume={7},
  number={1},
  pages={323--329},
  year={2025}
}
```

## Getting Started

### Dependencies and Installation

* Install dependencies: `pip install -r requirements.txt`
* Latest experiments use 64x64 cropped jet images; for quick local runs, many notebooks use smaller sample sizes (e.g. 1k). Some older notebooks still use 16x16 crops.
* Quantum or hybrid models use Pennylane for simulations, which can be much slower than the classical variant, so experimenting with the device could be helpful depending on hardware availability

### Running the notebooks

* Most of the work is in Jupyter notebooks with preloaded examples
* Start with `notebooks/data_prep.ipynb` to prepare cropped/normalized data files in `data/`
* Explore experiments in `notebooks/classical/` and `notebooks/quantum/`
* Helper functions and stats live in `utils/`
* Run each notebook separately cell by cell or export to python scripts
