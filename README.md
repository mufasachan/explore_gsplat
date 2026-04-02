# Study of the Gaussian splatting method

I wanted to learn more about the so-called Gaussian splatting method. Therefore, this repo uses the [nerfstudio's `gslapt`](https://github.com/nerfstudio-project/gsplat) package to explore this. What is being done in notebook folder [`study`](./study/) and in the [package code](./src/explore_gsplat/) is summarized in the [Studies](README.md#studies) section. On top of this, this repo serve as a nice opportunity to use `nix flake` to setup a Deep Learning dev env, see [flake setup](<README#flake setup>) for more details.

> [!NOTE] 
> The python dev env is managed with uv
> If you do not want to use `nix` and its flake to run the code, you can simply use the `uv` tool. You will need to have the `ffmpeg` in your path and the CUDA toolkit installed on your machine.

## Studies

This summarizes the three tasks done by this repo. [Learn a scene](README.md#learn-a-scene) enables you to use the gaussian splatting method to learns a set of 3D gaussians for a scene from COLMAP data. 

### Learn a scene

In order to train a scene, just run the [`simple_trainer.py`](./examples/simple_trainer.py) script. You will need the [MipNerfDataset](https://jonbarron.info/mipnerf360/), I only use the `garden` directory in part 1.

```bash
# Wait a little and you will have your learnt gaussian splats in `./results/garden`
cd examples
uv run python simple_trainer.py default --data_dir ../data/v2/garden/ --result_dir ../results
```

> [!NOTE]
> The `examples` folder is from the original [gsplat](https://github.com/nerfstudio-project/gsplat/tree/main/examples) code base.

### Breakdown the forward rendering

Once the gaussians are learnt, the [`view_splats.ipynb`](./study/view_splats.ipynb) performs a slow breakdown of the gaussian splatting method. It also comments on what I found interesting. No more comments are needed, if you are curious, go read the notebook.

> [!NOTE]
> The gaussian splatting method is also fascinating based on its backward propagation, notably the pruning and the split/clone/merge strategy. This is not in the scope of the notebook.

### Study on CLIP representations with rendered scene

> [!NOTE]
> WIP

## flake setup

You can use `nix` with the `nix-commands` and `flake` experimential features enabled to get a nice environment. Because I use `nixGL` to get the correct version of my drivers, you must run the *impure* flavour of `flake develop`; `flake develop --impure`. The [`flake.nix`](./flake.nix) does not need many additional comments. The CUDA toolkit has been included since the [`fused-ssim`](https://github.com/rahul-goel/fused-ssim) needs `nvcc` and co.
