import torch
import gsplat


def main():
    print(f"{gsplat.__version__ = }")
    print(f"{torch.cuda.is_available() = }")


if __name__ == "__main__":
    main()
