"""Download all pretrained weights once, before parallel jobs start (avoids concurrent writes
to the same hub cache by several training processes).

  python -m src.prefetch convnext_tiny vit_small bioclip2
"""
import sys


def main(names):
    for n in names:
        if n.startswith("convnext"):
            import torchvision.models as tvm
            getattr(tvm, n)(weights="DEFAULT")
        elif n == "vit_small":
            import timm
            timm.create_model("vit_small_patch16_384.augreg_in21k_ft_in1k", pretrained=True)
        elif n == "dinov2_small":
            import timm
            timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True)
        elif n == "bioclip2":
            import open_clip
            open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2")
        elif n == "lpips":
            import lpips
            lpips.LPIPS(net="alex", verbose=False)
        print("prefetched", n, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
