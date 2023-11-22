""" 
    Defined img_to_strips, img_to_patches functions
"""
import torch


def img_to_segments(kwargs, segment_type="strips"):
    if segment_type == "strips":
        return img_to_strips(**kwargs)
    elif segment_type == "patches":
        pass
        # return img_to_patches(**kwargs)


def img_to_strips(
    img: torch.Tensor,
    strip_thickness: int,
    flatten_channels: bool = True,
    batch_first: bool = False,
) -> torch.Tensor:
    # [TODO] Write test cases for the function
    """Given an image, it returns a set of vertical and horizontal strips

    Args:
        img (torch.Tensor): Tensor of the shape [B, C, H, W]
        strip_thickness (int): Thickness 's' of the strips to be generated
        flatten_channels (bool, optional): Whether to flatten the channels or not. Defaults to True.

    Returns:
        torch.Tensor: Tensor of size [T, B, s, H] where 'T' is "sequence length"
    """
    b, c, h, w = img.shape
    if not h == w:
        raise ValueError(
            f"The images are expected to have same (h,w) [ie., h==w]. But got (h,w) = ({h},{w})"
        )
    if not h % strip_thickness == 0:
        raise ValueError(
            f"Cannot divide image of size ({h},{w}) into equal strips of size s={strip_thickness}."
        )
    t = h // strip_thickness

    img_h = img.reshape(b, c, t, strip_thickness, w)
    img_w = img.reshape(b, c, h, t, strip_thickness)
    if batch_first:
        img_h = img_h.permute(0, 2, 1, 3, 4)  # [b, t, c, s, w]
        img_w = img_w.permute(0, 3, 1, 4, 2)  # [b, t, c, s, h]
        img1 = torch.cat((img_h, img_w), dim=1)
    else:
        img_h = img_h.permute(2, 0, 1, 3, 4)  # [t, b, c, s, w]
        img_w = img_w.permute(3, 0, 1, 4, 2)  # [t, b, c, s, h]
        img1 = torch.cat((img_h, img_w), dim=0)

    if flatten_channels:
        img1 = img1.flatten(2, 4)  # [2*t, b, c*s*h]
    return img1
