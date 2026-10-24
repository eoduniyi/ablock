import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import scipy.ndimage as ndi

class MolecularMultiWellPotential:
    """
    Canonical 3-Basin Conformational Free-Energy Surface.
    Basin A: (-1.0, 1.0), depth = 3.5 kcal/mol, width = 0.55
    Basin B: (1.0, -0.5), depth = 4.0 kcal/mol, width = 0.60
    Basin C: (-0.5, -1.0), depth = 3.2 kcal/mol, width = 0.50
    Harmonic confinement: 0.20 * ||x||^2
    """
    def __init__(self, device='cpu'):
        self.device = device
        self.centers = torch.tensor([
            [-1.0, 1.0],
            [1.0, -0.5],
            [-0.5, -1.0]
        ], dtype=torch.float32, device=device)
        self.depths = torch.tensor([3.5, 4.0, 3.2], dtype=torch.float32, device=device)
        self.widths = torch.tensor([0.55, 0.60, 0.50], dtype=torch.float32, device=device)
        
    def __call__(self, x):
        v_total = torch.zeros(x.shape[0], dtype=torch.float32, device=x.device)
        for i in range(3):
            c = self.centers[i]
            d = self.depths[i]
            w = self.widths[i]
            dist_sq = torch.sum((x - c)**2, dim=1)
            v_total -= d * torch.exp(-dist_sq / (2 * w))
        v_confinement = 0.20 * torch.sum(x**2, dim=1)
        return v_total + v_confinement

    def compute_gradient(self, x):
        x_req = x.detach().clone().requires_grad_(True)
        v = self(x_req)
        grad = torch.autograd.grad(v.sum(), x_req)[0]
        return grad

    def true_score(self, x):
        return -self.compute_gradient(x)


def build_glyph_potential(resolution=400, char='G'):
    """
    Non-Convex Typography Glyph Potential ('G') with circular boundary confinement.
    Returns:
      V: (H, W) float array of potential values
      mask: (H, W) binary float array of glyph stroke interior
      dish_info: (cx, cy, r_dish) aperture geometry
    """
    img = Image.new('L', (resolution, resolution), color=0)
    draw = ImageDraw.Draw(img)
    font = None
    for p in [
        '/System/Library/Fonts/Helvetica.ttc',
        '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
        '/System/Library/Fonts/Supplemental/Arial.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
    ]:
        try:
            font = ImageFont.truetype(p, int(resolution * 0.70))
            break
        except Exception:
            pass
    if font is None:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), char, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((resolution - w) / 2 - bbox[0], (resolution - h) / 2 - bbox[1]), char, fill=255, font=font)
    mask = (np.array(img) > 128).astype(float)
    H, W = mask.shape

    density = ndi.gaussian_filter(mask, sigma=7.5)
    density = density / density.max()
    V = -np.log(density + 1e-3)

    y_coords, x_coords = np.ogrid[:H, :W]
    cx, cy = W / 2.0, H / 2.0
    r_dish = min(H, W) * 0.44
    dish_dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
    dish_wall = np.maximum(0, dish_dist - r_dish)
    V += (dish_wall**2) * 0.08

    return V, mask, (cx, cy, r_dish)


class GlyphPotential:
    """Wrapper class around build_glyph_potential for unit tests and modular use."""
    def __init__(self, resolution=400, char='G'):
        self.resolution = resolution
        self.V, self.glyph_mask, self.dish_info = build_glyph_potential(resolution=resolution, char=char)
        self.gy, self.gx = np.gradient(self.V)
