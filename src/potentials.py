import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt, gaussian_filter

class MolecularMultiWellPotential:
    """
    Canonical 3-Basin Conformational Free-Energy Surface.
    Basin A: (-1.0, 1.0)
    Basin B: (1.0, -0.5)
    Basin C: (-0.5, -1.0)
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


class GlyphPotential:
    """
    Non-Convex Typography Glyph Potential ('G') with circular boundary confinement.
    Creates an energy landscape where stroke interior has low potential,
    boundary has steep potential, and circular aperture acts as hard outer wall.
    """
    def __init__(self, size=256, extent=2.4, device='cpu'):
        self.size = size
        self.extent = extent
        self.device = device
        self._build_grid()
        self._rasterize_glyph()

    def _build_grid(self):
        lin = np.linspace(-self.extent, self.extent, self.size)
        self.X, self.Y = np.meshgrid(lin, -lin)
        self.dx = 2.0 * self.extent / (self.size - 1)

    def _rasterize_glyph(self):
        img = Image.new('L', (self.size, self.size), color=0)
        draw = ImageDraw.Draw(img)
        font_size = int(self.size * 0.78)
        font = None
        for font_path in [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        ]:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except Exception:
                pass
        if font is None:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), "G", font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x_pos = (self.size - w) / 2 - bbox[0]
        y_pos = (self.size - h) / 2 - bbox[1] - 0.05 * self.size
        draw.text((x_pos, y_pos), "G", fill=255, font=font)
        
        glyph_mask = np.array(img) > 128
        self.glyph_mask = glyph_mask
        
        dist_in = distance_transform_edt(glyph_mask)
        dist_out = distance_transform_edt(~glyph_mask)
        signed_dist = dist_in - dist_out
        
        density = 1.0 / (1.0 + np.exp(-0.4 * signed_dist))
        density_smooth = gaussian_filter(density, sigma=2.0)
        
        r = np.sqrt(self.X**2 + self.Y**2)
        v_field = -np.log(density_smooth + 1e-3)
        v_field += 0.8 * np.maximum(0, r - 1.8)**2
        v_field += 5.0 * np.maximum(0, r - 2.15)**3
        
        self.V_grid = torch.tensor(v_field, dtype=torch.float32, device=self.device)
        gy, gx = np.gradient(v_field, self.dx, -self.dx)
        self.grad_x_grid = torch.tensor(gx, dtype=torch.float32, device=self.device)
        self.grad_y_grid = torch.tensor(gy, dtype=torch.float32, device=self.device)

    def evaluate_grad(self, x):
        px = ((x[:, 0] + self.extent) / (2 * self.extent) * (self.size - 1)).clamp(0, self.size - 1).long()
        py = ((-x[:, 1] + self.extent) / (2 * self.extent) * (self.size - 1)).clamp(0, self.size - 1).long()
        gx = self.grad_x_grid[py, px]
        gy = self.grad_y_grid[py, px]
        return torch.stack([gx, gy], dim=1)

    def is_in_glyph(self, x):
        px = ((x[:, 0] + self.extent) / (2 * self.extent) * (self.size - 1)).clamp(0, self.size - 1).long()
        py = ((-x[:, 1] + self.extent) / (2 * self.extent) * (self.size - 1)).clamp(0, self.size - 1).long()
        mask_t = torch.tensor(self.glyph_mask, device=x.device)
        return mask_t[py, px]
