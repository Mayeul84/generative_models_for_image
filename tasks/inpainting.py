import torch
import numpy as np
from PIL import Image
from . import register_operator, LinearOperator


@register_operator(name='inpainting')
class Inpainting(LinearOperator):
    """
    Inpainting operator for the PnP Split Gibbs Sampler.

    The forward model is: y = H x + n,  n ~ N(0, sigma^2 * I_M)

    H in {0,1}^{M x N} is a binary subsampling matrix (subset of rows of I_N).
    Because H selects rows of the identity, we have H H^T = I_M,
    which enables a closed-form diagonal covariance via Sherman-Morrison-Woodbury
    (eq. 18 of the paper):

        Q_x^{-1} = rho^2 * (I_N - (rho^2 / (sigma^2 + rho^2)) * H^T H)

    Since H^T H is a diagonal binary mask, Q_x^{-1} is diagonal and sampling
    from the conditional Gaussian (eq. 15) is exact and efficient (E-PO, eq. 17).

    Parameters (from YAML)
    ----------------------
    channels      : int   — number of image channels (1 or 3)
    img_dim       : int   — spatial dimension (assumes square images)
    mask_type     : str   — 'random_pixels' | 'box' | 'file'  (default: 'random_pixels')
    missing_ratio : float — fraction of pixels to mask, in [0, 1] (used by random_pixels)
    box_size      : int   — side length of the masked box in pixels     (used by box)
    mask_path     : str   — path to a PNG/NPY mask file                 (used by file)
    device        : torch.device

    After instantiation, you can replace the mask at any time:
        operator.set_mask(my_tensor)
        operator.set_mask_from_file("path/to/mask.png")
        operator.set_mask_random(missing_ratio=0.5)
        operator.set_mask_box(box_size=64)
    Or via direct assignment:
        operator.mask = my_tensor
    """

    def __init__(
        self,
        channels: int,
        img_dim: int,
        device,
        mask_type: str = 'random_pixels',
        missing_ratio: float = 0.5,
        box_size: int = 64,
        mask_path: str = None,
    ) -> None:
        self.channels = channels
        self.img_dim  = img_dim
        self.device   = device

        if mask_type == 'file' and mask_path is not None:
            self._mask = self._load_mask_from_file(mask_path)
        elif mask_type == 'box':
            self._mask = self._make_box_mask(box_size)
        else:  # 'random_pixels'  (default)
            self._mask = self._make_random_mask(missing_ratio)

    # ------------------------------------------------------------------
    # mask property — always accessible as operator.mask
    # ------------------------------------------------------------------

    @property
    def mask(self) -> torch.Tensor:
        return self._mask

    @mask.setter
    def mask(self, value: torch.Tensor):
        """Direct assignment:  operator.mask = my_tensor"""
        self._mask = value.float().to(self.device)

    # ------------------------------------------------------------------
    # Public helpers to change the mask after instantiation
    # ------------------------------------------------------------------

    def set_mask(self, mask: torch.Tensor):
        """Set the mask from a tensor of shape (1,H,W) or (C,H,W), values in {0,1}."""
        self._mask = mask.float().to(self.device)
        return self  # chainable

    def set_mask_from_file(self, path: str):
        """Load a mask from a PNG (values 0/255 -> 0/1) or a .npy file."""
        self._mask = self._load_mask_from_file(path)
        return self

    def set_mask_random(self, missing_ratio: float = 0.5, seed: int = None):
        """Generate a new i.i.d. Bernoulli mask with the given missing ratio."""
        if seed is not None:
            torch.manual_seed(seed)
        self._mask = self._make_random_mask(missing_ratio)
        return self

    def set_mask_box(self, box_size: int, top: int = None, left: int = None):
        """Generate a centred (or custom-positioned) box mask."""
        self._mask = self._make_box_mask(box_size, top=top, left=left)
        return self

    # ------------------------------------------------------------------
    # Private mask builders
    # ------------------------------------------------------------------

    def _make_random_mask(self, missing_ratio: float) -> torch.Tensor:
        """1 = observed, 0 = missing.  Shape: (1, img_dim, img_dim)"""
        keep_prob = 1.0 - missing_ratio
        mask = (torch.rand(1, self.img_dim, self.img_dim) < keep_prob).float()
        return mask.to(self.device)

    def _make_box_mask(self, box_size: int, top: int = None, left: int = None) -> torch.Tensor:
        """Mask a centred square box. 1 = observed, 0 = missing (inside box)."""
        mask = torch.ones(1, self.img_dim, self.img_dim)
        t = top  if top  is not None else (self.img_dim - box_size) // 2
        l = left if left is not None else (self.img_dim - box_size) // 2
        mask[:, t:t + box_size, l:l + box_size] = 0.0
        return mask.to(self.device)

    def _load_mask_from_file(self, path: str) -> torch.Tensor:
        """Load a mask from a .npy or image file. Returns shape (1, H, W)."""
        if path.endswith('.npy'):
            arr  = np.load(path).astype(np.float32)
            mask = torch.from_numpy(arr)
        else:
            img  = Image.open(path).convert('L')
            arr  = np.array(img, dtype=np.float32) / 255.0
            mask = torch.from_numpy(arr)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)  # (1, H, W)
        return (mask > 0.5).float().to(self.device)

    # ------------------------------------------------------------------
    # Operator identity
    # ------------------------------------------------------------------

    @property
    def display_name(self):
        return 'inpainting'

    # ------------------------------------------------------------------
    # Linear operator  (H and H^T are identical: pointwise mask)
    # ------------------------------------------------------------------

    def forward(self, x, **kwargs):
        """H x  :  zero out missing pixels."""
        return x * self._mask

    def transpose(self, y):
        """H^T y  :  identical to forward (H^T H = diag(mask))."""
        return y * self._mask

    # ------------------------------------------------------------------
    # Proximal / sampling steps
    # ------------------------------------------------------------------

    def proximal_generator(self, x, y, sigma, rho):
        """
        Exact sample from the conditional Gaussian (eq. 15 / 17).
        Uses Sherman-Morrison-Woodbury (eq. 18): Q_x^{-1} is diagonal.

        From eq. (18):
            Q_x^{-1} = rho^2 * (I_N - (rho^2 / (sigma^2 + rho^2)) * H^T H)

        Diagonal entries:
            var[i] = sigma^2 * rho^2 / (sigma^2 + rho^2)   if observed  (mask=1)
            var[i] = rho^2                                   if missing   (mask=0)

        Mean (eq. 17):
            mu_x = Q_x^{-1} * (H^T y / sigma^2 + x / rho^2)
            -> observed : weighted combination of y and x (z)
            -> missing  : rho^2 * x/rho^2 = x  ✓  (samples from prior N(z, rho^2))
        """
        sigma2 = sigma ** 2
        rho2   = rho   ** 2

        # Diagonal of Q_x^{-1} from eq. (18)
        var_obs   = (sigma2 * rho2) / (sigma2 + rho2) * torch.ones_like(x)
        var_unobs = rho2 * torch.ones_like(x)   # ✅ rho^2 for missing pixels
        diag_cov  = torch.where(self._mask > 0.5, var_obs, var_unobs)

        # RHS = H^T y / sigma^2 + z / rho^2   (x plays the role of z here)
        rhs  = self.transpose(y) / sigma2 + x / rho2
        mu_x = diag_cov * rhs

        # Sample: mu_x + sqrt(Q_x^{-1}) * eps
        noise = torch.sqrt(diag_cov) * torch.randn_like(x)
        return mu_x + noise

    def proximal_for_admm(self, x, y, rho):
        """
        Proximal operator for ADMM / MAP estimation.

            z[i] = (y[i] + rho * x[i]) / (1 + rho)   if observed
            z[i] = x[i]                                if missing
        """
        z_obs = (self.transpose(y) + rho * x) / (1.0 + rho)
        return torch.where(self._mask > 0.5, z_obs, x)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(self, gt, y):
        """Initialise with the zero-filled observation."""
        return y.clone()
