import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from scipy.ndimage import map_coordinates, gaussian_filter
import cv2

def gaussian_kernel_fourier_normalized(shape, sigma):
    """
    Create a Gaussian filter in the Fourier domain with normalized frequency coordinates.
    The sigma is independent of the image size.
    """
    h, w = shape
    freq_y = np.fft.fftfreq(h)
    freq_x = np.fft.fftfreq(w)
    freq_x, freq_y = np.meshgrid(freq_x, freq_y)
    gaussian_filter = np.exp(-2 * (np.pi**2) * (sigma**2) * (freq_x**2 + freq_y**2))
    return gaussian_filter

def apply_fourier_gaussian_blur(image, sigma):
    """
    Apply Gaussian blur in the Fourier domain.
    """
    fft_image = np.fft.fftshift(np.fft.fft2(image))
    gaussian_filter = gaussian_kernel_fourier_normalized(image.shape, sigma)
    filtered_fft = fft_image * gaussian_filter
    blurred_image = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_fft)))
    return blurred_image

def five_point_laplacian(x):
    """Computes the five-point Laplacian of a 2D array."""
    laplacian_kernel = np.array([[0,  1,  0],
                                  [1, -4,  1],
                                  [0,  1,  0]])
    return convolve2d(x, laplacian_kernel, mode='same', boundary='symm')

def gradient(x, delta=[1.0, 2.0]):
    """Computes a simple directional gradient approximation."""
    dx = np.gradient(x, axis=0) * delta[0]
    dy = np.gradient(x, axis=1) * delta[1]
    return dx + dy

def gaussian_semigroup(x, sigma=3):
    """Applies a Gaussian blur as the semigroup G."""
    return gaussian_filter(x, sigma=sigma)

def riesz_blur(x, alpha=0.6):
    """
    Apply a Riesz potential of order alpha using spatial kernel (||x||^{alpha-2})
    and convolution via FFT for 2D or 3D input.

    Parameters
    ----------
    x : np.ndarray
        Input image, shape (H, W) or (H, W, B)
    alpha : float
        Order of the Riesz potential (typically 0 < alpha < 2)

    Returns
    -------
    y_out : np.ndarray
        Blurred image(s), same shape as input, normalized per-channel
    """
    assert x.ndim in [2, 3], "Input must be 2D or 3D (H,W) or (H,W,B)"
    if x.ndim == 2:
        x = x[..., np.newaxis]  # make it (H, W, 1)

    H, W, B = x.shape

    # Build Riesz kernel
    y = np.arange(-H//2, H//2)
    x_ = np.arange(-W//2, W//2)
    X, Y = np.meshgrid(x_, y)
    R = np.sqrt(X**2 + Y**2)
    R[R == 0] = 1  # avoid division by zero

    kernel = R ** (alpha - 2)
    kernel /= kernel.sum()
    kernel = np.fft.fftshift(kernel)
    K_fft = np.fft.fft2(kernel)

    # Apply convolution in frequency domain per channel
    y_out = np.empty_like(x, dtype=np.float32)
    for b in range(B):
        X_fft = np.fft.fft2(x[..., b])
        Y_fft = X_fft * K_fft
        y = np.fft.ifft2(Y_fft).real
        y = (y - y.mean()) / y.std()
        y_out[..., b] = y.astype(np.float32)

    if y_out.shape[-1] == 1:
        return y_out[..., 0]  # return 2D if input was 2D
    return y_out

def warp_image(image, vector_field):
    h, w = image.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')

    new_x = np.clip(x + vector_field[..., 0], 0, w - 1).astype(np.float32)
    new_y = np.clip(y + vector_field[..., 1], 0, h - 1).astype(np.float32)

    return cv2.remap(image, new_x, new_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def add_noise(x, factor=2):
    """Adds Gaussian noise to the image."""
    noise = factor * np.random.randn(*x.shape)
    return x + noise

def up_scale(x, factor=8):
    mask = np.zeros((factor, factor), dtype=int)
    center = factor // 2
    mask[center, center] = 1  # Place the 1 at the center of each block
    x2 = np.kron(x, mask)
    return x2

def compute_divergence(flow):
    """
    Compute the divergence ∂u/∂x + ∂v/∂y for a 2D flow field (H, W, 2)
    """
    u, v = flow[..., 0], flow[..., 1]
    du_dx = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / 2
    dv_dy = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / 2
    return du_dx + dv_dy

def backwarp_image(image, flow, backdiv=0.0, backdet=None, mode='bilinear'):
    """
    Applies backward warping of `image` according to `flow`.
    - image: (H, W) or (H, W, C)
    - flow: (H, W, 2)
    - backdiv: scalar for divergence-based rescaling
    - backdet: ignored here, but can be added like in C code
    """
    H, W = flow.shape[:2]
    if image.ndim == 2:
        image = image[..., np.newaxis]
    C = image.shape[2]

    # Create sampling grid
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    coords_x = xx + flow[..., 0]
    coords_y = yy + flow[..., 1]

    coords = np.stack([coords_y.ravel(), coords_x.ravel()], axis=0)

    # Output array
    output = np.zeros((H, W, C), dtype=image.dtype)
    for c in range(C):
        output[..., c] = map_coordinates(image[..., c], coords, order=1, mode='nearest').reshape(H, W)

    # Apply divergence rescaling if needed
    if backdiv != 0:
        div = compute_divergence(flow)
        factor = np.exp(backdiv * div)
        output *= factor[..., np.newaxis]

    return output.squeeze()

def qauto(img, clip_percent=1.0):
    """
    Quantile-based automatic contrast stretching (qauto).
    
    Parameters
    ----------
    img : np.ndarray
        Input image (2D grayscale or 3D with shape HxWxC)
    clip_percent : float
        Percentage of pixels to clip (total). For example, 1.0 clips 0.5% low + 0.5% high.

    Returns
    -------
    np.ndarray
        Contrast-stretched image in [0, 1] (same shape as input)
    """
    assert 0 <= clip_percent < 100, "clip_percent must be in [0, 100)"
    if img.ndim == 2:
        img = img[..., np.newaxis]  # convert to (H, W, 1)
    H, W, C = img.shape
    result = np.empty_like(img, dtype=np.float32)

    q_low = clip_percent / 200
    q_high = 1 - q_low

    for c in range(C):
        channel = img[..., c]
        lo = np.quantile(channel, q_low)
        hi = np.quantile(channel, q_high)
        stretched = np.clip((channel - lo) / (hi - lo + 1e-8), 0, 1)
        result[..., c] = stretched

    if result.shape[-1] == 1:
        return result[..., 0]  # return 2D if input was 2D
    return result

def to_cv2_image(img):
    """
    Convert a float32 image (2D or 3D) in range [0,1] to uint8 [0,255] for cv2.imwrite.

    Parameters
    ----------
    img : np.ndarray
        Input image (2D or 3D), values assumed to be in [0, 1]

    Returns
    -------
    np.ndarray
        uint8 image ready for saving with cv2
    """
    img = np.clip(img, 0, 1)  # ensure range
    img = (img * 255).round().astype(np.uint8)
    if img.ndim == 2:
        return img
    elif img.ndim == 3 and img.shape[2] == 1:
        return img[..., 0]  # remove singleton channel
    elif img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # convert RGB -> BGR
    else:
        raise ValueError("Invalid image shape for saving with cv2")

def create_srefcifromci(DataMatrix):
    x2 = up_scale(DataMatrix, factor=8)    
    x3 = 10 * apply_fourier_gaussian_blur(x2, 1.5)
    x4 = np.clip(five_point_laplacian(x3), 2, 16) #2#16
    x5 = riesz_blur(x4, 0.3)
    x6 = 20 + 0.8 * x5 + gradient(x5,[1.0, 2.0])
    return to_cv2_image(qauto(x6))

def create_I_from_noise_and_DataMatrix(DataMatrix, Noise):
    
    F = riesz_blur(Noise, 1)

    x6 = create_srefcifromci(DataMatrix)    
    x7 = backwarp_image(x6, 0.7 * F)
    x8 = to_cv2_image(qauto(x7) + 0.05 * np.random.randn(*x7.shape))
    ref_before_adding_noise = to_cv2_image(qauto(x6))
    return x8,ref_before_adding_noise

def create_srefcifromci_batch(ci_batch):
    """
    Apply the existing create_srefcifromci() function to each element in a batch of tensors.

    Args:
        ci_batch (torch.Tensor): Batch of binary or float images with shape (B, 1, H, W)

    Returns:
        torch.Tensor: Batch of generated srefci images with shape (B, 1, H_out, W_out)
    """
    import torch
    assert isinstance(ci_batch, torch.Tensor), "ci_batch must be a PyTorch tensor"
    assert ci_batch.dim() == 4 and ci_batch.size(1) == 1, "ci_batch must have shape (B, 1, H, W)"

    results = []
    for i in range(ci_batch.size(0)):
        # Convert the tensor (single image) to NumPy float32
        DataMatrix = ci_batch[i, 0].detach().cpu().numpy().astype(np.float32)
        
        # Call the original NumPy-based function
        srefci_np = create_srefcifromci(DataMatrix)
        
        # Convert the resulting NumPy array back to a PyTorch tensor
        srefci_tensor = torch.from_numpy(srefci_np).unsqueeze(0).unsqueeze(0).float()
        results.append(srefci_tensor)
    
    # Stack all processed images back into a batch
    return torch.cat(results, dim=0)