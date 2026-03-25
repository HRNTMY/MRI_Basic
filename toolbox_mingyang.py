import h5py
import numpy as np
import matplotlib.pyplot as plt


def sos_combine(coil_imgs, axis=0, keepdims=False):
    """
    Sum-of-Squares combine

    coil_imgs: np.ndarray
    axis: 沿哪个维度做 SoS（通常是 coil 维度）
    keepdims: 是否保留该维度

    return: 合并后的图像
    """
    return np.sqrt(np.sum(np.abs(coil_imgs)**2, axis=axis, keepdims=keepdims))


def create_mask(shape, accel=4, center_fraction=0.08):
    """Mask create

    Args:
        shape (_type_): (H, W)
        accel (int, optional): 加速因子. Defaults to 4.
        center_fraction (float, optional): 中心保留比例. Defaults to 0.08.
    """
    H, W = shape
    mask = np.zeros((H, W))

    # Center
    center_lines = int(H * center_fraction)
    start = H//2 - center_lines//2
    end = H//2 + center_lines//2
    mask[start:end, :] = 1

    prob = (H/accel - center_lines) / (H - center_lines)

    for i in range(H):
        if i < start or i >= end:
            if np.random.rand() < prob:
                mask[i, :] = 1
    
    return mask


def psnr(gt, pred):
    mse = np.mean((gt - pred)**2)
    return 20 * np.log10(np.max(gt) / np.sqrt(mse))



def simulate_coils(Nro, Npe, n_coils=8):
    """
    Output: (Ncoils, Npe, Nro) complex
    """

    x = np.linspace(-1, 1, Nro)
    y = np.linspace(-1, 1, Npe)
    X, Y = np.meshgrid(x, y)

    csm = []

    for i in range(n_coils):
        angle = 2 * np.pi * i / n_coils

        # 1️⃣ 空间中心（你原来就有 ✔️）
        cx, cy = 0.5 * np.cos(angle), 0.5 * np.sin(angle)

        # 2️⃣ 幅度（保持你原来的）
        mag = np.exp(-((X - cx)**2 + (Y - cy)**2) / 0.3)

        # 3️⃣ 相位（🔥 新增，关键）
        phase = np.exp(1j * (X * np.cos(angle) + Y * np.sin(angle)) * np.pi)

        # 4️⃣ 合成复数CSM
        sens = mag * phase

        csm.append(sens)

    return np.stack(csm, axis=0)  # (Ncoils, Npe, Nro)



def fft2c_mingyang(img):
    kspace = np.fft.fftshift(np.fft.fft2(img, axes=(-2, -1)), axes=(-2, -1))
    return kspace


def ifft2c_mingyang(kspace):
    img = np.fft.ifft2(np.fft.ifftshift(kspace, axes=(-2, -1)), axes=(-2, -1))
    return img



def show_mri(img, mode="mag", title=None):
    """
    MRI image visualization

    mode:
        'mag'       : magnitude
        'phase'     : phase
        'real'      : real part
        'imag'      : imaginary part
        'complex'   : magnitude + phase
    """

    if mode == "mag":
        data = np.abs(img)
        cmap = "gray"

    elif mode == "phase":
        data = np.angle(img)
        cmap = "twilight"

    elif mode == "real":
        data = np.real(img)
        cmap = "gray"

    elif mode == "imag":
        data = np.imag(img)
        cmap = "gray"
    elif mode == "kspace":
        data = np.log(np.abs(img) + 1)
        cmap = "gray"

    elif mode == "complex":

        fig, ax = plt.subplots(1,2, figsize=(8,4))

        mag = np.abs(img).T
        pha = np.angle(img).T

        ax[0].imshow(mag, cmap="gray")
        ax[0].set_title("Magnitude")
        ax[0].axis("off")

        ax[1].imshow(pha, cmap="twilight")
        ax[1].set_title("Phase")
        ax[1].axis("off")

        plt.show()
        return

    else:
        raise ValueError("mode error")

    dataT = data.T

    plt.imshow(dataT, cmap=cmap)

    if title:
        plt.title(title)

    plt.axis("off")
    plt.show()

