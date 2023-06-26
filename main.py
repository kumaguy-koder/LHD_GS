import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

if __name__ == '__main__':
    # Initialization ------------------------------------------------
    # Target
    target_res = (1024, 1024)
    target_amp = np.zeros(target_res, dtype=float)
    target_amp[511, 521] = 1.0
    target_amp[511, 501] = 1.0
    target_amp[521, 511] = 1.0
    target_amp[501, 511] = 1.0
    # Initial amp. of SLM
    slm_amp = np.ones((target_res[0], target_res[1]), dtype=float)
    # Initial phase of SLM
    slm_phase = np.random.rand(target_res[0], target_res[1]) * (2 * np.pi)
    # Initial comp. amp. of SLM
    slm_field = np.zeros(target_res, dtype=complex)

    # Optimization ---------------------------------------------
    print('Now optimizing...')
    num_iters = 30
    for i in range(num_iters):
        print(str(i+1) + 'th')
        # Comp. amp. of SLM
        slm_field.real = slm_amp * np.cos(slm_phase)
        slm_field.imag = slm_amp * np.sin(slm_phase)

        # Propagation
        recon_field = np.fft.fft2(slm_field)
        recon_field = np.fft.fftshift(recon_field)

        # Amp. of reconstructed plane
        recon_intensity = np.abs(recon_field)**2
        # Phase of reconstructed plane
        recon_phase = np.angle(recon_field)

        # Constraints
        recon_field.real = target_amp * np.cos(recon_phase)
        recon_field.imag = target_amp * np.sin(recon_phase)

        # Inverse propagation
        recon_field = np.fft.ifftshift(recon_field)
        slm_field = np.fft.ifft2(recon_field)

        # Updating phase of SLM
        slm_phase = np.angle(slm_field)

    # Showing ------------------------------------------------
    fig = plt.figure(figsize=[15, 5])

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title("Target image")
    ax1.imshow(target_amp, cmap='gray')

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title("CGH")
    ax2.imshow(slm_phase, cmap='gray')

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_title("Reconstruction")
    ax3.imshow(recon_intensity, cmap='gray')

    plt.show()

    # Saving -----------------------------------------------
    # Normalization
    slm_phase = slm_phase + np.pi
    CGH = np.around(slm_phase / (2 * np.pi) * 122)
    CGH = np.uint8(CGH)
    # Cropping to SLM size
    CGH_img = Image.fromarray(CGH)
    CGH_img = CGH_img.crop((112, 212, 912, 812))
    # Exporting
    CGH_img.save('./CGH.bmp')

    print("Successfully completed\n")






