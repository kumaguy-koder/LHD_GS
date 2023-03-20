import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

if __name__ == '__main__':
    # 初期設定 ------------------------------------------------
    # 再生面の設定
    target_res = (1024, 1024)
    target_amp = np.zeros(target_res, dtype=float)
    target_amp[511, 521] = 1.0
    target_amp[511, 501] = 1.0
    target_amp[521, 511] = 1.0
    target_amp[501, 511] = 1.0
    # SLM面の初期振幅
    slm_amp = np.ones((target_res[0], target_res[1]), dtype=float)
    # SLM面の初期位相
    slm_phase = np.random.rand(target_res[0], target_res[1]) * (2 * np.pi)
    # SLM面の初期複素振幅
    slm_field = np.zeros(target_res, dtype=complex)

    # 最適化計算 ---------------------------------------------
    print('Now optimizing...')
    num_iters = 30
    for i in range(num_iters):
        print(str(i+1) + 'th')
        # SLM面の複素振幅設定
        slm_field.real = slm_amp * np.cos(slm_phase)
        slm_field.imag = slm_amp * np.sin(slm_phase)

        # 再生面へ伝搬
        recon_field = np.fft.fft2(slm_field)
        recon_field = np.fft.fftshift(recon_field)

        # 再生面の振幅
        recon_intensity = np.abs(recon_field)**2
        # 再生面の位相
        recon_phase = np.angle(recon_field)

        # 再生面における拘束条件の適用
        recon_field.real = target_amp * np.cos(recon_phase)
        recon_field.imag = target_amp * np.sin(recon_phase)

        # SLM面に逆伝搬
        recon_field = np.fft.ifftshift(recon_field)
        slm_field = np.fft.ifft2(recon_field)

        # SLM面の位相更新
        slm_phase = np.angle(slm_field)

    # 表示 ------------------------------------------------
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

    # 保存 -----------------------------------------------
    # SLMに表示できる画素値に正規化
    slm_phase = slm_phase + np.pi
    CGH = np.around(slm_phase / (2 * np.pi) * 122)
    CGH = np.uint8(CGH)
    # SLMサイズに合わせてクロップ
    CGH_img = Image.fromarray(CGH)
    CGH_img = CGH_img.crop((112, 212, 912, 812))
    # 書き出し
    CGH_img.save('./CGH.bmp')

    print("Successfully completed\n")






