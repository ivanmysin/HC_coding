import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.signal import oaconvolve as convolve2d
from scipy.signal import hilbert
from scipy.ndimage import convolve1d
from pprint import pprint
from sympy import diff, lambdify, exp, Symbol
import matplotlib.pyplot as plt


class HyperColumn:
    def __init__(self, cent_x, cent_y, xx, yy, directions, sigmas, sgmGauss, params={}):

        self.sgmGauss = sgmGauss
        self.params = params
        self.xx = np.copy(xx)  #
        self.yy = np.copy(yy)  #
        self.Nx, self.Ny = xx.shape

        self.x = np.copy(self.xx[:, 0])
        self.y = np.copy(self.yy[0, :])

        self.cent_x_idx = np.argmin(np.abs(self.x - cent_x))
        self.cent_y_idx = np.argmin(np.abs(self.y - cent_y))
        self.cent_x = self.x[self.cent_x_idx]  # corrected coordinates of HC center on the mesh
        self.cent_y = self.y[self.cent_y_idx]

        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        self.directions = directions
        self.dir = []
        self.sigmas = sigmas

        self.approximate_hilbert_kernel()

        self.rot_xx = []
        self.rot_yy = []
        self.rot_xc = []
        self.rot_yc = []

        self.params = params

        self.mexican_hats_y = []
        self.dGauss_x = []
        self.H_approxed_phi = []
        self.gaussian = np.ones_like(self.xx)

        self.rotate_xy()
        self.compute_kernels()

        self.normalization_factor_H = 1

    #    def derivate_gaussian_by_x(self, sigma, xx, yy):
    #        sigma_x = sigma
    #        sigma_y = 0.1 * sigma_x
    #        gaussian_dx = -xx * np.exp(-yy**2 / (2 * sigma_y**2) - xx**2 / (2 * sigma_x**2)) / sigma_x**2
    #        return gaussian_dx

    def derivate_gaussian_by_x(self, sigma, xx, yy):
        gaussian_dx = -xx * np.exp(-xx ** 2 / (2 * sigma ** 2)) / np.sqrt(2 * np.pi) / sigma ** 2
        Gauss = np.exp(- self.yy ** 2 / (2 * self.sgmGauss ** 2)) / np.sqrt(2 * np.pi) / self.sgmGauss
        gaussian_dx *= Gauss * self.dy
        return gaussian_dx

    def sum_gaussian_derivatives(self, w, xx, yy, sigmas):
        dgsum = np.zeros_like(xx)
        for idx, sigma in enumerate(sigmas):
            gaussian_dx = self.derivate_gaussian_by_x(sigma, xx, yy)
            dgsum += w[idx] * gaussian_dx
        return dgsum

    def _hilbert_dgs_diff(self, w, xx, yy, sigmas):
        H = 1 / (np.pi * self.xx) * self.dx
        # H = H / np.sqrt( np.sum(H**2) )
        Gauss = np.exp(- self.yy ** 2 / (2 * self.sgmGauss ** 2)) / np.sqrt(2 * np.pi) / self.sgmGauss
        H *= Gauss * self.dy
        sum_dg = self.sum_gaussian_derivatives(w, xx, yy, sigmas)
        E = np.sum((H - sum_dg) ** 2)
        return E

    def approximate_hilbert_kernel(self):
        w0 = np.ones(self.sigmas.size, dtype=np.float64)
        opt_res = minimize(self._hilbert_dgs_diff, x0=w0, args=(self.xx, self.yy, self.sigmas))
        self.dg_weigths = opt_res.x
        #### np.save(self.file_saving_ws, self.dg_weigths)
        self.H_approxed = self.sum_gaussian_derivatives(self.dg_weigths, self.xx, self.yy, self.sigmas)  #

        self.normalization_factor_H = np.sqrt(np.sum(self.H_approxed[:, self.y.size // 2] ** 2))
        self.H_approxed_normed = self.H_approxed / self.normalization_factor_H
        H = 1 / (np.pi * self.xx) * self.dx
        # H = H / np.sqrt( np.sum(H**2) )
        Gauss = np.exp(-self.yy ** 2 / (2 * self.sgmGauss ** 2)) / np.sqrt(2 * np.pi) / self.sgmGauss
        H *= Gauss * self.dy

        # plt.plot(self.x, H[:, self.y.size // 2], label="Hyperb", linewidth=3)
        # plt.plot(self.x, self.H_approxed[:, self.y.size // 2], label="Approxed", linewidth=1)
        # plt.legend()
        # plt.show()

    def get_rickers(self, sigma, xx, yy):
        sigma_x = sigma
        sigma_y = 0.5 * sigma_x
        ricker = (-1 + xx ** 2 / sigma_x ** 2) * np.exp(
            -yy ** 2 / (2 * sigma_y ** 2) - xx ** 2 / (2 * sigma_x ** 2)) / sigma_x ** 2

        ricker = ricker / np.sqrt(np.sum(ricker ** 2))  # -ricker / np.sqrt(np.sum(ricker**2))
        return ricker

    def rotate_xy(self):
        for phi in self.directions:
            xx = self.xx * np.cos(phi) - self.yy * np.sin(phi)
            yy = self.xx * np.sin(phi) + self.yy * np.cos(phi)
            xc = self.cent_x * np.cos(phi) - self.cent_y * np.sin(phi)
            yc = self.cent_x * np.sin(phi) + self.cent_y * np.cos(phi)

            self.rot_xx.append(xx)
            self.rot_yy.append(yy)
            self.rot_xc.append(xc)
            self.rot_yc.append(yc)

    def x_rot_of_xy(self, x, y, phi):
        x_rot = x * np.cos(phi) - y * np.sin(phi)
        return x_rot

    def y_rot_of_xy(self, x, y, phi):
        y_rot = x * np.sin(phi) + y * np.cos(phi)
        return y_rot

    def compute_kernels(self):

        ## find receptive field form
        x = Symbol("x")
        y = Symbol("y")
        sigma_x = Symbol("sigma_x")
        sigma_y = Symbol("sigma_y")
        gaussian = exp(- (x ** 2 / (2 * sigma_x ** 2) + y ** 2 / (2 * sigma_y ** 2)))
        kernel_expr = diff(gaussian, y, 2)  # differentiate twice
        dGauss_x_expr = diff(gaussian, x, 1)
        # compile in function
        kernel_func = lambdify([x, y, sigma_x, sigma_y], kernel_expr)
        dGauss_x_func = lambdify([x, y, sigma_x, sigma_y], dGauss_x_expr)
        ####################################################################

        for phi_idx in range(len(self.directions)):

            xx = self.rot_xx[phi_idx]
            yy = self.rot_yy[phi_idx]
            H_approx = self.sum_gaussian_derivatives(self.dg_weigths, xx, yy, self.sigmas)

            # print( np.sum(  H_approx**2 ) )
            # H_approx = H_approx / np.sqrt( np.sum(  H_approx**2 ) )

            ####################################
            self.H_approxed_phi.append(H_approx)
            ####################################

        # Define mexican hats to find the direction
        for phi_idx in range(len(self.directions)):
            self.dir.append(self.directions[phi_idx])
            xx = self.rot_xx[phi_idx] - self.rot_xc[phi_idx]
            yy = self.rot_yy[phi_idx] - self.rot_yc[phi_idx]
            sigma_y = self.sgmGauss
            sigma_x = sigma_y #0.2 * sigma_y
            # hat_y = (-2/sigma_y**2 + 4*yy**2/sigma_y**4)  * np.exp(-xx**2/(2*sigma_x**2) - yy**2/(2*sigma_y**2)) / (2*np.pi*sigma_x*sigma_y)
            # hat_y = hat_y - np.sum(hat_y)/self.Nx/self.Ny

            hat_y = kernel_func(xx, yy, sigma_x, sigma_y)   # calculate kernel
            hat_y = hat_y / np.sqrt(np.sum(hat_y ** 2))     # normalize kernel

            dGauss_x = dGauss_x_func(xx, yy, sigma_x, sigma_y)      # 03.07.2022: it was sigma_y, sigma_y
            dGauss_x = dGauss_x / np.sqrt(np.sum(dGauss_x ** 2))

            #################################
            self.mexican_hats_y.append(hat_y)
            self.dGauss_x.append(dGauss_x)
            #################################

        sigma = self.sgmGauss # 01.07.2022 | 0.1  # np.max(self.sigmas)  # !!!!
        self.gaussian = np.exp(
            -(self.yy - self.cent_y)**2 / (2* sigma**2) - (self.xx - self.cent_x)**2 / (2 * sigma**2))
        self.gaussian /= np.sum(self.gaussian)

        self.d_gauss_dx = -(self.xx - self.cent_x) / (sigma ** 2) * self.gaussian
        # self.d_gauss_dx /= np.sum(self.d_gauss_dx)
        self.d_gauss_dy = -(self.yy - self.cent_y) / (sigma ** 2) * self.gaussian
        # self.d_gauss_dy /= np.sum(self.d_gauss_dy)

    ######################################
    def find_dominant_direction(self, U):
    ######################################

        responses = np.zeros_like(self.dir)  # responses distributed by angles

        for an_idx in range(len(self.directions)):
            a = self.dGauss_x[an_idx]
            #b = self.mexican_hats_y[an_idx]
            kernel = a #- b
            responses[an_idx] = np.sum(a * U) #- np.abs(np.sum(b * U))

            #fig, axes = plt.subplots(ncols=2, figsize=(30, 15), sharex=True, sharey=True)
            #axes[0].pcolor(self.xx,self.yy, U, cmap="rainbow")#, shading='auto')
            #axes[1].pcolor(self.xx,self.yy, kernel, cmap="rainbow")#, shading='auto')
            #axes[1].set_title(f"{ responses[an_idx] :.2f}"+"  "+f"{ self.directions[an_idx]/np.pi*180 :.0f}")
            #plt.show()
            #print("response=", f"{ responses[an_idx] :.2f}", " angle=", f"{ self.directions[an_idx]/np.pi*180 :.0f}")

        phi_idx = np.argmax(responses)
        phi = self.directions[phi_idx]  # angle corresponding to maximal response
        #print("phi=", f"{phi / np.pi * 180 :.0f}")
        return phi, phi_idx

    def _get_phi_0(self, slope, phi_train, x_train):
        # s = np.sum(np.cos(phi_train - 2 * np.pi * slope * x_train))
        # s += 1j * np.sum(np.sin(phi_train - 2 * np.pi * slope * x_train))

        s = np.sum(np.exp(1j * (phi_train - 2 * np.pi * slope * x_train)))
        phi_0 = np.angle(s)
        return phi_0

    def _get_Dist(self, slope, phi_train, x_train):
        phi_0 = self._get_phi_0(slope, phi_train, x_train)
        D = 2 * (1 - np.mean(np.cos(phi_train - 2 * np.pi * slope * x_train - phi_0)))
        if slope < 0.5:
            D += 100

        return D

    def find_peak_freq(self, phases_train, x_train, freq, y_train):
        # res = minimize_scalar(self._get_Dist, args=(phases_train, x_train), method='Golden')
        # np.savez("./results/saved.npz", phases_train, x_train)

        if self.params["use_circ_regression"]:
            slopes = np.linspace(0.5 * freq, 1.5 * freq, 200)
            D = np.zeros_like(slopes)
            for idx, slope in enumerate(slopes):
                D[idx] = self._get_Dist(slope, phases_train, x_train)
            slope = slopes[np.argmin(D)]

            res = minimize_scalar(self._get_Dist, args=(phases_train, x_train), bounds=[slope - 2, slope + 2],
                                  method='bounded')
            slope = float(res.x)
        else:
            idx1 = np.argmin(np.abs(x_train - 2 * self.dx) + np.abs(y_train))
            idx2 = np.argmin(np.abs(x_train + 2 * self.dx) + np.abs(y_train))
            dist_x = np.abs(x_train[idx2] - x_train[idx1])

            phase_diff = phases_train[idx2] - phases_train[idx1]
            phase_diff = phase_diff % (2 * np.pi)
            if phase_diff < 0:
                phase_diff += 2 * np.pi
            slope = phase_diff / (2 * np.pi * dist_x)

        return slope

    ######################################
    def encode(self, U, Contrasted_U):
    ######################################

        ## 1 ## Extract mean and gradient

        mean_intensity = np.sum(self.gaussian * U)
        Grad_x = -np.sum(self.d_gauss_dx * U)
        Grad_y = -np.sum(self.d_gauss_dy * U)
        UnoGrad = U - mean_intensity # 14.07.2022 - (self.xx - self.cent_x) * Grad_x - (self.yy - self.cent_y) * Grad_y


        ## 2 ## Find direction

        phi, phi_idx = self.find_dominant_direction(UnoGrad)

        ## 3 ## Analytical signal

        u_H_phi = convolve2d(UnoGrad, self.H_approxed_phi[phi_idx], mode="same")

        ###########################
        uu = UnoGrad + 1j * u_H_phi  # u_imag
        ###########################

        dxy_dist = np.sqrt(self.dx ** 2 + self.dy ** 2)
        RF_ = self.sgmGauss   # 06.07.2022 | 5 * dxy_dist

        ## 4 ## Find phase
            
        phase_0 = np.angle(uu[self.cent_x_idx, self.cent_y_idx])

        ## 5 ## Find frequency and amplitude, or Break position and amplitude
            
        selected_vals = (np.abs(
            self.x_rot_of_xy(self.xx - self.cent_x, self.yy - self.cent_y, phi)) < RF_) & (
                                np.abs(self.y_rot_of_xy(self.xx - self.cent_x, self.yy - self.cent_y,
                                                        phi)) < 0.7 * dxy_dist)
        #x_train = self.rot_xx[phi_idx][selected_vals].ravel()
        x_train = self.x_rot_of_xy((self.xx - self.cent_x)[selected_vals].ravel(), (self.yy - self.cent_y)[selected_vals].ravel(), phi)
        phases_train  = np.angle(uu[selected_vals]).ravel()
        ampl_train    = np.abs(uu[selected_vals]).ravel()
        I_train       = u_H_phi[selected_vals].ravel()
        u_train       = UnoGrad[selected_vals].ravel()
        Cu_train      = Contrasted_U[selected_vals].ravel()
        i_I_train_max = np.argmax(I_train)
        i_I_train_min = np.argmin(I_train)
        if I_train[i_I_train_max]>-I_train[i_I_train_min]: 
            aBreak = I_train[i_I_train_max]
            xBreak = x_train[i_I_train_max]
        else:
            aBreak = I_train[i_I_train_min]
            xBreak = x_train[i_I_train_min]

        # Find aBreak as a difference between mean values before and after xBreak

        #du_train=u_train
        #du_train[0]=0
        #for i in range(1,len(u_train)-1):
        #    dx=x_train[i] - x_train[i - 1]
        #    if abs(dx)>1e-6:
        #        du_train[i] = (u_train[i] - u_train[i-1])/dx
        #    else:
        #        du_train[i]=0
        #i_abs_dtrain_max = np.argmax(np.abs(du_train))
        #xBreak=x_train[i_abs_dtrain_max]
        xBreak=x_train[np.argmax(np.abs(Cu_train))]

        u_train_mean = sum(u_train)/len(u_train)
        u1=u_train_mean
        u2=u_train_mean
        i1=0
        i2=0
        x1=0
        x2=0
        for i in range(len(x_train)):
            if      x_train[i] < xBreak: # before Break
                u1 = u1 * i1 / (i1 + 1) + u_train[i] / (i1 + 1)
                x1 = x1 * i1 / (i1 + 1) + x_train[i] / (i1 + 1)
                i1 = i1 + 1
            else:
                if x_train[i] > xBreak: # after Break
                    u2 = u2 * i2 / (i2 + 1) + u_train[i] / (i2 + 1)
                    x2 = x2 * i2 / (i2 + 1) + x_train[i] / (i2 + 1)
                    i2 = i2 + 1
        aBreak = u1 - u2

        """
        fig, axes = plt.subplots(ncols=2, figsize=(30, 15), sharex=True, sharey=True)
        axes[0].pcolor(self.xx,self.yy, np.real(uu), cmap='gray')
        axes[0].scatter(self.xx[selected_vals],self.yy[selected_vals], s=150, color="green")
        plt.show()

        fig, ax2 = plt.subplots(nrows=1)
        ax2.plot(x_train, u_train, 'o', color='tab:brown', label='u_train')
        ax2.plot(x_train, I_train, 'o', color='tab:red', label='I_train')
        ax2.plot(x_train, Cu_train, 'o', color='tab:blue', label='Cu_train')
        ax2.plot(x1, u1, 'x', color='tab:green')
        ax2.plot(x2, u2, 'x', color='tab:green')
        ax2.plot(xBreak, u_train_mean, 'o', color='tab:orange')
        ax2.legend()
        plt.show()
        """

        ii = 0
        peak_freq = 0
        ampl = 0
        I_mean = 0
        for i in range(len(x_train)):
            x1 = x_train[i]
            i1 = np.argmin(np.abs(x_train - x1 - dxy_dist))
            x2 = x_train[i1]
            p1 = phases_train[i]
            p2 = phases_train[i1]
            I_mean = I_mean * i / (i+1) +            I_train[i] / (i+1)

            if (p1*p2>0) & (x2!=x1) & ((p2-p1)*(x2-x1)>0) & (abs(x2-x1)<2*dxy_dist) & (abs(x2-x1)>0.5*dxy_dist):
                ii = ii + 1
                peak_freq = peak_freq*(ii-1)/ii + (p2-p1)/(x2-x1)/(2*np.pi)/ii
                ampl = ampl * (ii - 1) / ii + ampl_train[i] / ii



        Encoded = {
            "peak_freq": peak_freq,
            "phase_0": phase_0,
            "abs": ampl,
            "dominant_direction": phi,
            "direction_idx": phi_idx,
            "mean_intensity": mean_intensity,
            "Grad_x": Grad_x,
            "Grad_y": Grad_y,
            "cent_x": self.cent_x,
            "cent_y": self.cent_y,
            "intensity_at_cent": U[self.cent_x_idx, self.cent_y_idx],
            "aBreak": aBreak, #aBreak-I_mean,
            "xBreak": xBreak,
            "beforeBreak": u1,
            "afterBreak": u2
        }

        return Encoded

    ###############################################################################
    ###############################################################################

    def decode(self, Encoded):
        U_restored = np.zeros_like(self.xx)

        Abs_HC = np.zeros_like(self.xx)
        Phases_HC = np.zeros_like(self.xx)
        Mean_U_HC = np.zeros_like(self.xx)

        Umean = freq_encoded["mean_intensity"]
        A = freq_encoded["abs"]
        peak_freq = freq_encoded["peak_freq"]
        phi_0 = freq_encoded["phi_0"]
        phi_idx = freq_encoded["direction_idx"]

        xhc_ = self.cent_x * np.cos(self.directions[phi_idx]) - self.cent_y * np.sin(self.directions[phi_idx])
        U_restored += A * np.cos(peak_freq * (self.rot_xx[phi_idx] - xhc_) * 2 * np.pi + phi_0)
        Phases_HC = peak_freq * (self.rot_xx[phi_idx] - xhc_) * 2 * np.pi + phi_0
        Abs_HC += A
        Mean_U_HC += Umean

        U_restored += Umean

        return U_restored, Abs_HC, Phases_HC, Mean_U_HC


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    params = {
        "use_circ_regression": False,
    }

    directions = np.linspace(-np.pi, np.pi, 16, endpoint=False)

    Nx = 200
    Ny = 200
    sgmGauss = 0.1

    nsigmas = 8
    sigminimum = 0.05
    sigmaximum = 0.005
    sigmas = np.linspace(sigminimum, sigmaximum, nsigmas)
    yy, xx = np.meshgrid(np.linspace(-1.0, 1.0, Ny), np.linspace(1.0, -1.0, Nx))

    image = np.zeros_like(xx)

    f = 15.0
    an = 0.25 * np.pi

    xx_ = xx * np.cos(an) - yy * np.sin(an)
    image = 0.5 * (np.cos(2 * np.pi * xx_ * f + 0.5 * np.pi) + 1) * np.exp(
        -0.5 * ((xx - 0.5) / 0.05) ** 2 - 0.5 * ((yy - 0.5) / 0.05) ** 2)

    hc = HyperColumn(0, 0, xx, yy, directions, sigmas, sgmGauss, params=params)

    for H_approxed_phi in hc.H_approxed_phi:
        image_im = convolve2d(image, H_approxed_phi, mode='same')
        fig, axes = plt.subplots(ncols=3, figsize=(20, 5), sharex=True, sharey=True)
        # plt.pcolormesh(hc.x, hc.y, hc.H_approxed_normed, cmap="rainbow", shading="auto")
        axes[0].pcolormesh(hc.x, hc.y, image, cmap="gray", shading="auto")
        axes[1].pcolormesh(hc.x, hc.y, hc.H_approxed_normed, cmap="gray", shading="auto")

        axes[2].pcolormesh(hc.x, hc.y, image_im, cmap="gray", shading="auto")

        # plt.plot(hc.x, hc.H_approxed_normed[100, :])

        # plt.figure()
        # plt.pcolormesh(hc.x, hc.y, image, cmap="rainbow", shading="auto")
        # plt.figure()
        # plt.pcolormesh(hc.x, hc.y, image_restored, cmap="rainbow", shading="auto")

        plt.show()