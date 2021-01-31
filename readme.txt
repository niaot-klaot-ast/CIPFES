CIPFES is a set of Python codes for the Characterization of Instrumental Profile for Fiber-fed Echelle Spectrograph.
It is based on the method proposed in the paper "The backbone-residual model – accurately characterising the instrumental profile of a fibre-fed echelle spectrograph".

global_IP_characterize_br.py: the IP characterization with the backbone-residual model.
global_IP_characterize_gh_dcp.py: the IP characterization with the Gauss-Hermite model and the discrete Chebyshev polynomial model.
IP_display_multiple_models.py: display the IP-characterization results of three models.
IP_accuracy_br_mc.py: determine the IP-characterization accuracy of the backbone-residual model.
IP_accuracy_gh_mc.py: determine the IP-characterization accuracy of the Gauss-Hermite model.
IP_accuracy_cheby_mc.py: determine the IP-characterization accuracy of the discrete Chebyshev polynomial model.
IP_accuracy_sg_mc.py: determine the IP-characterization accuracy of the super Gaussian model.
ThAr_line_width_histogram.py: show the histogram of the intrinsic widths of ThAr emission lines.

a201709220022.fits-a201709220025.fits: the 1D spectra of four sequential ThAr exposures on the Fiber-fed Echelle Spectrograph (HRS) of Chinese Xinglong 2.16m Telescope on September 22, 2017.
a201709220026.fits: the 1D spectrum of an astro-comb exposure on HRS.

table1.all.txt: Table 1 in "Redman S. L., Nave G., Sansonetti C. J., 2014, ApJS, 211, 4" which records the measurement results of ThAr lines with the NIST 2 m Fourier transform spectrometer.