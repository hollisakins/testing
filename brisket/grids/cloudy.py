import numpy as np
import os
from astropy.io import fits
from ..console import setup_logger
from .grids import Grid
from .. import config

if "CLOUDY_DATA_PATH" in list(os.environ):
    cloudy_data_path = os.environ["CLOUDY_DATA_PATH"]
    cloudy_sed_dir = os.path.join(cloudy_data_path, 'SED')

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

except ImportError:
    rank = 0
    size = 1


def air_to_vac(wav_air):
    # from SDSS: 
    # AIR = VAC / (1.0 + 2.735182E-4 + 131.4182 / VAC^2 + 2.76249E8 / VAC^4)
    vac_wavs = np.logspace(2, 9, 10000)
    air_wavs = vac_wavs / (1.0 + 2.735182E-4 + 131.4182 / vac_wavs**2 + 2.76249E8 / vac_wavs**4)
    return np.interp(wav_air, air_wavs, vac_wavs)

from dataclasses import dataclass
@dataclass
class Line:
    name: str
    label: str
    wav: float
    cloudy_label: str

@dataclass
class LineList:
    lines: list[Line]

    @property
    def wavs(self):
        return np.array([line.wav for line in self.lines])

    @property
    def labels(self):
        return [line.label for line in self.lines]

    @property
    def names(self):
        return np.array([line.name for line in self.lines])

    @property
    def cloudy_labels(self):
        return [line.cloudy_label for line in self.lines]
    
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, name):
        for line in self.lines:
            if line.name == name:
                return line
        return None

# vacuum wavelengths compiled from NIST or SDSS data
lines = [
    # lyman series ###########################################################################################################################
    Line(name='Lya', label=r'${\rm Ly}\alpha$', wav=1215.670, cloudy_label='H  1  1215.67A'),
    Line(name='Lyb', label=r'${\rm Ly}\beta$', wav=1025.720, cloudy_label='H  1  1025.72A'),
    Line(name='Lyg', label=r'${\rm Ly}\gamma$', wav=972.537, cloudy_label='H  1  972.537A'),
    Line(name='Lyd', label=r'${\rm Ly}\delta$', wav=949.743, cloudy_label='H  1  949.743A'),
    Line(name='Lye', label=r'${\rm Ly}\epsilon$', wav=937.804, cloudy_label='H  1  937.804A'),
    Line(name='Ly7', label=r'${\rm Ly}7$', wav=930.748, cloudy_label='H  1  930.748A'),
    Line(name='Ly8', label=r'${\rm Ly}8$', wav=926.226, cloudy_label='H  1  926.226A'),
    Line(name='Ly9', label=r'${\rm Ly}9$', wav=923.150, cloudy_label='H  1  923.150A'),
    # balmer series ##########################################################################################################################
    Line(name='Ha', label=r'${\rm H}\alpha$', wav=air_to_vac(6562.819), cloudy_label='H  1  6562.81A'),
    Line(name='Hb', label=r'${\rm H}\beta$', wav=air_to_vac(4861.333), cloudy_label='H  1  4861.33A'),
    Line(name='Hg', label=r'${\rm H}\gamma$', wav=air_to_vac(4340.471), cloudy_label='H  1  4340.46A'),
    Line(name='Hd', label=r'${\rm H}\delta$', wav=air_to_vac(4101.742), cloudy_label='H  1  4101.73A'),
    Line(name='He', label=r'${\rm H}\epsilon$', wav=air_to_vac(3970.079), cloudy_label='H  1  3970.07A'),
    Line(name='H8', label=r'${\rm H}8$', wav=air_to_vac(3889.050), cloudy_label='H  1  3889.05A'),
    Line(name='H9', label=r'${\rm H}9$', wav=air_to_vac(3835.380), cloudy_label='H  1  3835.38A'),
    Line(name='H10', label=r'${\rm H}10$', wav=air_to_vac(3797.890), cloudy_label='H  1  3797.89A'),
    Line(name='H11', label=r'${\rm H}11$', wav=air_to_vac(3770.637), cloudy_label='H  1  3770.63A'),
    Line(name='H12', label=r'${\rm H}12$', wav=air_to_vac(3750.158), cloudy_label='H  1  3750.15A'),
    Line(name='H13', label=r'${\rm H}13$', wav=air_to_vac(3734.369), cloudy_label='H  1  3734.37A'),
    Line(name='H14', label=r'${\rm H}14$', wav=air_to_vac(3721.945), cloudy_label='H  1  3721.94A'),
    Line(name='H15', label=r'${\rm H}15$', wav=air_to_vac(3711.977), cloudy_label='H  1  3711.97A'),
    Line(name='H16', label=r'${\rm H}16$', wav=air_to_vac(3703.859), cloudy_label='H  1  3703.85A'),
    Line(name='H17', label=r'${\rm H}17$', wav=air_to_vac(3697.157), cloudy_label='H  1  3697.15A'),
    Line(name='H18', label=r'${\rm H}18$', wav=air_to_vac(3691.551), cloudy_label='H  1  3691.55A'),
    Line(name='H19', label=r'${\rm H}19$', wav=air_to_vac(3686.831), cloudy_label='H  1  3686.83A'),
    # paschen series #########################################################################################################################
    Line(name='Paa', label=r'${\rm Pa}\alpha$', wav=18756.1, cloudy_label='H  1  1.87510m'),
    Line(name='Pab', label=r'${\rm Pa}\beta$', wav=12821.6, cloudy_label='H  1  1.28181m'),
    Line(name='Pag', label=r'${\rm Pa}\gamma$', wav=air_to_vac(10938.086), cloudy_label='H  1  1.09381m'),
    Line(name='Pad', label=r'${\rm Pa}\delta$', wav=air_to_vac(10049.368), cloudy_label='H  1  1.00494m'),
    Line(name='Pae', label=r'${\rm Pa}\epsilon$', wav=air_to_vac(9545.969), cloudy_label='H  1  9545.97A'),
    Line(name='Pa9', label=r'${\rm Pa}9$', wav=air_to_vac(9229.014), cloudy_label='H  1  9229.02A'),
    Line(name='Pa10', label=r'${\rm Pa}10$', wav=air_to_vac(9014.909), cloudy_label='H  1  9014.91A'),
    Line(name='Pa11', label=r'${\rm Pa}11$', wav=air_to_vac(8862.782), cloudy_label='H  1  8862.79A'),
    Line(name='Pa12', label=r'${\rm Pa}12$', wav=air_to_vac(8750.472), cloudy_label='H  1  8750.48A'),
    Line(name='Pa13', label=r'${\rm Pa}13$', wav=air_to_vac(8665.019), cloudy_label='H  1  8665.02A'),
    Line(name='Pa14', label=r'${\rm Pa}14$', wav=air_to_vac(8598.392), cloudy_label='H  1  8598.40A'),
    Line(name='Pa15', label=r'${\rm Pa}15$', wav=air_to_vac(8545.383), cloudy_label='H  1  8545.39A'),
    Line(name='Pa16', label=r'${\rm Pa}16$', wav=air_to_vac(8502.483), cloudy_label='H  1  8502.49A'),
    Line(name='Pa17', label=r'${\rm Pa}17$', wav=air_to_vac(8467.254), cloudy_label='H  1  8467.26A'),
    Line(name='Pa18', label=r'${\rm Pa}18$', wav=air_to_vac(8437.956), cloudy_label='H  1  8437.96A'),
    Line(name='Pa19', label=r'${\rm Pa}19$', wav=air_to_vac(8413.318), cloudy_label='H  1  8413.32A'),
    Line(name='Pa20', label=r'${\rm Pa}20$', wav=air_to_vac(8392.397), cloudy_label='H  1  8392.40A'),
    # brackett series ########################################################################################################################
    Line(name='Bra', label=r'${\rm Br}\alpha$', wav=air_to_vac(40511.30), cloudy_label='H  1  4.05115m'),
    Line(name='Brb', label=r'${\rm Br}\alpha$', wav=air_to_vac(26251.29), cloudy_label='H  1  2.62515m'),
    Line(name='Brg', label=r'${\rm Br}\alpha$', wav=air_to_vac(21655.09), cloudy_label='H  1  2.16553m'),
    Line(name='Brd', label=r'${\rm Br}\alpha$', wav=air_to_vac(19445.40), cloudy_label='H  1  1.94456m'),
    Line(name='Bre', label=r'${\rm Br}\alpha$', wav=air_to_vac(18174.00), cloudy_label='H  1  1.81741m'),
    Line(name='Br10', label=r'${\rm Br}\alpha$', wav=air_to_vac(17362.00), cloudy_label='H  1  1.73621m'),
    # pfund series ###########################################################################################################################
    Line(name='Pf6', label=r'${\rm Pf}6$', wav=air_to_vac(74577.699), cloudy_label='H  1  7.45777m'),
    Line(name='Pf7', label=r'${\rm Pf}7$', wav=air_to_vac(46524.699), cloudy_label='H  1  4.65247m'),
    Line(name='Pf8', label=r'${\rm Pf}8$', wav=air_to_vac(37395.099), cloudy_label='H  1  3.73951m'),
    Line(name='Pf9', label=r'${\rm Pf}9$', wav=air_to_vac(32960.699), cloudy_label='H  1  3.29607m'),
    Line(name='Pf10', label=r'${\rm Pf}10$', wav=air_to_vac(30383.500), cloudy_label='H  1  3.03835m'),
    # humphreys series #######################################################################################################################
    Line(name='Hu7', label=r'${\rm Hu}7$', wav=air_to_vac(123684.000), cloudy_label='H  1  12.3684m'),
    Line(name='Hu8', label=r'${\rm Hu}8$', wav=air_to_vac(75003.800), cloudy_label='H  1  7.50038m'),
    Line(name='Hu9', label=r'${\rm Hu}9$', wav=air_to_vac(59065.500), cloudy_label='H  1  5.90655m'),
    Line(name='Hu10', label=r'${\rm Hu}10$', wav=air_to_vac(51272.199), cloudy_label='H  1  5.12722m'),
    # helium lines ###########################################################################################################################
    Line(name='HeI10830', label=r'${\rm He}\,I$', wav=air_to_vac(10830.340), cloudy_label='Blnd  1.08302m'),
    Line(name='HeI7065', label=r'${\rm He}\,I$', wav=air_to_vac(7065.196), cloudy_label='Blnd  7065.25A'),
    Line(name='HeI6678', label=r'${\rm He}\,I$', wav=air_to_vac(6678.15), cloudy_label='He 1  6678.15A'),
    Line(name='HeI5876', label=r'${\rm He}\,I$', wav=air_to_vac(5875.624), cloudy_label='Blnd  5875.66A'),
    Line(name='HeI4471', label=r'${\rm He}\,I$', wav=air_to_vac(4471.479), cloudy_label='Blnd  4471.50A'),
    Line(name='HeI3889', label=r'${\rm He}\,I$', wav=air_to_vac(3888.647), cloudy_label='He 1  3888.64A'),
    Line(name='HeI3188', label=r'${\rm He}\,I$', wav=air_to_vac(3187.745), cloudy_label='He 1  3187.74A'),
    Line(name='HeII4685', label=r'He\,II\,$\lambda 4685$', wav=air_to_vac(4685.710), cloudy_label='He 2  4685.68A'),
    Line(name='HeII3203', label=r'He\,II\,$\lambda 3203$', wav=air_to_vac(3203.100), cloudy_label='He 2  3203.08A'),
    Line(name='HeII2733', label=r'He\,II\,$\lambda 2733$', wav=air_to_vac(2733.289), cloudy_label='He 2  2733.28A'),
    Line(name='HeII1640', label=r'He\,II\,$\lambda 1640$', wav=1640.400, cloudy_label='He 2  1640.41A'),
    # carbon lines ###########################################################################################################################
    Line(name='[CI]9850', label=r'$[{\rm C}\,\textsc{i}]\,\lambda 9850$', wav=air_to_vac(9850.260), cloudy_label='Blnd  9850.00A'),
    Line(name='[CI]8727', label=r'$[{\rm C}\,\textsc{i}]\,\lambda 8727$', wav=air_to_vac(8727.130), cloudy_label='C  1  8727.13A'),
    Line(name='[CI]4621', label=r'$[{\rm C}\,\textsc{i}]\,\lambda 4621$', wav=air_to_vac(4621.570), cloudy_label='C  1  4621.57A'),
    Line(name='[CI](1-0)', label=r'$[{\rm C}\,\textsc{i}](1-0)$', wav=6095900., cloudy_label='C  1  609.590m'),
    Line(name='[CI](2-1)', label=r'$[{\rm C}\,\textsc{i}](2-1)$', wav=3702690., cloudy_label='C  1  370.269m'),
    Line(name='[CII]158', label=r'$[{\rm C}\,\textsc{ii}]\,158\,\mu$m', wav=1576360., cloudy_label='C  2  157.636m'),
    Line(name='CII]2326', label=r'${\rm C}\,\textsc{ii}]\,\lambda 2326', wav=2326., cloudy_label='Blnd  2326.00A'),
    Line(name='CII1335', label=r'${\rm C}\,\textsc{ii}\,\lambda 1335', wav=1335.708, cloudy_label='Blnd  1335.00A'),
    Line(name='CIII1907', label=r'${\rm C}\,\textsc{iii}]\,\lambda 1907$', wav=1906.624, cloudy_label='C  3  1908.73A'),
    Line(name='CIII1909', label=r'${\rm C}\,\textsc{iii}]\,\lambda 1909$', wav=1908.791, cloudy_label='C  3  1906.68A'),
    Line(name='CIV1548', label=r'${\rm C}\,\textsc{iv}]\,\lambda 1548$', wav=1548.187, cloudy_label='C  4  1548.19A'),
    Line(name='CIV1551', label=r'${\rm C}\,\textsc{iv}]\,\lambda 1551$', wav=1550.772, cloudy_label='C  4  1550.77A'),
    # nitrogen lines #########################################################################################################################
    Line(name='[NI]5200', label=r'$[{\rm N}\,\textsc{i}]\,\lambda 5200$', wav=air_to_vac(5200.257), cloudy_label='N  1  5200.26A'),
    Line(name='[NII]6583', label=r'$[{\rm N}\,\textsc{ii}]\,\lambda 6583$', wav=air_to_vac(6583.460), cloudy_label='N  2  6583.45A'),
    Line(name='[NII]6548', label=r'$[{\rm N}\,\textsc{ii}]\,\lambda 6548$', wav=air_to_vac(6548.050), cloudy_label='N  2  6548.05A'),
    Line(name='[NII]5754', label=r'$[{\rm N}\,\textsc{ii}]\,\lambda 5754$', wav=air_to_vac(5754.590), cloudy_label='N  2  5754.61A'),
    Line(name='[NII]2139', label=r'$[{\rm N}\,\textsc{ii}]\,\lambda 2139$', wav=air_to_vac(2139.010), cloudy_label='N  2  2139.01A'),
    Line(name='[NII]122', label=r'$[{\rm N}\,\textsc{ii}]\,122\,\mu$m', wav=1217670., cloudy_label='N  2  121.767m'),
    Line(name='[NII]205', label=r'$[{\rm N}\,\textsc{ii}]\,205\,\mu$m', wav=2052440., cloudy_label='N  2  205.244m'),
    Line(name='[NIII]57', label=r'$[{\rm N}\,\textsc{iii}]\,57\,\mu$m', wav=573238., cloudy_label='N  3  57.3238m'),
    # oxygen lines ###########################################################################################################################
    Line(name='OI8446', label=r'${\rm O}\,\textsc{i}\,\lambda 8446$', wav=air_to_vac(8446.359), cloudy_label='Blnd  8446.00A'),
    Line(name='OI7254', label=r'${\rm O}\,\textsc{i}\,\lambda 7254$', wav=air_to_vac(7254.448), cloudy_label='Blnd  8446.00A'),
    Line(name='[OI]6364', label=r'$[{\rm O}\,\textsc{i}]\,\lambda 6364$', wav=air_to_vac(6363.776), cloudy_label='O  1  6363.78A'),
    Line(name='[OI]6300', label=r'$[{\rm O}\,\textsc{i}]\,\lambda 6300$', wav=air_to_vac(6300.304), cloudy_label='O  1  6300.30A'),
    Line(name='[OI]63', label=r'$[{\rm O}\,\textsc{i}]\,63\,\mu$m', wav=631679., cloudy_label='O  1  63.1679m'),
    Line(name='[OI]145', label=r'$[{\rm O}\,\textsc{i}]\,145\,\mu$m', wav=1454950., cloudy_label='O  1  145.495m'),
    Line(name='[OII]7331', label=r'$[{\rm O}\,\textsc{ii}]\,\lambda 7331$', wav=air_to_vac(7330.730), cloudy_label='Blnd  7332.00A'),
    Line(name='[OII]7320', label=r'$[{\rm O}\,\textsc{ii}]\,\lambda 7320', wav=air_to_vac(7319.990), cloudy_label='Blnd  7323.00A'),
    Line(name='[OII]3729', label=r'$[{\rm O}\,\textsc{ii}]\,\lambda 3729', wav=air_to_vac(3728.815), cloudy_label='Blnd  3729.00A'),
    Line(name='[OII]3726', label=r'$[{\rm O}\,\textsc{ii}]\,\lambda 3726', wav=air_to_vac(3726.032), cloudy_label='Blnd  3726.00A'),
    Line(name='[OII]2471', label=r'$[{\rm O}\,\textsc{ii}]\,\lambda 2471', wav=2471.00, cloudy_label='Blnd  2471.00A'),
    Line(name='[OIII]5007', label=r'$[{\rm O}\,\textsc{iii}]\,\lambda 5007', wav=air_to_vac(5006.843), cloudy_label='O  3  5006.84A'),
    Line(name='[OIII]4959', label=r'$[{\rm O}\,\textsc{iii}]\,\lambda 4959', wav=air_to_vac(4958.911), cloudy_label='O  3  4958.91A'),
    Line(name='[OIII]4363', label=r'$[{\rm O}\,\textsc{iii}]\,\lambda 4363', wav=air_to_vac(4363.210), cloudy_label='Blnd  4363.00A'),
    Line(name='[OIII]2321', label=r'$[{\rm O}\,\textsc{iii}]\,\lambda 2321', wav=air_to_vac(2320.951), cloudy_label='O  3  2320.95A'),
    Line(name='OIII]1666', label=r'${\rm O}\,\textsc{iii}]\,\lambda 1666', wav=air_to_vac(1666.150), cloudy_label='O  3  1666.15A'),
    Line(name='OIII]1661', label=r'${\rm O}\,\textsc{iii}]\,\lambda 1661', wav=air_to_vac(1660.809), cloudy_label='O  3  1660.81A'),
    Line(name='[OIII]88', label=r'$[{\rm O}\,\textsc{iii}]\,88\,\mu$m', wav=883323., cloudy_label='O  3  88.3323m'),
    Line(name='[OIII]52', label=r'$[{\rm O}\,\textsc{iii}]\,52\,\mu$m', wav=518004., cloudy_label='O  3  51.8004m'),
    # neon lines #############################################################################################################################
    Line(name='NeII_12um', label='', wav=128101.0, cloudy_label='Ne 2  12.8101m'),
    Line(name='NeIII_15um', label='', wav=155509.0, cloudy_label='Ne 3  15.5509m'),
    Line(name='NeIII_36um', label='', wav=360036.0, cloudy_label='Ne 3  36.0036m'),
    Line(name='NeIII3967', label=r'$[{\rm Ne}\,\textsc{iii}]\,\lambda 3967$', wav=air_to_vac(3967.470), cloudy_label='Ne 3  3967.47A'),
    Line(name='NeIII3869', label=r'$[{\rm Ne}\,\textsc{iii}]\,\lambda 3869$', wav=air_to_vac(3868.760), cloudy_label='Ne 3  3868.76A'),
    Line(name='NeIII3342', label='', wav=air_to_vac(3342.180), cloudy_label='Ne 3  3342.18A'),
    Line(name='NeIII1814', label='', wav=1814.560, cloudy_label='Ne 3  1814.56A'),
    Line(name='NeIV2423', label=r'$[{\rm Ne}\,\textsc{iv}]\,\lambda 2423$', wav=air_to_vac(2421.66), cloudy_label='Ne 4  2421.66A'),
    Line(name='NeV3427', label=r'$[{\rm Ne}\,\textsc{v}]\,\lambda 3427$', wav=air_to_vac(3425.881), cloudy_label='Ne 5  3425.88A'),
    Line(name='MgII2796', label=r'$[{\rm Mg}\,\textsc{ii}]\,\lambda\lambda 2796$', wav=air_to_vac(2795.528), cloudy_label='Mg 2  2795.53A'),
    Line(name='MgII2803', label=r'$[{\rm Mg}\,\textsc{ii}]\,\lambda\lambda 2803$', wav=air_to_vac(2802.705), cloudy_label='Mg 2  2802.71A'),
    Line(name='Blnd  4720.00A', label='', wav=4.720000000000000000e+03, cloudy_label='Blnd  4720.00A'),
    Line(name='Si 2  34.8046m', label='', wav=3.480460000000000000e+05, cloudy_label='Si 2  34.8046m'),
    Line(name='S  2  1.03364m', label='', wav=1.033639999999999964e+04, cloudy_label='S  2  1.03364m'),
    Line(name='S  2  6730.82A', label='', wav=6.730819999999999709e+03, cloudy_label='S  2  6730.82A'),
    Line(name='S  2  6716.44A', label='', wav=6.716439999999999600e+03, cloudy_label='S  2  6716.44A'),
    Line(name='S  2  4068.60A', label='', wav=4.068599999999999909e+03, cloudy_label='S  2  4068.60A'),
    Line(name='S  2  4076.35A', label='', wav=4.076349999999999909e+03, cloudy_label='S  2  4076.35A'),
    Line(name='S  3  18.7078m', label='', wav=1.870780000000000000e+05, cloudy_label='S  3  18.7078m'),
    Line(name='S  3  33.4704m', label='', wav=3.347040000000000000e+05, cloudy_label='S  3  33.4704m'),
    Line(name='S  3  9530.62A', label='', wav=9.530620000000000800e+03, cloudy_label='S  3  9530.62A'),
    Line(name='S  3  9068.62A', label='', wav=9.068620000000000800e+03, cloudy_label='S  3  9068.62A'),
    Line(name='S  3  6312.06A', label='', wav=6.312060000000000400e+03, cloudy_label='S  3  6312.06A'),
    Line(name='S  3  3721.63A', label='', wav=3.721630000000000109e+03, cloudy_label='S  3  3721.63A'),
    Line(name='S  4  10.5076m', label='', wav=1.050760000000000000e+05, cloudy_label='S  4  10.5076m'),
    Line(name='Ar 2  6.98337m', label='', wav=6.983369999999999709e+04, cloudy_label='Ar 2  6.98337m'),
    Line(name='Ar 3  7135.79A', label='', wav=7.135789999999999964e+03, cloudy_label='Ar 3  7135.79A'),
    Line(name='Ar 3  7751.11A', label='', wav=7.751109999999999673e+03, cloudy_label='Ar 3  7751.11A'),
    Line(name='Ar 3  5191.82A', label='', wav=5.191819999999999709e+03, cloudy_label='Ar 3  5191.82A'),
    Line(name='Ar 3  3109.18A', label='', wav=3.109179999999999836e+03, cloudy_label='Ar 3  3109.18A'),
    Line(name='Ar 3  21.8253m', label='', wav=2.182530000000000000e+05, cloudy_label='Ar 3  21.8253m'),
    Line(name='Ar 3  8.98898m', label='', wav=8.988980000000000291e+04, cloudy_label='Ar 3  8.98898m'),
    Line(name='Ar 4  7332.15A', label='', wav=7.332149999999999636e+03, cloudy_label='Ar 4  7332.15A'),
    Line(name='Al 2  2669.15A', label='', wav=2.669150000000000091e+03, cloudy_label='Al 2  2669.15A'),
    Line(name='Al 2  2660.35A', label='', wav=2.660349999999999909e+03, cloudy_label='Al 2  2660.35A'),
    Line(name='Al 2  1855.93A', label='', wav=1.855930000000000064e+03, cloudy_label='Al 2  1855.93A'),
    Line(name='Al 2  1862.31A', label='', wav=1.862309999999999945e+03, cloudy_label='Al 2  1862.31A'),
    Line(name='Cl 2  14.3639m', label='', wav=1.436390000000000000e+05, cloudy_label='Cl 2  14.3639m'),
    Line(name='Cl 2  8578.70A', label='', wav=8.578700000000000728e+03, cloudy_label='Cl 2  8578.70A'),
    Line(name='Cl 2  9123.60A', label='', wav=9.123600000000000364e+03, cloudy_label='Cl 2  9123.60A'),
    Line(name='Cl 3  5537.87A', label='', wav=5.537869999999999891e+03, cloudy_label='Cl 3  5537.87A'),
    Line(name='Cl 3  5517.71A', label='', wav=5.517710000000000036e+03, cloudy_label='Cl 3  5517.71A'),
    Line(name='P  2  60.6263m', label='', wav=6.062630000000000000e+05, cloudy_label='P  2  60.6263m'),
    Line(name='P  2  32.8620m', label='', wav=3.286200000000000000e+05, cloudy_label='P  2  32.8620m'),
    Line(name='Fe 2  1.25668m', label='', wav=1.256679999999999927e+04, cloudy_label='Fe 2  1.25668m'),
]
linelist = LineList(lines)
# linelist.add_row(['NIII',          r'N\,III\,$\lambda\lambda 1749$--$1753$',                 1749.246])
# linelist.add_row(['FeXI',          r'$[{\rm Fe}\,\textsc{xi}]$',                             air_to_vac(2648.710)])
# linelist.add_row(['HeIHeII',       r'He\,I\,$\lambda 3188+$He\,II\,$\lambda 3203$',          air_to_vac(3195.423)])
# linelist.add_row(['FeII',          r'$[{\rm Fe}\,\textsc{ii}]\,\lambda 4287$',               air_to_vac(4287.394)])
# linelist.add_row(['SiII',          r'${\rm Si}\,\textsc{ii}$',                               air_to_vac(6347.100)])
# linelist.add_row(['FeX',           r'$[{\rm Fe}\,\textsc{x}]$',                              air_to_vac(6374.510)])
# linelist.add_row(['SII6716',       r'$[{\rm S}\,\textsc{ii}]$\,$\lambda 6716$',              air_to_vac(6716.440)])
# linelist.add_row(['SII6731',       r'$[{\rm S}\,\textsc{ii}]$\,$\lambda 6731$',              air_to_vac(6730.810)])





def mpi_split_array(array):
    """ Distributes array elements to cores when using mpi. """
    if size > 1: # If running on more than one core

        n_per_core = array.shape[0]//size

        # How many are left over after division between cores
        remainder = array.shape[0]%size

        if rank == 0:
            if remainder == 0:
                core_array = array[:n_per_core, ...]

            else:
                core_array = array[:n_per_core+1, ...]

            for i in range(1, remainder):
                start = i*(n_per_core+1)
                stop = (i+1)*(n_per_core+1)
                comm.send(array[start:stop, ...], dest=i)

            for i in range(np.max([1, remainder]), size):
                start = remainder+i*n_per_core
                stop = remainder+(i+1)*n_per_core
                comm.send(array[start:stop, ...], dest=i)

        if rank != 0:
            core_array = comm.recv(source=0)

    else:
        core_array = array

    return core_array


def mpi_combine_array(core_array, total_len):
    """ Combines array sections from different cores. """
    if size > 1: # If running on more than one core

        n_per_core = total_len//size

        # How many are left over after division between cores
        remainder = total_len%size

        if rank != 0:
            comm.send(core_array, dest=0)
            array = None

        if rank == 0:
            array = np.zeros([total_len] + list(core_array.shape[1:]))
            array[:core_array.shape[0], ...] = core_array

            for i in range(1, remainder):
                start = i*(n_per_core+1)
                stop = (i+1)*(n_per_core+1)
                array[start:stop, ...] = comm.recv(source=i)

            for i in range(np.max([1, remainder]), size):
                start = remainder+i*n_per_core
                stop = remainder+(i+1)*n_per_core
                array[start:stop, ...] = comm.recv(source=i)

        array = comm.bcast(array, root=0)

    else:
        array = core_array

    return array

def logQ_from_logU(logU, lognH, logr):
    '''Compute logQ from logU, log(nH/cm^-3), and log(radius/pc). Osterbrok & Ferland eq. 14.7.'''
    U = np.power(10., logU)
    nH = np.power(10., lognH)
    r = np.power(10., logr) * 3.086e18  # pc -> cm
    c = 2.99e10  # cm s^-1
    return np.log10(4*np.pi * r**2 * c * nH * U)

def make_cloudy_input_file(dir, filename: str, params: dict):
    """Generates in input parameter file for cloudy. Much of this code is adapted from synthesizer"""

    # # Copy file with emission line names to the correct directory
    # if not os.path.exists(cloudy_data_path + "/brisket_lines.txt"):
    #     os.system("cp " + utils.install_dir + "/models/grids/cloudy_lines.txt "
    #               + cloudy_data_path + "/pipes_cloudy_lines.txt")

    f = open(f"{dir}/{filename}.in", "w+")
    # input ionizing spectrum
    f.write(f"table SED \"{filename}.sed\"\n")

    zmet = params['zmet'] # metallicity in solar units
    CO = params['CO'] # C/O ratio, relative to solar
    xid = params['xid'] # dust-to-gas ratio 
    lognH = params['lognH'] # log10 of hydrogen density in cm^-3
    logU = params['logU'] # log10 of ionization parameter

    
    if params['geometry'] == 'spherical':
        radius = params['radius']
        logr = np.log10(radius)
        logQ = logQ_from_logU(logU, lognH, logr)
        f.write("sphere\n")
        f.write(f"radius {logr:.3f} log parsecs\n")
    else:
        raise NotImplementedError("Only spherical geometry is currently supported")

    if params["cosmic_rays"] is not None:
        f.write("cosmic rays background\n")

    if params["CMB"] is not None:
        f.write(f'CMB {params["z"]}\n')

    f.write(f"hden {lognH:.3f} log\n")
    f.write(f"Q(H) = {logQ:.3f} log\n")
    
    # # constant density flag
    # if params["constant_density"] is not None:
    #     cinput.append("constant density\n")

    # # constant pressure flag
    # if params["constant_pressure"] is not None:
    #     cinput.append("constant pressure\n")

    # if (params["constant_density"] is not None) and (
    #     params["constant_pressure"] is not None
    # ):
    #     raise InconsistentArguments(
    #         """Cannot specify both constant pressure and density"""
    #     )

    # # covering factor
    # if params["covering_factor"] is not None:
    #     cinput.append(f'covering factor {params["covering_factor"]} linear\n')



    #######################################################################################################################################
    # Chemical composition ################################################################################################################
    #######################################################################################################################################
    if not 'abundances' in params:
        # set the default abundance model
        params['abundances'] = {'model': 'Gutkin16'}

    if params['abundances']['model'] == 'Gutkin16':
        zsol = 0.01508
        numbers = np.arange(1, 31)
        masses = np.array([1.0080, 4.00260,7.0,9.012183,10.81,12.011,14.007,15.999,18.99840316,20.180,22.9897693,24.305,26.981538,28.085,30.97376200,32.07,35.45,39.9,39.0983,40.08,44.95591,47.867,50.9415,51.996,54.93804,55.84,58.93319,58.693,63.55,65.4])
        elements = np.array(['hydrogen', 'helium','lithium','beryllium','boron','carbon','nitrogen','oxygen','fluorine','neon','sodium','magnesium','aluminium','silicon','phosphorus','sulphur','chlorine','argon','potassium','calcium','scandium','titanium','vanadium','chromium','manganese','iron','cobalt','nickel','copper','zinc'])
        abundances = np.array([0, -1.01, -10.99, -10.63, -9.47, -3.53, -4.32, -3.17, -7.47, -4.01, -5.70, -4.45, -5.56, -4.48, -6.57, -4.87, -6.53, -5.63, -6.92, -5.67, -8.86, -7.01, -8.03, -6.36, -6.64, -4.51, -7.11, -5.78, -7.82, -7.43])
        fdpl = np.array([0, 0, 0.84, 0.4, 0.87, 0.5, 0, 0.3, 0.7, 0, 0.75, 0.8, 0.98, 0.9, 0.75, 0, 0.5, 0, 0.7, 0.997, 0.995, 0.992, 0.994, 0.994, 0.95, 0.99, 0.99, 0.96, 0.9, 0.75])
        abundances[numbers>=3] += np.log10(zmet)
        # z_ism = np.sum(np.power(10.,abundances[numbers>=3])*masses[numbers>=3])/np.sum(np.power(10.,abundances)*masses)/zsol
        
        # primary+secondary nitrogren abundance prescription from Gutkin+16
        i_N = np.where(elements == 'nitrogen')[0][0]
        i_O = np.where(elements == 'oxygen')[0][0]
        abundances[i_N] = np.log10(0.41 * np.power(10., abundances[i_O]) * (np.power(10., -1.6) + np.power(10., 2.33+abundances[i_O])))
        z_ism = np.sum(np.power(10.,abundances[numbers>=3])*masses[numbers>=3])/np.sum(np.power(10.,abundances)*masses)/zsol
        abundances[numbers>=3] += np.log10(zmet/z_ism)

        # variable C/O prescription from Gutkin+16
        i_C = np.where(elements == 'carbon')[0][0]
        # CO_sol = np.power(10, abundances[i_C])/np.power(10., abundances[i_O])
        abundances[i_C] += np.log10(CO)
        z_ism = np.sum(np.power(10.,abundances[numbers>=3])*masses[numbers>=3])/np.sum(np.power(10.,abundances)*masses)/zsol
        abundances[numbers>=3] += np.log10(zmet/z_ism)

        # He abundance scaling w/ metallicity from Gutkin+16, following Bressan+12
        i_He = np.where(elements == 'helium')[0][0]
        abundances[i_He] = np.log10(np.power(10., abundances[i_He]) + 1.7756*zmet*zsol) 

        # adjust the depletion factors based on the dust-to-metals mass ratio
        xid0 = 0.36
        for i in range(len(fdpl)):
            if not fdpl[i] == 0:
                fdpl[i] = np.interp(xid, [0, xid0, 1], [0, fdpl[i], 1])

        abundances_depleted = np.log10(np.power(10., abundances * 1-fdpl))

        for i in range(len(elements)):
            # print(f'element abundance {elements[i]} {abundances_depleted[i]:.2f} no grains')
            f.write(f'element abundance {elements[i]} {abundances_depleted[i]:.2f} no grains\n')
    else:
        raise ValueError('Unknown abundance model, currently only Gutkin16 is supported')
    

    #######################################################################################################################################
    # Processing commands #################################################################################################################
    #######################################################################################################################################
    if params["iterate_to_convergence"] is not None:
        f.write("iterate to convergence\n")

    # if params["T_floor"] is not None:
    #     f.write(f'set temperature floor {params["T_floor"]} linear\n')

    # if params["stop_T"] is not None:
    #     f.write(f'stop temperature {params["stop_T"]}K\n')

    # if params["stop_efrac"] is not None:
    #     f.write(f'stop efrac {params["stop_efrac"]}\n')

    # if params["stop_column_density"] is not None:
    #     f.write(f'stop column density {params["stop_column_density"]}\n')
    #     # For some horrible reason the above is ignored in favour of a
    #     # built in temperature stop (4000K) unless that is turned off.
    #     f.write("stop temperature off\n")



    # # --- output commands
    # # cinput.append(f'print line vacuum\n')  # output vacuum wavelengths
    # cinput.append(
    #     f'set continuum resolution {params["resolution"]}\n'
    # )  # set the continuum resolution
    # cinput.append(f'save overview  "{model_name}.ovr" last\n')


    #######################################################################################################################################
    # Output commands #####################################################################################################################
    #######################################################################################################################################
    f.write(f'save last outward continuum "{filename}.cont" units Angstroms\n')
    f.write(f'save last line list intrinsic absolute column "{filename}.lines" "brisket_cloudy_lines.txt" \n')

    # f.write(f'save line list column absolute last units angstroms "{filename}.intrinsic_elin" "linelist.dat"\n')
    # f.write(f'save line list emergent column absolute last units angstroms "{filename}.emergent_elin" "linelist.dat"\n')
    
    
    # # save input file
    # if output_dir is not None:
    #     print(f"created input file: {output_dir}/{model_name}.in")
    #     open(f"{output_dir}/{model_name}.in", "w").writelines(cinput)

    # f.write("##### Output continuum and lines #####\n")
    # f.write("set save prefix \"" + "%.5f" % age + "\"\n")
    # f.write("save last outward continuum \".econ\" units microns\n")
    # f.write("save last line list intrinsic absolute column"
    #         + " \".lines\" \"pipes_cloudy_lines.txt\"\n")

    # f.write("########################################")

    f.close()


# def run_cloudy_model(age, zmet, logU, path):
#     """ Run an individual cloudy model. """

#     make_cloudy_sed_file(age, zmet)
#     make_cloudy_input_file(age, zmet, logU, path)
#     os.chdir(path + "/cloudy_temp_files/"
#              + "logU_" + "%.1f" % logU + "_zmet_" + "%.3f" % zmet)

#     os.system(os.environ["CLOUDY_EXE"] + " -r " + "%.5f" % age)
#     os.chdir("../../..")


# def extract_cloudy_results(age, zmet, logU, path):
#     """ Loads individual cloudy results from the output files and converts the
#     units to L_sol/A for continuum, L_sol for lines. """

#     cloudy_lines = np.loadtxt(path + "/cloudy_temp_files/"
#                               + "logU_" + "%.1f" % logU
#                               + "_zmet_" + "%.3f" % zmet + "/" + "%.5f" % age
#                               + ".lines", usecols=(1),
#                               delimiter="\t", skiprows=2)

#     cloudy_cont = np.loadtxt(path + "/cloudy_temp_files/"
#                              + "logU_" + "%.1f" % logU + "_zmet_"
#                              + "%.3f" % zmet + "/" + "%.5f" % age + ".econ",
#                              usecols=(0, 3, 8))[::-1, :]

#     # wavelengths from microns to angstroms
#     cloudy_cont[:, 0] *= 10**4

#     # subtract lines from nebular continuum model
#     cloudy_cont[:, 1] -= cloudy_cont[:, 2]

#     # continuum from erg/s to erg/s/A.
#     cloudy_cont[:, 1] /= cloudy_cont[:, 0]

#     # Get bagpipes input spectrum: angstroms, erg/s/A
#     input_spectrum = get_bagpipes_spectrum(age, zmet)

#     # Total ionizing flux in the bagpipes model in erg/s
#     ionizing_spec = input_spectrum[(input_spectrum[:, 0] <= 911.8), 1]
#     ionizing_wavs = input_spectrum[(input_spectrum[:, 0] <= 911.8), 0]
#     pipes_ionizing_flux = np.trapz(ionizing_spec, x=ionizing_wavs)

#     # Total ionizing flux in the cloudy outputs in erg/s
#     cloudy_ionizing_flux = np.sum(cloudy_lines) + np.trapz(cloudy_cont[:, 1],
#                                                            x=cloudy_cont[:, 0])

#     # Normalise cloudy fluxes to the level of the input bagpipes model
#     cloudy_lines *= pipes_ionizing_flux/cloudy_ionizing_flux
#     cloudy_cont[:, 1] *= pipes_ionizing_flux/cloudy_ionizing_flux

#     # Convert cloudy fluxes from erg/s/A to L_sol/A
#     cloudy_lines /= 3.826*10**33
#     cloudy_cont[:, 1] /= 3.826*10**33

#     nlines = config.wavelengths.shape[0]
#     cloudy_cont_resampled = np.zeros((nlines, 2))

#     # Resample the nebular continuum onto wavelengths of stellar models
#     cloudy_cont_resampled[:, 0] = config.wavelengths
#     cloudy_cont_resampled[:, 1] = np.interp(cloudy_cont_resampled[:, 0],
#                                             cloudy_cont[:, 0],
#                                             cloudy_cont[:, 1])

#     return cloudy_cont_resampled[:, 1], cloudy_lines


# def compile_cloudy_grid(path):

#     line_wavs = np.loadtxt(utils.install_dir
#                            + "/models/grids/cloudy_linewavs.txt")

#     for logU in config.logU:
#         for zmet in config.metallicities:

#             print("logU: " + str(np.round(logU, 1))
#                   + ", zmet: " + str(np.round(zmet, 4)))

#             mask = (config.age_sampling < age_lim)
#             contgrid = np.zeros((config.age_sampling[mask].shape[0]+1,
#                                  config.wavelengths.shape[0]+1))

#             contgrid[0, 1:] = config.wavelengths
#             contgrid[1:, 0] = config.age_sampling[config.age_sampling < age_lim]

#             linegrid = np.zeros((config.age_sampling[mask].shape[0]+1,
#                                 line_wavs.shape[0]+1))

#             linegrid[0, 1:] = line_wavs
#             linegrid[1:, 0] = config.age_sampling[mask]

#             for i in range(config.age_sampling[mask].shape[0]):
#                 age = config.age_sampling[mask][i]
#                 cont_fluxes, line_fluxes = extract_cloudy_results(age*10**-9,
#                                                                   zmet, logU,
#                                                                   path)

#                 contgrid[i+1, 1:] = cont_fluxes
#                 linegrid[i+1, 1:] = line_fluxes

#             if not os.path.exists(path + "/cloudy_temp_files/grids"):
#                 os.mkdir(path + "/cloudy_temp_files/grids")

#             np.savetxt(path + "/cloudy_temp_files/grids/"
#                        + "zmet_" + str(zmet) + "_logU_" + str(logU)
#                        + ".neb_lines", linegrid)

#             np.savetxt(path + "/cloudy_temp_files/grids/"
#                        + "zmet_" + str(zmet) + "_logU_" + str(logU)
#                        + ".neb_cont", contgrid)

#     # Nebular grids
#     list_of_hdus_lines = [fits.PrimaryHDU()]
#     list_of_hdus_cont = [fits.PrimaryHDU()]

#     for logU in config.logU:
#         for zmet in config.metallicities:

#             line_data = np.loadtxt(path + "/cloudy_temp_files/"
#                                    + "grids/zmet_" + str(zmet)
#                                    + "_logU_" + str(logU) + ".neb_lines")

#             hdu_line = fits.ImageHDU(name="zmet_" + "%.3f" % zmet + "_logU_"
#                                      + "%.1f" % logU, data=line_data)

#             cont_data = np.loadtxt(path + "/cloudy_temp_files/"
#                                    + "grids/zmet_" + str(zmet)
#                                    + "_logU_" + str(logU) + ".neb_cont")

#             hdu_cont = fits.ImageHDU(name="zmet_" + "%.3f" % zmet + "_logU_"
#                                      + "%.1f" % logU, data=cont_data)

#             list_of_hdus_lines.append(hdu_line)
#             list_of_hdus_cont.append(hdu_cont)

#     hdulist_lines = fits.HDUList(hdus=list_of_hdus_lines)
#     hdulist_cont = fits.HDUList(hdus=list_of_hdus_cont)

#     hdulist_lines.writeto(path + "/cloudy_temp_files"
#                           + "/grids/bagpipes_nebular_line_grids.fits",
#                           overwrite=True)

#     hdulist_cont.writeto(path + "/cloudy_temp_files"
#                          + "/grids/bagpipes_nebular_cont_grids.fits",
#                          overwrite=True)


# def run_cloudy_grid(path=None):
#     """ Generate the whole grid of cloudy models and save to file. """

#     if path is None:
#         path = utils.working_dir

#     if rank == 0 and not os.path.exists(path + "/cloudy_temp_files"):
#         os.mkdir(path + "/cloudy_temp_files")

#     ages = config.age_sampling[config.age_sampling < age_lim]

#     n_models = config.logU.shape[0]*ages.shape[0]*config.metallicities.shape[0]

#     params = np.zeros((n_models, 3))

#     n = 0
#     for i in range(config.logU.shape[0]):
#         for j in range(config.metallicities.shape[0]):

#             # Make directory to store cloudy inputs/outputs
#             if rank == 0:
#                 if not os.path.exists(path + "/cloudy_temp_files/"
#                                       + "logU_" + "%.1f" % config.logU[i]
#                                       + "_zmet_" + "%.3f" % config.metallicities[j]):

#                     os.mkdir(path + "/cloudy_temp_files/"
#                              + "logU_" + "%.1f" % config.logU[i]
#                              + "_zmet_" + "%.3f" % config.metallicities[j])

#             # Populate array of parameter values
#             for k in range(ages.shape[0]):

#                 params[n, 0] = ages[k]
#                 params[n, 1] = config.metallicities[j]
#                 params[n, 2] = config.logU[i]
#                 n += 1

#     # Assign models to cores
#     thread_nos = mpi_split_array(np.arange(n_models))

#     # Run models assigned to this core
#     for n in thread_nos:
#         age = params[n, 0]
#         zmet = params[n, 1]
#         logU = params[n, 2]

#         print("logU: " + str(np.round(logU, 1)) + ", zmet: "
#               + str(np.round(zmet, 4)) + ", age: "
#               + str(np.round(age*10**-9, 5)))

#         run_cloudy_model(age*10**-9, zmet, logU, path)

#     # Combine arrays of models assigned to cores, checks all is finished
#     mpi_combine_array(thread_nos, n_models)

#     # Put the final grid fits files together
#     if rank == 0:
#         compile_cloudy_grid(path)

def make_cloudy_sed_file(output_dir, filename, wavs, fluxes):
    energy = 911.8/wavs
    nu = 2.998e8/(wavs*1e-10) # in Hz
    fluxes = fluxes * 3.826e33 / nu # in erg/s/Hz
    fluxes[fluxes <= 0] = np.power(10.,-99)
    energy = np.flip(energy)
    fluxes = np.flip(fluxes)
    np.savetxt(f"{output_dir}/{filename}.sed", 
               np.array([energy, fluxes]).T, 
               header="Energy units: Rydbergs, Flux units: erg/s/Hz",)

def extract_cloudy_results(dir, filename, input_wav, input_flux):
    """ Loads individual cloudy results from the output files and converts the
    units to L_sol/A for continuum, L_sol for lines. """

    lines = np.loadtxt(f"{dir}/{filename}.lines", usecols=(1), delimiter="\t", skiprows=2)
    cont_wav, cont_incident, cont_flux, cont_lineflux = np.loadtxt(f"{dir}/{filename}.cont", usecols=(0, 1, 3, 8)).T
    cont_wav = np.flip(cont_wav)
    cont_flux = np.flip(cont_flux)
    cont_lineflux = np.flip(cont_lineflux)

    # Convert cloudy fluxes from erg/s to Lsun
    lines /= 3.826e33
    cont_flux /= 3.826e33
    cont_lineflux /= 3.826e33

    # subtract lines from nebular continuum model
    cont_flux -= cont_lineflux

    # continuum from Lsun to Lsun/A.
    cont_flux /= cont_wav

    cont_flux[cont_wav < 911.8] = 0

    # # Total ionizing flux in the input model
    ionizing_spec = input_flux[input_wav <= 911.8]
    ionizing_wavs = input_flux[input_wav <= 911.8]
    input_ionizing_flux = np.trapezoid(ionizing_spec, x=ionizing_wavs)

    # # Total ionizing flux in the cloudy outputs
    cloudy_ionizing_flux = np.sum(lines) + np.trapezoid(cont_flux, x=cont_wav)

    # # Normalise cloudy fluxes to the level of the input model
    lines *= input_ionizing_flux/cloudy_ionizing_flux
    cont_flux *= input_ionizing_flux/cloudy_ionizing_flux

    # Resample the nebular continuum onto wavelengths of stellar models
    cont_flux = np.interp(input_wav, cont_wav, cont_flux)
    return cont_flux, lines




default_cloudy_params = {
        "no_grain_scaling": False,
        "ionisation_parameter": None, # ionisation parameter
        "radius": None, # radius in log10 parsecs, only important for spherical geometry
        "covering_factor": None, # covering factor. Keep as 1 as it is more efficient to simply combine SEDs to get != 1.0 values
        "stop_T": None, # K, if not provided the command is not used
        "stop_efrac": None, # if not provided the command is not used
        "stop_column_density": None, # log10(N_H/cm^2), if not provided the command is not used
        "T_floor": None, # K, if not provided the command is not used
        "hydrogen_density": None, # Hydrogen density
        "z": 0.0, # redshift, only necessary if CMB heating included
        "CMB": None, # include CMB heating
        "cosmic_rays": None, # include cosmic rays
        "metals": True, # include metals
        "grains": None, # include dust grains
        "geometry": None, # the geometry

        "constant_density": None, # constant density flag
        "constant_pressure": None, # constant pressure flag # need one of these two

        "resolution": 1.0, # relative resolution the saved continuum spectra
        "output_abundances": None, # output abundances
        "output_cont": None, # output continuum
        "output_lines": None, # output full list of all available lines
        "output_linelist": None, # output linelist
    }

# to run: python -m brisket.grids.cloudy <params>.toml
import argparse, h5py, tqdm, toml
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        description='Run cloudy on a given input grid.')
    parser.add_argument('param', help='Path to TOML parameter file', type=str)
    args = parser.parse_args()
    params = toml.load(args.param)
    params = default_cloudy_params | params

    logger = setup_logger(__name__, level='DEBUG')
    if rank==0: logger.info(f"Running cloudy with parameters from: [bold]{args.param}[/bold]")

    if rank==0: logger.info(f'Running on input grid: [bold]{params["grid"]}[/bold]')
    grid = Grid(params['grid'])
    wavs = grid.wavelengths
    if rank==0: logger.info(f'Detected input grid axes: {list(grid.axes)}')


    if 'age' in grid.axes: # TODO FIX NAMING OF AGES IN GRID FILES
        if rank==0: logger.info(f'Shrinking grid to only include ages less than age_lim = {params["age_lim"]} Myr')
        age_axis = grid.array_axes.index('age')
        indices = np.where(grid.age <= params['age_lim']*1e6)[0]
        grid.data = np.take(grid.data, indices, axis=age_axis)
        grid.age = grid.age[grid.age <= params['age_lim']*1e6]
        if rank==0: logger.info(f'Reduced grid shape: {grid.shape}')
    else:
        if rank==0: logger.info(f'Detected input grid shape: {grid.shape}')

    # copy over the line list into a file cloudy can read
    # these are the lines we want to track in the cloudy models
    if rank==0: logger.info(f"Exporting cloudy line list to {cloudy_data_path}/brisket_cloudy_lines.txt")
    with open(f"{cloudy_data_path}/brisket_cloudy_lines.txt", 'w') as f:
        f.writelines([l + '\n' for l in linelist.cloudy_labels])
    
    # create a temporary directory to store the CLOUDY input/output files
    if rank==0: logger.info(f"Creating cloudy_temp/brisket_{grid.name}/ directory")
    os.chdir(config.grid_dir)
    base_dir = f'./cloudy_temp/brisket_{grid.name}/'
    if not os.path.exists(base_dir): 
        os.makedirs(base_dir)

    if rank==0: logger.info("[bold]Detecting CLOUDY parameter input axes")
    # handle the input parameters
    # things that can be free: logU, lognH, xid, CO
    default_vals = {'logU':-2, 'lognH':2, 'xid':0.36, 'CO':1.0}
    cloudy_axes = []
    cloudy_axes_vals = []
    for k,v in default_vals.items():
        if k in params:
            # then we have specified this parameter, whether free or fixed
            fixed = False
            if 'fixed' in params[k]:
                fixed = params[k]['fixed']
            elif 'isfixed' in params[k]:
                fixed = params[k]['isfixed']
            elif 'free' in params[k]:
                fixed = not params[k]['free']
            elif 'isfree' in params[k]:
                fixed = not params[k]['isfree']
            else:
                raise ValueError(f"Parameter {k} must have a key specifying whether it is free or fixed: 'fixed', 'free', 'isfree', isfixed'")

            if fixed:
                if rank==0: logger.info(f"Parameter {k} fixed at {params[k]['value']}")
                params[k] = params[k]['value']
            else:
                if rank==0: logger.info(f"Parameter {k} free from {params[k]['low']} to {params[k]['high']} in steps of {params[k]['step']}")
                cloudy_axes.append(k)
                cloudy_axes_vals.append(np.arange(params[k]['low'], params[k]['high'], params[k]['step']))
        else:
            logger.warning(f'Parameter {k} not specified, fixing at default value {v}')
            params[k] = v

    cloudy_axes_shape = [len(vals) for vals in cloudy_axes_vals]

    final_grid_axes = list([str(s) for s in grid.axes]) + cloudy_axes
    final_grid_shape = grid.shape + tuple(cloudy_axes_shape)
    n_runs = np.prod(final_grid_shape)

    if rank==0: logger.info(f"Final grid will have axes: {final_grid_axes}")
    if rank==0: logger.info(f"Final grid will have shape: {final_grid_shape}")
    if rank==0: logger.info(f"Preparing to run cloudy {n_runs} times!")

    if size > 1 and not params['MPI']:
        if rank == 0: logger.error("MPI is not enabled, but more than one core is available. Please enable MPI in the parameter file.")
        quit()
    elif size > 1:
        # Assign models to cores
        if rank==0: logger.info(f"Splitting up into {size} MPI threads")
        threads = mpi_split_array(np.arange(n_runs))
        indices = np.array(list(np.ndindex(final_grid_shape)))[threads]
        indices = [tuple(i) for i in indices]
    else:
        threads = np.arange(n_runs)
        indices = list(np.ndindex(final_grid_shape))
    n_threads = len(threads)

    n_current = 1
    for index in indices:
        filename = "brisket_" + "_".join([k+str(i) for k,i in zip(final_grid_axes, index)])
        logger.info(f"[bold green]Thread {rank}[/bold green]: running model {n_current}/{n_threads} ({filename})")

        if len(cloudy_axes) > 0:
            input_grid_index = index[:-len(cloudy_axes)]
            cloudy_grid_index = index[-len(cloudy_axes):] 
            cloudy_params = {cloudy_axes[i] : cloudy_axes_vals[i][cloudy_grid_index[i]] for i in range(len(cloudy_axes))}
            params = params | cloudy_params
        else:
            input_grid_index = index



        if 'zmet' in grid.axes: 
            zmet_index = grid.axes.index('zmet')
            zmet = grid.zmet[input_grid_index[zmet_index]]
            params['zmet'] = zmet
        else:
            raise Exception('grid needs to have metallicity (for now)')

        # export the SED to a file that cloudy can read
        make_cloudy_sed_file(cloudy_sed_dir, filename, wavs, grid.data[input_grid_index])

        # create the cloudy input file, using the params
        make_cloudy_input_file(base_dir, filename=filename, params=params)

        os.chdir(base_dir)

        # remove any existing cloudy output files
        if os.path.exists(f"{filename}.cont"):
            os.remove(f"{filename}.cont")
        if os.path.exists(f"{filename}.lines"):
            os.remove(f"{filename}.lines")
        if os.path.exists(f"{filename}.out"):
            os.remove(f"{filename}.out")

        # run cloudy!
        cloudy_exe = os.environ["CLOUDY_EXE"]
        os.system(f'{cloudy_exe} -r {filename}')
        
        # remove the SED file, since we don't need it anymore (and we don't want to clog the filesystem)
        os.remove(f"{filename}.in")
        os.remove(f"{filename}.out")
        os.remove(f"{cloudy_sed_dir}/{filename}.sed")
        os.chdir(config.grid_dir)
        n_current += 1

    
    if params['MPI']:
        # Combine arrays of models assigned to cores, checks all is finished
        # if rank==0: logger.info(f"Combining cloudy runs from {size} MPI threads
        threads = mpi_combine_array(threads, n_runs)

    if rank == 0:
        final_grid_cont_data = np.zeros(final_grid_shape + (len(wavs),))
        final_grid_line_data = np.zeros(final_grid_shape + (len(linelist),))

        for index in (pbar := tqdm.tqdm(np.ndindex(final_grid_shape), total=n_runs)):
            filename = "brisket_" + "_".join([k+str(i) for k,i in zip(final_grid_axes, index)])
            pbar.set_description(filename.removeprefix('brisket_'))

            if len(cloudy_axes) > 0:
                input_grid_index = index[:-len(cloudy_axes)]
            else:
                input_grid_index = index

            cont, lines = extract_cloudy_results(base_dir, filename, wavs, grid[input_grid_index])

            final_grid_cont_data[index] = cont
            final_grid_line_data[index] = lines


        outfilepath = os.path.join(config.grid_dir, params['grid']+'_cloudy_lines.hdf5')
        with h5py.File(outfilepath, 'w') as hf:
            hf.create_dataset('wavs', data=linelist.wavs)
            hf.create_dataset('names', data=list(linelist.names), dtype=h5py.string_dtype())
            hf.create_dataset('labels', data=list(linelist.labels),  dtype=h5py.string_dtype())
            hf.create_dataset('cloudy_labels', data=list(linelist.cloudy_labels),  dtype=h5py.string_dtype())
            
            hf.create_dataset('axes', data=final_grid_axes + ['wavs'])
            input_params = {axis : getattr(grid, axis) for axis in grid.axes}
            for axis in final_grid_axes:
                if axis in cloudy_axes:
                    hf.create_dataset(axis, data=cloudy_axes_vals[cloudy_axes.index(axis)])
                else:
                    hf.create_dataset(axis, data=getattr(grid, axis))

            hf.create_dataset('grid', data=final_grid_line_data)

        outfilepath = os.path.join(config.grid_dir, params['grid']+'_cloudy_cont.hdf5')
        with h5py.File(outfilepath, 'w') as hf:
            hf.create_dataset('wavs', data=wavs)
            
            hf.create_dataset('axes', data=final_grid_axes + ['wavs'])
            input_params = {axis : getattr(grid, axis) for axis in grid.axes}
            for axis in final_grid_axes:
                if axis in cloudy_axes:
                    hf.create_dataset(axis, data=cloudy_axes_vals[cloudy_axes.index(axis)])
                else:
                    hf.create_dataset(axis, data=getattr(grid, axis))

            hf.create_dataset('grid', data=final_grid_cont_data)

        for index in (pbar := tqdm.tqdm(np.ndindex(final_grid_shape), total=n_runs)):
            filename = "brisket_" + "_".join([k+str(i) for k,i in zip(final_grid_axes, index)])
            os.remove(f"{base_dir}/{filename}.lines")
            os.remove(f"{base_dir}/{filename}.cont")

        
        # for axis in axes:
        #     if axis != 'wavs':
        #         print(f[axis][:])




        # print(f['grid'].keys())

