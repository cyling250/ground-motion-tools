# -*- coding:utf-8 -*-
# @Time:    2025/3/17 17:15
# @Author:  RichardoGu
"""
Intensity measures
"""
import numpy as np
from process import gm_data_fill
from sbs_integration_linear import segmented_parsing
from spectrum import SPECTRUM_PERIOD, get_spectrum
from enums import GMIMEnum, GMDataEnum

IM_ADJUST_DICT = {
    GMIMEnum.PGA.name: {  # 按照PGA进行调幅的其余IM变化率
        GMIMEnum.PGA.name: lambda x: x,
        GMIMEnum.PGV.name: lambda x: x,
        GMIMEnum.PGD.name: lambda x: x,

        GMIMEnum.RMSA.name: lambda x: x,
        GMIMEnum.RMSV.name: lambda x: x,
        GMIMEnum.RMSD.name: lambda x: x,

        GMIMEnum.I_SUFFIX_A.name: lambda x: x,
        GMIMEnum.I_SUFFIX_C.name: lambda x: x ** (3 / 2),

        GMIMEnum.SED.name: lambda x: x,
        GMIMEnum.CAV.name: lambda x: x,

        GMIMEnum.ASI.name: lambda x: x,
        GMIMEnum.VSI.name: lambda x: x,
        GMIMEnum.HI.name: lambda x: x,

        GMIMEnum.SMA.name: lambda x: x,
        GMIMEnum.SMV.name: lambda x: x,

        GMIMEnum.I_A.name: lambda x: x,
        GMIMEnum.I_D.name: lambda x: x,
        GMIMEnum.I_V.name: lambda x: x ** (2 / 3),
        GMIMEnum.I_F.name: lambda x: x,

        GMIMEnum.SA_T1.name: lambda x: x,
        GMIMEnum.SV_T1.name: lambda x: x,
        GMIMEnum.SD_T1.name: lambda x: x
    }
}


class GMIntensityMeasures:
    def __init__(self, gm_acc_data: np.ndarray, time_step: float):
        self.acc, self.vel, self.disp = gm_data_fill(gm_acc_data, time_step, GMDataEnum.ACC)
        if self.acc.ndim == 1:
            self.acc = np.expand_dims(self.acc, axis=0)
            self.vel = np.expand_dims(self.vel, axis=0)
            self.disp = np.expand_dims(self.disp, axis=0)

        self.time_step = time_step
        self.batch_size = self.acc.shape[0]
        self.seq_len = self.acc.shape[1]
        self.duration = self.seq_len * self.time_step

        self.spectrum_acc = None
        self.spectrum_vel = None
        self.spectrum_disp = None

        self.intensity_measures = {}

    def _get_spectrum(self):
        self.spectrum_acc, self.spectrum_vel, self.spectrum_disp, _, _ = get_spectrum(
            self.acc, self.time_step, 0.05
        )
        if self.spectrum_acc.ndim == 1:
            self.spectrum_acc = np.expand_dims(self.spectrum_acc, axis=0)
            self.spectrum_vel = np.expand_dims(self.spectrum_vel, axis=0)
            self.spectrum_disp = np.expand_dims(self.spectrum_disp, axis=0)

    def get_im(
            self,
            im_list: [list, GMIMEnum],
            period: float = 1
    ) -> dict[str, np.ndarray[float]]:
        """
        An external query must call this interface to get intensity measures.

        The parameter ``im`` is the intensity measures' name, and this func will output the corresponding result.

        All the input is saved by a dict named ``self.intensity_measures``, and each im will just be calculated once
        when it is first queried.

        The input im can be both upper and lower, but must be included in :class:`GroundMotionDataIntensityMeasures`.
        Args:
            im_list: Intensity measures' name.
            period: Some intensity measures need param ``period`` to calculate. Default 0.9s

        Returns:
            A dict ``{str,np.ndarray[float]}`` of ``{im_name, im_value}``

        """
        if im_list is None or len(im_list) == 0:
            raise ValueError("Parameter 'im_list' can not be None or empty.")

        result = {}
        if type(im_list) is GMIMEnum:
            im_list = [im_list]
        for im in im_list:
            im_upper = im.name.upper()
            im_lower = im.name.lower()
            try:
                result[im_upper] = self.intensity_measures[im_upper]
            except KeyError:
                self.intensity_measures[im_upper] = eval("self.im_" + im_lower)(period=period)
                result[im_upper] = self.intensity_measures[im_upper]
        return result

    def im_pga(self, **kwargs):
        """
        PGA
        .. math:: |max(a(t))|
        """
        return np.abs(self.acc).max(1)

    def im_pgv(self, **kwargs):
        """
        PGV
        .. math:: |max(v(t))|
        """
        return np.abs(self.vel).max(1)

    def im_pgd(self, **kwargs):
        """
        PGD
        .. math:: |max(d(t))|
        """
        return np.abs(self.disp).max(1)

    def im_rmsa(self, **kwargs):
        """
        Arms
        .. math:: \\sqrt{\\frac{1}{t_{tot}} \\int_{0}^{tot}a(t)^2dt}
        """
        return ((self.acc ** 2).sum(1) * self.time_step / self.duration) ** 0.5

    def im_rmsv(self, **kwargs):
        """
        Vrms
        .. math:: \\sqrt{\\frac{1}{t_{tot}} \\int_{0}^{tot}v(t)^2dt}
        """
        return ((self.vel ** 2).sum(1) * self.time_step / self.duration) ** 0.5

    def im_rmsd(self, **kwargs):
        """
        Drms
        .. math:: \\sqrt{\\frac{1}{t_{tot}} \\int_{0}^{tot}d(t)^2dt}
        """
        return ((self.disp ** 2).sum(1) * self.time_step / self.duration) ** 0.5

    def im_i_suffix_a(self, **kwargs):
        """
        IA
        .. math:: \\frac{\\pi}{2g}\\int^{t_{tot}}_{0}{a(t)^2dt}
        """
        return (self.acc ** 2).sum(1) * self.time_step * np.pi / (2 * 9.8)

    def im_i_suffix_c(self, **kwargs):
        """
        IC
        .. math:: (Arms)^{3/2}\\sqrt{t_{tot}}
        """
        return self.get_im(GMIMEnum.RMSA)[GMIMEnum.RMSA.name.upper()] ** 1.5 * (self.duration ** 0.5)

    def im_sed(self, **kwargs):
        """
        SED
        .. math:: \\int_{0}^{tot}{v(t)^2}dt
        """
        return (self.vel ** 2).sum(1) * self.time_step

    def im_cav(self, **kwargs):
        """
        CAV
        .. math:: \\int_{0}^{tot}{|a(t)|}dt
        """
        return np.abs(self.acc).sum(1) * self.time_step

    def im_sa_t1(self, **kwargs):
        """
        Sa(T1)
        Spectrum acceleration at the first natural period of vibration.
        """
        acc, vel, disp = segmented_parsing(mass=1,
                                           stiffness=((2 * np.pi) / kwargs["period"]) ** 2,
                                           damping_ratio=0.05,
                                           load=self.acc, time_step=self.time_step)
        self.intensity_measures[GMIMEnum.SV_T1.name.upper()] = np.abs(vel).max(1)
        self.intensity_measures[GMIMEnum.SD_T1.name.upper()] = np.abs(disp).max(1)
        return np.abs(acc).max(1)

    def im_sv_t1(self, **kwargs):
        """
        Sv(T1)
        Spectrum velocity at the first natural period of vibration.
        """
        acc, vel, disp = segmented_parsing(mass=1,
                                           stiffness=((2 * np.pi) / kwargs["period"]) ** 2,
                                           damping_ratio=0.05,
                                           load=self.acc, time_step=self.time_step)
        self.intensity_measures[GMIMEnum.SA_T1.name.upper()] = np.abs(acc).max(1)
        self.intensity_measures[GMIMEnum.SD_T1.name.upper()] = np.abs(disp).max(1)
        return np.abs(vel).max(1)

    def im_sd_t1(self, **kwargs):
        """
        Sd(T1)
        Spectrum displacement at the first natural period of vibration.
        """
        acc, vel, disp = segmented_parsing(mass=1,
                                           stiffness=((2 * np.pi) / kwargs["period"]) ** 2,
                                           damping_ratio=0.05,
                                           load=self.acc, time_step=self.time_step)
        self.intensity_measures[GMIMEnum.SA_T1.name.upper()] = np.abs(acc).max(1)
        self.intensity_measures[GMIMEnum.SV_T1.name.upper()] = np.abs(vel).max(1)
        return np.abs(disp).max(1)

    def im_asi(self, **kwargs):
        """
        ASI
        .. math:: \\int_{0.1}^{0.5}{Sa(\\xi = 0.05, t) dt}
        """
        if self.spectrum_acc is None:
            self._get_spectrum()

        result = np.zeros(self.batch_size)
        for i in range(len(SPECTRUM_PERIOD)):
            if SPECTRUM_PERIOD[i] < 0.1:
                continue
            if SPECTRUM_PERIOD[i] > 0.5:
                break
            result += self.spectrum_acc[:, i] * (SPECTRUM_PERIOD[i] - SPECTRUM_PERIOD[i - 1])
        return result

    def im_vsi(self, **kwargs):
        """
        VSI
        .. math:: \\int_{0.1}^{2.5}{Sv(\\xi = 0.05, t) dt}
        """
        if self.spectrum_vel is None:
            self._get_spectrum()
        result = np.zeros(self.batch_size)
        for i in range(len(SPECTRUM_PERIOD)):
            if SPECTRUM_PERIOD[i] < 0.1:
                continue
            if SPECTRUM_PERIOD[i] > 2.5:
                break
            result += self.spectrum_vel[:, i] * (SPECTRUM_PERIOD[i] - SPECTRUM_PERIOD[i - 1])
        return result

    def im_hi(self, **kwargs):
        """
        HI
        .. math:: \\int_{0.1}^{2.5}{PSv(\\xi = 0.05, t) dt}
        """
        if self.spectrum_disp is None:
            self._get_spectrum()
        result = np.zeros(self.batch_size)
        for i in range(len(SPECTRUM_PERIOD)):
            if SPECTRUM_PERIOD[i] < 0.1:
                continue
            if SPECTRUM_PERIOD[i] > 2.5:
                break
            result += self.spectrum_disp[:, i] * (SPECTRUM_PERIOD[i] - SPECTRUM_PERIOD[i - 1])
        return result

    def im_sma(self, **kwargs):
        """
        The third peek in acceleration time history.
        """
        return np.sort(np.abs(self.acc), axis=1)[:, -3]

    def im_smv(self, **kwargs):
        """
        The third peek in velocity time history.
        """
        return np.sort(np.abs(self.vel), axis=1)[:, -3]

    def im_i_a(self, **kwargs):
        """
        Ia
        .. math:: PGA{\\dot}t^{1/3}_{tot}
        """
        return self.get_im(GMIMEnum.PGA)[GMIMEnum.PGA.name.upper()] * self.duration ** (1 / 3)

    def im_i_d(self, **kwargs):
        """
        Id
        .. math:: PGD{\\dot}t^{1/3}_{tot}
        """
        return self.get_im(GMIMEnum.PGD)[GMIMEnum.PGD.name.upper()] * self.duration ** (1 / 3)

    def im_i_v(self, **kwargs):
        """
        Iv
        .. math:: PGV^{2/3}{\\dot}t^{1/3}_{tot}
        """
        return self.get_im(GMIMEnum.PGV)[GMIMEnum.PGV.name.upper()] ** (2 / 3) * self.duration ** (1 / 3)

    def im_i_f(self, **kwargs):
        """
        IF
        .. math:: PGV{\\dot}t^{1/4}_{tot}
        """
        return self.get_im(GMIMEnum.PGV)[GMIMEnum.PGV.name.upper()] * self.duration ** (1 / 4)

    def im_miv(self, **kwargs):
        """
        MIV
        """
        # TODO ADD
        return np.zeros(self.batch_size)

    def im_di(self, **kwargs):
        """
        DI
        """
        # TODO ADD
        return np.zeros(self.batch_size)

    def im_t70(self, **kwargs):
        """
        T0.75-T0.05
        """
        # TODO ADD
        return np.zeros(self.batch_size)

    def im_t90(self, **kwargs):
        """
        T0.95-T0.05
        """
        # TODO ADD
        return np.zeros(self.batch_size)

    @staticmethod
    def im_adjust(im_data: dict, base_im: GMIMEnum, target_value: float):
        """
        对IM指标进行调幅
        Args:
            im_data: 需要进行调幅的IM指标
            base_im: 以哪个IM为基准调幅
            target_value: 要调幅的值

        Returns:

        """
        # 初始化一个新数组，不改变原有的数据
        adjusted_im_data = {}

        # 首先将base_im调幅到target_value,并记录调幅参数
        base_im_data = im_data[base_im.name.upper()]
        base_adjust_value = target_value / base_im_data
        adjusted_im_data[base_im.name.upper()] = im_data[base_im.name.upper()] * base_adjust_value

        # 逐项计算
        for im_key in im_data.keys():
            if im_key not in IM_ADJUST_DICT[base_im.name].keys():
                raise KeyError(f"调幅系数中并未收录IM指标：{im_key}")
            adjust_func = IM_ADJUST_DICT[base_im.name][GMIMEnum[im_key].name]  # 获取该IM的调幅系数
            adjust_value = np.array([adjust_func(bav_i) for bav_i in base_adjust_value])  # 计算调幅矩阵
            adjusted_im_data[im_key] = im_data[im_key] * adjust_value  # 原始值与调幅矩阵逐项相乘

        return adjusted_im_data
