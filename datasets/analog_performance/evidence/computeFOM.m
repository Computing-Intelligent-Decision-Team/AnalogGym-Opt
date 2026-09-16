% function FOM_AW = computeFOM(gbw, PSRR, CMRR, gain, ivdd_27, tc, noise, vos)
%     % 计算放大器的性能表征值 FOM_AW
%     %
%     % 输入:
%     % - PSRR, CMRR, AOL, PM, SR, IQ, TS, EN, VOS: 放大器的性能指标
%     %
%     % 输出:
%     % - FOM_AW: 放大器的性能表征值
%
%     % 典型值
%     PSRR_norm = 100;
%     CMRR_norm = 100;
%     gain_norm = 100;
%     ivdd_27_norm = 1e-6;
%     tc_norm = 1e-6;
%     noise_norm = 1e-5;
%     vos_norm = 1e-6;
%     gbw_norm = 1e8;
%     % 计算 FOM_AW
%     FOM_AW = ((PSRR/PSRR_norm) * (CMRR/CMRR_norm) * (gain/gain_norm))^(-2) ...
%              * (ivdd_27/ivdd_27_norm) * (tc/tc_norm) * (noise/noise_norm) * (vos/vos_norm) * (gbw / gbw_norm)^(-0.5);
%
% end


function FOM_AW = computeFOM(gbw, PSRR, CMRR, gain, ivdd_27, tc, noise, vos)
    % 计算放大器的性能表征值 FOM_AW
    %
    % 输入:
    % - PSRR, CMRR, AOL, PM, SR, IQ, TS, EN, VOS: 放大器的性能指标
    %
    % 输出:
    % - FOM_AW: 放大器的性能表征值

    % 典型值
    % 计算 FOM_AW
    FOM_AW = gbw*1500e-12 / ivdd_27;

end
