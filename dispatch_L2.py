import torch
import math 

def dispatch_L2(y, student_ests, teacher_ests, sb, k):

    # student_ests, teacher_ests, y : [B, L]

    n_fft = 512
    hop_length = 128
    win_length = 512
    window = torch.hann_window(win_length).to(student_ests.device)

    # STFT
    s_spec = torch.stft(student_ests, n_fft=n_fft, hop_length=hop_length,
                            win_length=win_length, window=window, return_complex=True)
    t_spec = torch.stft(teacher_ests, n_fft=n_fft, hop_length=hop_length,
                            win_length=win_length, window=window, return_complex=True)
    y_spec = torch.stft(y, n_fft=n_fft, hop_length=hop_length,
                            win_length=win_length, window=window, return_complex=True)

    s_mag, t_mag, y_mag = s_spec.abs(), t_spec.abs(), y_spec.abs()

    B,F,T = y_mag.shape

    # padding
    # nb : number of sub-bands per patch, sb : size of sub-bands per patch

    nb = math.ceil(F / sb)
    pad_f = nb * sb - F
    if pad_f > 0:
        pad = torch.zeros(B, pad_f, T, device=y_mag.device)
        y_mag = torch.cat([y_mag, pad], dim=1)
        s_mag = torch.cat([s_mag, pad], dim=1)
        t_mag = torch.cat([t_mag, pad], dim=1)


    # [B, nb, sb, T]
    y_mag = y_mag.view(B, nb, sb, T)
    s_mag = s_mag.view(B, nb, sb, T)
    t_mag = t_mag.view(B, nb, sb, T)

    # [B, nb, T]
    stud_err  = ((s_mag - y_mag)**2).mean(dim=2)
    teach_err = ((t_mag - y_mag)**2).mean(dim=2) 

    # DISPATCH
    delta = stud_err - teach_err 

    total = nb * T
    m = max(int(total*k), 1)
    flat = delta.view(B, -1)
    thr, _ = torch.kthvalue(-flat, m, dim=1)
    mask = (flat >= -thr.unsqueeze(1)).view(B, nb, T).float()

    kd_per_tok = ((s_mag - t_mag)**2).mean(dim=2)   # [B, nb, T]

    kd_loss = (kd_per_tok * mask).sum() / (B*nb*T*k)

    return kd_loss
