import torch
import math


def frequency_adapter(teacher_mag, teacher_spec, eps=1e-8):
    B, F, T = teacher_mag.shape

    # Step 1: Fo = cumulative max across frequency axis
    Fo = teacher_mag.abs().cummax(dim=1)[0]  # (B, F, T)

    # Step 2: First-order derivative ∇Fo
    dFo = (Fo[:, 1:, :] - Fo[:, :-1, :]) / (Fo[:, :-1, :] + eps)  # (B, F-1 T)


    # Step 3: Find m_t = argmax_f ∇Fo for each frame
    m = dFo.argmax(dim=1)  # (B, T), frequency index per frame


    # Step 4: Construct binary masks for high / low bands
    freq_idx = torch.arange(F, device=teacher_mag.device).view(1, F, 1)  # (1, F, 1)
    m_expanded = m.unsqueeze(1)  # (B, 1, T)


    # Low band: freq >= m
    low_mask = freq_idx <= m_expanded  # Tl: 0 to m (inclusive)
    high_mask = freq_idx >= m_expanded  # Th: m to 256 (inclusive)


    T_l = teacher_mag * low_mask  # Low-frequency part
    T_h = teacher_mag * high_mask  # High-frequency part
    T_spec_l = teacher_spec * low_mask  # Low-frequency part
    T_spec_h = teacher_spec * high_mask  # High-frequency part


    return T_l, T_h, T_spec_l, T_spec_h, low_mask, high_mask


def cosine_loss_complex(T, S, eps=1e-8):
    # Step 1: complex inner product (real part of dot product over freq)
    dot = torch.sum(T.real * S.real + T.imag * S.imag, dim=2)  # shape (B, T)


    # Step 2: norms
    T_norm = torch.sqrt(torch.sum(T.real ** 2 + T.imag ** 2, dim=2) + eps)  # (B, T)
    S_norm = torch.sqrt(torch.sum(S.real ** 2 + S.imag ** 2, dim=2) + eps)  # (B, T)


    # Step 3: cosine similarity
    cos_sim = dot / (T_norm * S_norm + eps)  # shape (B, T)
    return (1 - cos_sim)  # Typically used as: (cos_sim - 1).mean() for loss


def l2_loss_subband(a, b):
    l2_subband = ((a - b) ** 2).mean(dim=2)
    return l2_subband  # Mean squared error


def low_loss_Rho1(Y_spec_l, T_spec_l, S_spec_l, low_mask, k, low_sb):
    # Low frequency: cosine loss only

    B, F, T = Y_spec_l.shape

    # make patches
    sb = low_sb
    nb = math.ceil(F / sb)
    pad_f = nb * sb - F

    if pad_f > 0:
        pad_mask = torch.zeros(B, pad_f, T, dtype=torch.bool, device=low_mask.device)
        low_mask = torch.cat([low_mask, pad_mask], dim=1)

    if pad_f > 0:
        pad = torch.zeros(B, pad_f, T, device=Y_spec_l.device)
        Y_spec_l = torch.cat([Y_spec_l, pad], dim=1)
        S_spec_l = torch.cat([S_spec_l, pad], dim=1)
        T_spec_l = torch.cat([T_spec_l, pad], dim=1)

    # -> [B, nb, sb, T]
    Y_spec_l = Y_spec_l.view(B, nb, sb, T)
    S_spec_l = S_spec_l.view(B, nb, sb, T)
    T_spec_l = T_spec_l.view(B, nb, sb, T)

    low_mask_patch = low_mask.view(B, nb, sb, T)
    result_mask = low_mask_patch.any(dim=2)  # shape: (16, 26, 501) # If any sb value is True, calculate Loss for that location

    stud_err = cosine_loss_complex(Y_spec_l, S_spec_l)
    teach_err = cosine_loss_complex(Y_spec_l, T_spec_l)

    delta = stud_err - teach_err

    # 7) Top k% token mask

    # 1) Calculate flattened mask count (per batch)
    mask_counts = result_mask.view(B, -1).sum(dim=1)  # shape: (B,)

    # 2) Calculate selection count m_b per batch
    #    Clamp to select at least 1
    m = (mask_counts.float() * k).long().clamp(min=1)  # shape: (B,)

    # 3) Create final mask
    final_mask = torch.zeros_like(result_mask, dtype=torch.bool)  # shape: (B, nb, T)

    for b in range(B):
        # From the current batch, extract deltas where mask is True to create a 1D tensor
        masked_delta = delta[b][result_mask[b]]  # shape: (mask_counts[b],)
        if masked_delta.numel() == 0:
            continue  # Skip if there are no masked elements

        # Get top-m[b] values and use the minimum as thr_b
        topk_vals, _ = masked_delta.topk(m[b].item(), largest=True)
        thr_b = topk_vals.min()

        # True where delta[b] >= thr_b and at result_mask[b] positions
        # i.e., select k%, but only calculate for the low_frequency
        final_mask[b] = (delta[b] >= thr_b) & result_mask[b]

    # Convert to float if necessary
    mask = final_mask.float()

    L_low = cosine_loss_complex(T_spec_l, S_spec_l) * mask

    return L_low.sum() / mask.sum()  # Normalize by the number of selected tokens


def high_loss_Rho1(Y_h, Y_spec_h, T_h, T_spec_h, S_h, S_spec_h, beta, high_mask, k, high_sb):
    B, F, T = Y_h.shape

    # Tokenize into sub-bands
    sb = high_sb
    nb = math.ceil(F / sb)
    pad_f = nb * sb - F

    if pad_f > 0:
        pad_mask = torch.zeros(B, pad_f, T, dtype=torch.bool, device=high_mask.device)
        high_mask = torch.cat([high_mask, pad_mask], dim=1)

    if pad_f > 0:
        pad = torch.zeros(B, pad_f, T, device=Y_h.device)
        Y_h = torch.cat([Y_h, pad], dim=1)
        S_h = torch.cat([S_h, pad], dim=1)
        T_h = torch.cat([T_h, pad], dim=1)
        Y_spec_h = torch.cat([Y_spec_h, pad], dim=1)
        S_spec_h = torch.cat([S_spec_h, pad], dim=1)
        T_spec_h = torch.cat([T_spec_h, pad], dim=1)

    # -> [B, nb, sb, T]
    Y_h = Y_h.view(B, nb, sb, T)
    S_h = S_h.view(B, nb, sb, T)
    T_h = T_h.view(B, nb, sb, T)
    Y_spec_h = Y_spec_h.view(B, nb, sb, T)
    S_spec_h = S_spec_h.view(B, nb, sb, T)
    T_spec_h = T_spec_h.view(B, nb, sb, T)

    high_mask_patch = high_mask.view(B, nb, sb, T)
    result_mask = high_mask_patch.any(dim=2)  # shape: (16, 26, 501) # If any sb value is True, calculate Loss for that location

    stud_err = beta * cosine_loss_complex(Y_spec_h, S_spec_h) + (1 - beta) * l2_loss_subband(Y_h, S_h)
    teach_err = beta * cosine_loss_complex(Y_spec_h, T_spec_h) + (1 - beta) * l2_loss_subband(Y_h, T_h)

    delta = stud_err - teach_err

    # 1) Calculate flattened mask count (per batch)
    mask_counts = result_mask.view(B, -1).sum(dim=1)  # shape: (B,)

    # 2) Calculate selection count m_b per batch
    #    Clamp to select at least 1
    m = (mask_counts.float() * k).long().clamp(min=1)  # shape: (B,)

    # 3) Create final mask
    final_mask = torch.zeros_like(result_mask, dtype=torch.bool)  # shape: (B, nb, T)

    for b in range(B):
        # From the current batch, extract deltas where mask is True to create a 1D tensor
        masked_delta = delta[b][result_mask[b]]  # shape: (mask_counts[b],)
        if masked_delta.numel() == 0:
            continue  # Skip if there are no masked elements

        # Get top-m[b] values and use the minimum as thr_b
        topk_vals, _ = masked_delta.topk(m[b].item(), largest=True)
        thr_b = topk_vals.min()

        # True where delta[b] >= thr_b and at result_mask[b] positions
        # i.e., select k%, but only calculate for the low_frequency
        final_mask[b] = (delta[b] >= thr_b) & result_mask[b]

    # Convert to float if necessary
    mask = final_mask.float()

    L_high_cos = cosine_loss_complex(T_spec_h, S_spec_h) * mask
    L_high_l2 = l2_loss_subband(T_h, S_h) * mask

    L_high = beta * L_high_cos + (1 - beta) * L_high_l2
    return L_high.sum() / mask.sum()  # Normalize by the number of selected tokens


def dispatch_DFKD(y, student_ests, teacher_ests, beta, k, high_sb, low_sb, n_fft=512, hop_length=128,
                   win_length=512):
    window = torch.hann_window(win_length).to(student_ests.device)

    # ests: (B, C, T)
    # flatten batch and channel to apply STFT
    # B, C, T = student_ests.shape
    # student_ests = student_ests.view(B * C, T)
    # teacher_ests = teacher_ests.view(B * C, T)

    student_spec = torch.stft(student_ests, n_fft=n_fft, hop_length=hop_length,
                              win_length=win_length, window=window, return_complex=True)
    teacher_spec = torch.stft(teacher_ests, n_fft=n_fft, hop_length=hop_length,
                              win_length=win_length, window=window, return_complex=True)
    y_spec = torch.stft(y, n_fft=n_fft, hop_length=hop_length,
                        win_length=win_length, window=window, return_complex=True)

    # MSE on magnitude or complex
    # Option 1: Use magnitude (realistic)
    student_mag = student_spec.abs()
    teacher_mag = teacher_spec.abs()
    y_mag = y_spec.abs()  # Not used in this loss function
    # print("start")

    T_l, T_h, T_spec_l, T_spec_h, low_mask, high_mask = frequency_adapter(teacher_mag, teacher_spec)

    _, Freq, _ = student_spec.shape

    S_h = student_mag * high_mask  # High-frequency part
    S_spec_h = student_spec * high_mask  # High-frequency part
    S_spec_l = student_spec * low_mask  # Low-frequency part

    Y_h = y_mag * high_mask  # High-frequency part
    Y_spec_h = y_spec * high_mask  # High-frequency part
    Y_spec_l = y_spec * low_mask  # Low-frequency part

    h_loss = high_loss_Rho1(Y_h, Y_spec_h, T_h, T_spec_h, S_h, S_spec_h, beta, high_mask, k, high_sb)
    l_loss = low_loss_Rho1(Y_spec_l, T_spec_l, S_spec_l, low_mask, k, low_sb)

    kd_loss = h_loss + l_loss

    return kd_loss, h_loss, l_loss


if __name__ == '__main__':
    # Create dummy tensors for demonstration
    B, L = 2, 16000 * 4
    y = torch.randn(B, L)
    student_ests = torch.randn(B, L)
    teacher_ests = torch.randn(B, L)

    # Hyperparameters
    beta = 0.5
    k = 0.8

    high_sb, low_sb = 40, 10 # w/ MSSP
    # high_sb, low_sb = 20, 20 # w/o MSSP



    kd_loss, h_loss, l_loss = dispatch_DFKD(
        y, student_ests, teacher_ests, beta, k, high_sb, low_sb
    )

    print("\n--- Output ---")
    print(f"Total KD Loss: {kd_loss.item():.4f}")
    print(f"High-band Loss: {h_loss.item():.4f}")
    print(f"Low-band Loss: {l_loss.item():.4f}")
