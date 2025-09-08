import sys
import os
import time
import torch
import numpy as np


sizes = [32, 64, 128, 256, 512, 1024, 2048]


def hmm_pytorch(X):
    """
    Pure PyTorch version of HMM runner.

    Args:
        X: Tensor of shape (T, B, N, N)
           - T: num steps
           - B: batch size
           - N: number of hidden states

    Returns:
        s_state: Tensor of shape (T, B, N)
    """
    T, B, N, _ = X.shape

    # Initialize s_state[0] as zeros (shape: [B, N])
    s_state = torch.zeros((T, B, N), dtype=X.dtype, device=X.device)

    for t in range(1, T):
        # prev_state shape: [B, N]
        prev_state = s_state[t-1]

        # scores shape: [B, N, N]
        scores = prev_state.unsqueeze(-1) + X[t-1]

        # M: max over k
        M, _ = torch.max(scores, dim=1)  # shape: [B, N]

        # M2: sum over k2
        scores_shifted = scores - M.unsqueeze(1)  # for numerical stability
        exp_scores = torch.exp(scores_shifted)    # shape: [B, N, N]
        M2 = torch.sum(exp_scores, dim=1)         # shape: [B, N]

        # C: log(M2) + M
        C = torch.log(M2 + 1e-8) + M              # add epsilon to avoid log(0)

        s_state[t] = C

    return s_state



#task = autotvm.task.create(hmm, args=('float32',), target='cuda', target_host="llvm")

def log_eye(K, dtype, device):
    x = torch.empty(K, K, dtype = dtype, device = device)
    x.fill_(float("-inf"))
    x.diagonal().fill_(0)
    return x

def log_eye_cat(x):
    K = x.shape[-1]
    batch = x.shape[1]
    return torch.cat([
        x,
        log_eye(K, x.dtype, x.device).view(1, 1, K, K).expand(1, batch, K, K),
    ], dim=0)

# from tvm.contrib.dlpack import to_pytorch_func

def get_fb(size):
    """
    with autotvm.apply_history_best(f'best_hmm_k{size}.log'):
        with tvm.target.create("cuda"):
            s_mult, arg_bufs = hmm_runner('float32', size)
            mod = tvm.build(s_mult, arg_bufs, target="cuda", target_host="llvm")
            hmm_pytorch = to_pytorch_func(mod)
    """

    # if the padding doesn't make a difference this must be an inclusive scan
    # x: batch x time x zt x zt-1
    #@profile
    def fb(x, mask=None):
        batch, time, size, _ = x.shape
        lex = log_eye(size, dtype=x.dtype, device=x.device)
        # need time x batch x zt-1, zt
        x = x.permute(1, 0, 3, 2)
        if mask is not None:
            mask = mask.t()
            x[~mask[1:]] = lex
            """
            x.masked_scatter_(
                ~mask[1:,:,None,None],
                # EXPAND HERE IS BAD?
                lex[None,None].expand(x.shape),
            )
            import pdb; pdb.set_trace()
            """

        """
        out_fb = torch.empty(time+1, batch * 2, size, device=x.device)
        out_fb.fill_(float("-inf"))
        hmm_pytorch(
            log_eye_cat(torch.cat([x, x.flip(0).transpose(-2, -1)], 1)),
            out_fb,
        )

        out_fb2 = torch.empty(time+1, batch * 2, size, device=x.device)
        out_fb2.fill_(float("-inf"))
        hmm_pytorch(
            log_eye_cat(x),
            out_fb2[:,:batch],
        )
        hmm_pytorch(
            log_eye_cat(x.flip(0).transpose(-2, -1)),
            out_fb2[:,batch:],
        )
        alphas = out_fb[:, :batch]
        betas = out_fb[:, batch:].flip(0)
        """

        out_fb = torch.empty(2, time+1, batch, size, device=x.device)
        out_fb.fill_(float("-inf"))
        inp = torch.empty(time+1, batch, size, size, device=x.device)
        inp[-1] = lex
        # forward
        inp[:-1] = x
        out_fb[0] = hmm_pytorch(inp)
        # backward
        inp[range(time-1, -1, -1)] = x.transpose(-2, -1)
        out_fb[1] = hmm_pytorch(inp)

        alphas = out_fb[0]
        betas = out_fb[1].flip(0) # pay the memory cost here
        # not sure if i can flip the argument to hmm_pytorch

        log_marginals = x
        log_marginals += alphas[:-1].view(time, batch, size, 1)
        log_marginals += betas[1:].view(time, batch, 1, size)
        log_marginals -= alphas[-1].logsumexp(-1).view(1, -1, 1, 1)
        if mask is not None:
            log_marginals.masked_fill_(~mask[1:,:,None,None], float("-inf"))
        log_marginals = log_marginals.permute(1, 0, 3, 2)
        return log_marginals, alphas
        #marginals = log_marginals.exp()
        # switch back marginals: batch x time x zt x zt-1
        #return marginals, alphas, betas, log_marginals

    return fb

# if __name__ == "__main__":
#     from tvm import autotvm
#     import logging
#     logging.getLogger('autotvm').setLevel(logging.DEBUG)
#     logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

#     for size in sizes:
#         task = autotvm.task.create(hmm_runner, args=('float32', size),
#                                    target='cuda', target_host="llvm")

#         measure_option = autotvm.measure_option(
#             builder=autotvm.LocalBuilder(n_parallel=5),
#             runner=autotvm.LocalRunner(number=10, repeat=3, timeout=10, min_repeat_ms=50))


#         tuner = autotvm.tuner.RandomTuner(task)
#         tuner.tune(n_trial=100,
#                    measure_option=measure_option,
#                    callbacks=[autotvm.callback.log_to_file(f'hmm_k{size}.log')])

#         autotvm.record.pick_best(f"hmm_k{size}.log", f"best_hmm_k{size}.log")
