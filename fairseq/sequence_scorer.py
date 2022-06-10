# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys

import torch
from fairseq import utils


class SequenceScorer(object):
    """Scores the target for a given source sentence."""

    def __init__(
        self,
        tgt_dict,
        softmax_batch=None,
        compute_alignment=False,
        eos=None,
        symbols_to_strip_from_output=None,
        lm_model=None,
        lm_weight=1.0,
        ent_threshold=0.0,
    ):
        self.pad = tgt_dict.pad()
        self.eos = tgt_dict.eos() if eos is None else eos
        self.softmax_batch = softmax_batch or sys.maxsize
        assert self.softmax_batch > 0
        self.compute_alignment = compute_alignment
        self.symbols_to_strip_from_output = (
            symbols_to_strip_from_output.union({self.eos})
            if symbols_to_strip_from_output is not None
            else {self.eos}
        )

        # ---------------- LIAM ----------------
        self.lm_model = lm_model
        self.lm_weight = lm_weight
        self.ent_threshold = ent_threshold
        if self.lm_model is not None:
            self.lm_model.eval()
        # ---------------- LIAM ----------------


    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Score a batch of translations."""
        net_input = sample["net_input"]

        def batch_for_softmax(dec_out, target):
            # assumes decoder_out[0] is the only thing needed (may not be correct for future models!)
            first, rest = dec_out[0], dec_out[1:]
            bsz, tsz, dim = first.shape
            if bsz * tsz < self.softmax_batch:
                yield dec_out, target, True
            else:
                flat = first.contiguous().view(1, -1, dim)
                flat_tgt = target.contiguous().view(flat.shape[:-1])
                s = 0
                while s < flat.size(1):
                    e = s + self.softmax_batch
                    yield (flat[:, s:e],) + rest, flat_tgt[:, s:e], False
                    s = e

        def gather_target_probs(probs, target):
            probs = probs.gather(
                dim=2,
                index=target.unsqueeze(-1),
            )
            return probs

        orig_target = sample["target"]

        # compute scores for each model in the ensemble
        avg_probs = None
        avg_attn = None
        for model in models:
            model.eval()
            decoder_out = model(**net_input)
            attn = decoder_out[1] if len(decoder_out) > 1 else None
            if type(attn) is dict:
                attn = attn.get("attn", None)

            batched = batch_for_softmax(decoder_out, orig_target)
            probs, idx = None, 0
            for bd, tgt, is_single in batched:
                sample["target"] = tgt
                curr_prob = model.get_normalized_probs(
                    bd, log_probs=len(models) == 1, sample=sample
                ).data

                # ---------------- LIAM ----------------
                if not is_single:
                    NotImplementedError("All modifications assume a single model")
                curr_prob_not_log = model.get_normalized_probs(
                    bd, log_probs=False, sample=sample
                ).data
                ent = -(curr_prob*curr_prob_not_log).sum(-1)

                if self.lm_model is not None:
                    # FIXME: This might be wrong
                    lm_out = self.lm_model(tgt)
                    lm_prob = self.lm_model.get_normalized_probs(
                        lm_out, log_probs=True, sample=sample
                    ).data

                    lm_prob_not_log = self.lm_model.get_normalized_probs(
                        lm_out, log_probs=False, sample=sample
                    ).data

                    lm_ent = -(lm_prob*lm_prob_not_log).sum(-1)

                    sm_prob = curr_prob
                    high_ents = ent > self.ent_threshold
                    weights = self.lm_weight * high_ents
                    mmi_prob = curr_prob + lm_prob * weights[:, :, None]
                    # Normalize the probabilities
                    Z = torch.logsumexp(mmi_prob, -1)
                    curr_prob = mmi_prob - Z[:,:,None]

                    sm_prob = gather_target_probs(sm_prob, orig_target).view(sample["target"].shape)
                    lm_prob = gather_target_probs(lm_prob, orig_target).view(sample["target"].shape)

                rank = curr_prob.argsort(descending=True).add(1)
                rank = gather_target_probs(rank, orig_target).view(sample["target"].shape)
                # ---------------- LIAM ----------------

                if is_single:
                    probs = gather_target_probs(curr_prob, orig_target)
                else:
                    if probs is None:
                        probs = curr_prob.new(orig_target.numel())
                    step = curr_prob.size(0) * curr_prob.size(1)
                    end = step + idx
                    tgt_probs = gather_target_probs(
                        curr_prob.view(tgt.shape + (curr_prob.size(-1),)), tgt
                    )
                    probs[idx:end] = tgt_probs.view(-1)
                    idx = end
                sample["target"] = orig_target

            probs = probs.view(sample["target"].shape)

            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs.add_(probs)
            if attn is not None:
                if torch.is_tensor(attn):
                    attn = attn.data
                else:
                    attn = attn[0]
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        if len(models) > 1:
            avg_probs.div_(len(models))
            avg_probs.log_()
            if avg_attn is not None:
                avg_attn.div_(len(models))

        bsz = avg_probs.size(0)
        hypos = []
        start_idxs = sample["start_indices"] if "start_indices" in sample else [0] * bsz
        for i in range(bsz):
            # remove padding from ref
            ref = (
                utils.strip_pad(sample["target"][i, start_idxs[i] :], self.pad)
                if sample["target"] is not None
                else None
            )
            tgt_len = ref.numel()
            avg_probs_i = avg_probs[i][start_idxs[i] : start_idxs[i] + tgt_len]
            score_i = avg_probs_i.sum() / tgt_len
            # ---------------- LIAM ----------------
            rank_i = rank[i][start_idxs[i] : start_idxs[i] + tgt_len]
            ent_i = ent[i][start_idxs[i] : start_idxs[i] + tgt_len]
            if self.lm_model is not None:
                lm_ent_i = lm_ent[i][start_idxs[i] : start_idxs[i] + tgt_len]
                sm_prob_i = sm_prob[i][start_idxs[i] : start_idxs[i] + tgt_len]
                lm_prob_i = lm_prob[i][start_idxs[i] : start_idxs[i] + tgt_len]
            # ---------------- LIAM ----------------
            if avg_attn is not None:
                avg_attn_i = avg_attn[i]
                if self.compute_alignment:
                    alignment = utils.extract_hard_alignment(
                        avg_attn_i,
                        sample["net_input"]["src_tokens"][i],
                        sample["target"][i],
                        self.pad,
                        self.eos,
                    )
                else:
                    alignment = None
            else:
                avg_attn_i = alignment = None
            # ---------------- LIAM ----------------
            if self.lm_model is None:
                hypos.append(
                    [
                        {
                            "tokens": ref,
                            "score": score_i,
                            "attention": avg_attn_i,
                            "alignment": alignment,
                            "positional_scores": avg_probs_i,
                            "rank": rank_i,
                            "sm_entropy": ent_i,
                        }
                    ]
                )
            else:
                hypos.append(
                    [
                        {
                            "tokens": ref,
                            "score": score_i,
                            "attention": avg_attn_i,
                            "alignment": alignment,
                            "positional_scores": avg_probs_i,
                            "rank": rank_i,
                            "sm_entropy": ent_i,
                            "lm_entropy": lm_ent_i,
                            "sm_pos_scores": sm_prob_i,
                            "lm_pos_scores": lm_prob_i,
                        }
                    ]
                )
            # ---------------- LIAM ----------------
        return hypos
