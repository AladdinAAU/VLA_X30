"""verify: ask the reasoner VLM a yes/no/unsure question about the current camera view."""
from __future__ import annotations

import json
import math
import re

from .base import SkillConfig, SkillContext, SkillResult

PROMPT = (
    "You are the vision module of a quadruped robot that is searching for objects indoors. The first image is the robot's "
    "current forward camera view.{crop} Question: {question}\n"
    "Rules: answer 'yes' only if the thing is clearly visible and matches. Answer 'no' only if you see positive evidence that "
    "it is something else or clearly absent from a good view. Answer 'unsure' if the view is too far, too dark, cut off or "
    "ambiguous. Do not guess.\n"
    'Reply with JSON only: {{"answer": "yes" | "no" | "unsure", "plausibility": <0-100>, "reason": "<one short sentence about what you see>"}}'
)
CROP_TEXT = " The second image is the best stored crop of the object being checked (taken earlier, possibly from another angle)."


def parse_verdict(text: str) -> dict:
    """VLM reply -> {'answer': yes|no|unsure, 'plausibility': float, 'reason': str}. Unparseable -> unsure."""
    obj = None
    try:
        from x30_common.jsonrepair import loads_lenient
        obj, _ = loads_lenient(text)
    except Exception:
        m = re.search(r"\{.*\}", text or "", re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                obj = None
    if not isinstance(obj, dict):
        low = (text or "").lower()
        ans = "yes" if re.search(r"\byes\b", low) and not re.search(r"\bno\b", low) else "no" if re.search(r"\bno\b", low) and not re.search(r"\byes\b", low) else "unsure"
        return {"answer": ans, "plausibility": 50.0, "reason": f"free-text reply: {(text or '').strip()[:120]}", "parsed": False}
    ans = str(obj.get("answer", "unsure")).strip().lower()
    if ans not in ("yes", "no", "unsure"):
        ans = "unsure"
    try:
        pl = float(obj.get("plausibility", obj.get("confidence", 50)))
    except (TypeError, ValueError):
        pl = 50.0
    if pl <= 1.0 and pl > 0.0 and isinstance(obj.get("plausibility", obj.get("confidence")), float):
        pl *= 100.0
    return {"answer": ans, "plausibility": max(0.0, min(100.0, pl)), "reason": str(obj.get("reason", obj.get("evidence", ""))).strip(), "parsed": True}


def build_question(args: dict) -> str:
    if args.get("question"):
        return str(args["question"])
    label = str(args["label"])
    if args.get("object_id") is not None:
        return f"Is the object in the middle of the image a {label}? It should be within about two metres."
    return f"Is there a {label} clearly visible in this view, within about three metres of the robot?"


class VerifySkill:
    def __init__(self, api, cfg: SkillConfig, motion, client=None):
        self.api, self.cfg, self.motion = api, cfg, motion
        self._client = client

    def client(self):
        if self._client is None:
            from x30_common.config import load_models_config
            from x30_common.vlm_client import VLMClient
            models = load_models_config(self.cfg.models_config or None)
            self._client = VLMClient.from_config("reasoner", models, max_tokens=self.cfg.vlm_max_tokens,
                                                 temperature=self.cfg.vlm_temperature, timeout_s=self.cfg.vlm_timeout_s)
        return self._client

    def run(self, ctx: SkillContext, args: dict) -> SkillResult:
        api, cfg = self.api, self.cfg
        oid = args.get("object_id")
        obj = api.get_object(int(oid)) if oid is not None else None
        if oid is not None and obj is None:
            return SkillResult(False, f"Object #{oid} is not in the semantic map, so it cannot be checked. Next: search or scan first.", {"reason": "unknown_object"})
        data: dict = {"object_id": oid, "distance_m": None}
        turned = ""
        pose = api.map_pose()
        if obj is not None and pose is not None:
            delta = math.atan2(obj.y - pose.y, obj.x - pose.x) - pose.yaw
            delta = math.atan2(math.sin(delta), math.cos(delta))
            data["distance_m"] = round(math.hypot(obj.x - pose.x, obj.y - pose.y), 2)
            if args.get("face_object", True) and abs(delta) > math.radians(12):
                r = self.motion.rotate(ctx, delta)
                if r["reason"] in ("cancelled", "timeout"):
                    return SkillResult(False, f"Verify {r['reason']} while turning to face the object.", data)
                turned = f" (turned {math.degrees(r['achieved_rad']):.0f} deg to face it first)"
                ctx.sleep(0.7)                                  # fresh frame after the turn
        img = api.latest_rgb()
        if img is None:
            return SkillResult(False, f"No camera image on {cfg.rgb_topic} in the last 3 s. Check that the camera is running; "
                               "you cannot verify visually until it is.", {**data, "reason": "no_image"})
        from x30_common.images import array_from_image_msg, user_message
        from x30_common.vlm_client import VLMError
        crop = api.object_view(int(oid)) if oid is not None else None
        question = build_question(args)
        text = PROMPT.format(question=question, crop=CROP_TEXT if crop else "")
        try:
            images = [array_from_image_msg(img)] + ([crop] if crop else [])
            msg = user_message(text, images, max_side=cfg.vlm_image_max_side)
            res = self.client().chat([msg], cancel=ctx.cancel_event, timeout_s=min(cfg.vlm_timeout_s, max(ctx.remaining(), 5.0)))
        except VLMError as ex:
            if ctx.cancelled:
                return SkillResult(False, "Verify was cancelled.", {**data, "reason": "cancelled"})
            return SkillResult(False, f"The vision-language model could not be reached or failed ({type(ex).__name__}: {str(ex)[:120]}). "
                               "Next: retry once; if it keeps failing, decide without verification and say so.", {**data, "reason": "vlm_error"})
        except Exception as ex:                                   # image decoding etc.
            return SkillResult(False, f"Verify failed before asking the model: {type(ex).__name__}: {ex}", {**data, "reason": "error"})
        v = parse_verdict(res.content)
        data.update(answer=v["answer"], plausibility=v["plausibility"], reason=v["reason"])
        what = f"'{args['label']}'" if args.get("label") else "the question"
        dist = f" It is about {data['distance_m']:.1f} m away." if data["distance_m"] is not None else ""
        reason = v["reason"] or "no reason given"
        if v["answer"] == "yes":
            msg_txt = (f"YES for {what} (plausibility {v['plausibility']:.0f}/100){turned}: {reason}.{dist} "
                       "Next: if this satisfies the task and you are close enough, finish; otherwise approach and verify again.")
        elif v["answer"] == "no":
            msg_txt = (f"NO for {what} (plausibility {v['plausibility']:.0f}/100){turned}: {reason}.{dist} "
                       "A single viewpoint can mislead; if you doubt it, look again from another side before rejecting. "
                       "Next: reject this object (add its id to rejected_ids) and search for another.")
        else:
            msg_txt = (f"UNSURE for {what} (plausibility {v['plausibility']:.0f}/100){turned}: {reason}.{dist} "
                       "Next: get a better view (approach closer, turn_to the object, or move to another side) and verify again, "
                       "or ask a more specific question.")
        return SkillResult(True, msg_txt, data)
