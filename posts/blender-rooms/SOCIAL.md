# Shareable media and copy

[MP4](assets/social/qwen-blender-rooms.mp4): 18 seconds, 1080×1080, H.264.
[GIF](assets/social/qwen-blender-rooms.gif): 18-second loop, 640×640.
[Poster](assets/social/poster.jpg) · [Captions](assets/social/captions.vtt) · [Provenance](assets/social/provenance.json).

## Short caption

Fine-tuned Qwen3.5-9B on 192 procedural Blender programs.

30/32 test renders. 18/32 geometry passes. $7.82 on Compute.

Actual outputs, including the failures:
https://letsusecompute.com/posts/blender-rooms/

## Longer post

I fine-tuned Qwen3.5-9B to write low-poly rooms in Blender.

192 procedural examples, one H100, and real saved adapter weights. On the primary test, 30/32 prompts rendered and 18/32 passed geometry checks. The completed pilot cost $7.82.

The slideshow shows the first palette from each test family, including the studio that failed. These are four layout groups with correlated color variants. A separate compatibility diagnostic broke all eight adapter programs; geometry and robustness remain open problems. We have no independent aesthetic score.

Code, actual outputs, and limitations: https://letsusecompute.com/posts/blender-rooms/

Inspired by Merve’s Blender challenge: https://huggingface.co/buckets/merve/blender-grpo-gepa

## Alt text

A slideshow of four actual renders from a fine-tuned Qwen3.5-9B model: a bedroom, studio, kitchen, and living room. Labels show that the pictured studio failed geometry checks; the other three pictured examples passed. The final card reports 30/32 test renders, 18/32 geometry passes, and $7.82 in completed-pilot GPU charges.

## Scope

This media covers the completed pilot. The separate repair experiment is excluded. This file is prepared copy; no social post has been sent.
