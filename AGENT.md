# Fall Detection Agent

## Overview
This agent monitors a single video source (RTSP, webcam, or file) and detects human falls using the YOLOv11 pose model. Temporal heuristics smooth noisy pose estimates, while a HUD overlay keeps the operator informed about each active track. When a fall is confirmed, the agent can notify downstream systems via a signed webhook callback.

## Runtime Composition
- app.run – prepares the capture device, manages frame skipping, renders the HUD, and routes fall events to the notifier.
- app.detector – wraps Ultralytics tracking, maintains per-track state, and implements the fall state machine.
- app.hud – draws per-track telemetry cards and on-screen annotations.
- app.webhook – performs optional webhook delivery with HMAC-SHA256 signing and per-track cooldown.
- config.yaml – primary runtime configuration; override with config.local.yaml for secrets in production environments.

## Event Flow
1. app.run loads configuration and spins up FallDetector and WebhookClient.
2. Each processed frame yields a FrameResult. Overlay rendering and HUD drawing occur in-place on the frame buffer.
3. When FrameResult.events is non-empty, each FallEvent is submitted to the webhook client. Cooldown logic prevents alert storms.
4. Tracks remain visible on the HUD until they time out based on detect.prune_sec.

## Operational Notes
- Keep model weights under models/ (git-ignored in version control). Placeholders or .gitkeep files can sit beside private assets.
- For headless deployments set app.enable_hud to false; the agent continues to process frames and emit webhooks without creating a GUI window.
- Use config.local.yaml to store private endpoints or secrets. Values there override the base configuration without leaving your repository.
- The current runner focuses on a single stream. Extend app.run if you need multi-stream orchestration.

## Next Steps
- Integrate automated tests around the fall state machine (e.g., synthetic pose sequences) to protect against regressions.
- Consider persisting event logs locally or pushing them to a message queue in addition to the webhook.
- Harden reconnection logic for RTSP feeds (e.g., exponential backoff, health probes) before production rollout.
