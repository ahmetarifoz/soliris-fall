# Fall Detection Agent

## Overview
Bu ajan, YOLOv11 pose modeli kullanarak insan düşmelerini tespit eden bir video analiz sistemidir. RTSP, webcam veya video dosyası kaynaklarını izler. Temporal heuristikler gürültülü pose tahminlerini yumuşatır, HUD overlay operatörü aktif track'ler hakkında bilgilendirir. Düşme onaylandığında webhook callback ve OP500 entegrasyonu ile downstream sistemlere bildirim gönderir.

## Proje Yapısı

```
fall/
├── app/                    # Ana uygulama modülleri
│   ├── __init__.py         # Paket exports
│   ├── api.py              # REST API server (FastAPI)
│   ├── config.py           # Yapılandırma yönetimi (dataclasses)
│   ├── detector.py         # Düşme algılama pipeline'ı
│   ├── run.py              # Ana çalışma döngüsü
│   ├── stream.py           # Video stream okuyucu + RTP loopback
│   ├── alert.py            # Webhook ve OP500 trigger desteği
│   ├── webhook.py          # Legacy webhook client
│   ├── hud.py              # Görsel overlay ve telemetri
│   └── logger.py           # Merkezi loglama sistemi
├── models/                 # YOLO model ağırlıkları (git-ignored)
├── config.yaml             # Ana yapılandırma dosyası
├── config.example.yaml     # Örnek yapılandırma
└── requirements.txt        # Python bağımlılıkları
```

## Modül Açıklamaları

### `app.api`
REST API server (FastAPI). Detector'ı uzaktan kontrol etmek için standart endpoint'ler sağlar.
- `GET /` - Health check
- `GET /status` - Detection durumu (enabled, active_tracks, rtp_active)
- `POST /enable` - Detection'ı etkinleştir
- `POST /disable` - Detection'ı devre dışı bırak
- `POST /reset-cooldown` - Cooldown timer'larını sıfırla
- `GET /config` - Mevcut yapılandırmayı getir
- `PUT /config` - Yapılandırmayı güncelle (runtime)
- `GET /tracks` - Aktif track bilgilerini getir
- `POST /rtp/start` - RTP streaming başlat
- `POST /rtp/stop` - RTP streaming durdur

### `app.run`
Ana çalışma döngüsü. Video kaynağını hazırlar, frame skip yönetir, HUD render eder ve düşme olaylarını alert sistemine yönlendirir.
- `StreamReader` veya `cv2.VideoCapture` ile frame okuma
- RTSP reconnect mantığı (exponential backoff)
- OP500 trigger ve async webhook gönderimi
- API server başlatma (opsiyonel)
- **Lazy Stream Init**: Detection disabled iken stream okuma başlatılmaz, kaynak tüketimi minimize edilir

### `app.detector`
Düşme algılama pipeline'ı. Ultralytics tracking'i sarar, per-track state tutar ve fall state machine'i uygular.
- **State Machine**: `idle` → `candidate` → `fallen` → `recover` → `idle`
- **FallEvent**: Düşme olayı veri yapısı (track_id, angle, confidence, timestamp)
- **FrameResult**: Frame işleme sonucu (frame, tracks_view, events)
- Per-track cooldown ve alert hold mantığı
- RTP loopback state tracking
- **Initial State**: Detection varsayılan olarak `disabled` başlar, API ile etkinleştirilmelidir

### `app.stream`
Video stream okuyucu. FFmpeg veya OpenCV backend desteği, RTP loopback özelliği.
- Hardware acceleration (CUDA) desteği
- RTSP over TCP optimizasyonları
- RTP sender (FFmpeg subprocess ile H.264 encoding)
- OP500 sender registration
- Auto-reconnect ve health check

### `app.alert`
Webhook ve OP500 trigger desteği.
- `send_webhook()`: HMAC-SHA256 imzalı async webhook
- `send_op500_trigger()`: OP500 trigger gönderimi
- `RtpLoopbackManager`: RTP loopback state yönetimi ve auto-close

### `app.webhook`
Legacy webhook client. Per-track cooldown ile HMAC imzalı HTTP POST.

### `app.hud`
Görsel overlay ve telemetri kartları. Per-track bilgi panelleri, state badge'leri.
- **RTP Status**: Header'da ve sağ alt köşede RTP durumu göstergesi (ON/OFF, port)
- **Cooldown Indicator**: Track kartlarında per-track cooldown sayacı (CD:X.Xs)
- **Status Panel**: Sağ alt köşede RTP durumu ve alert cooldown ayarı

### `app.logger`
Merkezi loglama sistemi. Renkli konsol çıktısı, dosya logging desteği.
- Alert logları: `alert_candidate()`, `alert_fallen()`, `alert_recovered()`
- RTP logları: `rtp_started()`, `rtp_stopped()`
- Webhook logları: `webhook_sent()`, `webhook_failed()`
- Stream logları: `stream_connected()`, `stream_disconnected()`, `stream_reconnecting()`

### `app.config`
Yapılandırma yönetimi. YAML dosyasından dataclass'lara parse.
- `AppSection`: Model, device, HUD, RTSP ayarları
- `DetectSection`: Algılama parametreleri (threshold, cooldown, temporal window)
- `WebhookSection`: Webhook URL, HMAC secret, timeout
- `Op500Section`: OP500 entegrasyonu ayarları
- `ApiSection`: API server ayarları (enabled, host, port)
- `IngestSection`: Video ingest backend ayarları
- `StreamSection`: Per-stream yapılandırma (camera_id, source, RTP port)
- `HudSection`: HUD persistence ayarları
- `resolve_stream_ingest()`: Stream ayarlarını ingest'ten inherit eder

## Yapılandırma Bölümleri

### `app`
```yaml
app:
  model_path: models/yolo11m-pose.pt
  device: 0                    # GPU index veya "cpu"
  enable_hud: true             # Görsel overlay
  rtsp_backend: ffmpeg         # "ffmpeg" veya "opencv"
  rtsp_reconnect: true         # Otomatik yeniden bağlanma
```

### `detect`
```yaml
detect:
  conf: 0.16                   # Pose confidence threshold
  angle_th: 46                 # Düşme açısı eşiği (derece)
  cooldown_sec: 0.4            # Per-track cooldown
  candidate_alert_sec: 10.0    # Candidate alert gecikmesi
  fallen_alert_sec: 5.0        # Fallen alert gecikmesi
  temporal_window: 60          # Temporal smoothing penceresi
```

### `webhook`
```yaml
webhook:
  url: "https://example.com/fall-alert"
  hmac_secret: "your-secret"
  cooldown_seconds: 5.0
```

### `op500`
```yaml
op500:
  enabled: true
  base_url: "http://localhost:8080/"
  rtp_auto_close_seconds: 120
```

### `streams`
```yaml
streams:
  - camera_id: "cam1"
    source: "rtsp://user:pass@camera:554/stream"
    rtp_port: 10160
    rtp_loopback_enabled: true
```

## Event Flow

1. `app.run` yapılandırmayı yükler, `FallDetector` ve `StreamReader` başlatır
2. Her frame `detector.process_frame()` ile işlenir → `FrameResult` döner
3. `FrameResult.events` boş değilse:
   - Legacy webhook client'a gönderilir
   - Async webhook + OP500 trigger tetiklenir
   - RTP loopback başlatılır (eğer etkinse)
4. HUD overlay frame üzerine çizilir
5. Track'ler `detect.prune_sec` süresince görünür kalır

## Log Çıktısı Örnekleri

```
[12:34:56.789] [INFO] 🎥 STREAM CONNECTED | rtsp://camera:554/stream
[12:34:57.123] [WARN] ⚠️  CANDIDATE | track=1 cam=cam1 angle=52.3° conf=0.87
[12:35:02.456] [CRIT] 🚨 FALL DETECTED | track=1 cam=cam1 angle=68.5° conf=0.92
[12:35:02.789] [INFO] 📤 WEBHOOK | cam=cam1 status=fallen track=1
[12:35:03.012] [INFO] 🔔 OP500 TRIGGER | port=10160 type=FALL
[12:35:15.345] [INFO] ✅ RECOVERED | track=1 cam=cam1
```

## Çalıştırma

```bash
# Bağımlılıkları yükle
pip install -r requirements.txt

# Uygulamayı başlat
python -m app.run

# Veya
python main.py
```

## Operasyonel Notlar

- Model ağırlıklarını `models/` altında tutun (git-ignored)
- Headless deployment için `app.enable_hud: false` ayarlayın
- Gizli bilgiler için `config.local.yaml` kullanın (base config'i override eder)
- Şu an tek stream desteklenir; multi-stream için `app.run` genişletilebilir

## Bağımlılıklar

- `ultralytics==8.3.197` - YOLO pose modeli
- `opencv-python>=4.10.0` - Video işleme
- `torch>=2.2.0` - PyTorch (CUDA desteği)
- `numpy>=1.24.0` - Sayısal işlemler
- `pyyaml>=6.0.0` - YAML parsing
- `requests>=2.31.0` - HTTP client
- `lap>=0.5.12` - Linear assignment (tracking)
