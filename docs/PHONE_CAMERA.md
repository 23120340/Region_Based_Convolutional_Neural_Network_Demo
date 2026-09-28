# Camera điện thoại qua QR code

Dùng camera điện thoại làm nguồn vào cho runtime hybrid (`scripts/run_hybrid.py`, `run_hybrid.ps1`). Pipeline YOLO → DINOv2/BiLSTM → Fusion → FSM không đổi; chỉ thêm một camera input adapter.

## Cách dùng

```powershell
python -m pip install -r requirements-camera.txt   # thêm aiohttp, aiortc, segno, cryptography
.\run_hybrid.ps1                   # camera PC; nhấn nút "Kết nối camera điện thoại" hoặc phím P
.\run_hybrid.ps1 -Source phone     # chỉ dùng điện thoại; QR hiện ngay khi mở
```

1. Nhấn **Kết nối camera điện thoại** (góc phải dưới) hoặc phím `P`. Panel bên phải hiện QR.
2. Quét QR bằng điện thoại **cùng Wi-Fi/LAN** với PC. Lần đầu trình duyệt cảnh báo chứng chỉ tự ký: chọn *Nâng cao → Tiếp tục truy cập* (Safari: *Hiển thị chi tiết → truy cập trang web này*).
3. Cho phép camera. Trang hiện preview và tự gửi video về PC. Có nút *Đổi camera trước/sau*, *Ngắt kết nối* và *Gửi về hệ thống* để kết nối lại.
4. Khi nhận được frame, dashboard ghi `ĐT: Connected · N FPS` và `Nguồn: điện thoại`, panel QR tự đóng. Frame điện thoại thay camera PC trong pipeline; khi điện thoại ngắt, runtime tự quay lại camera PC (hoặc màn hình chờ khi `--source phone`).

Mỗi lần đổi nguồn, runtime bắt đầu lượt mới (như phím `R`), vì vị trí khe đã ghi nhớ và cửa sổ thời gian của LSTM chỉ đúng với một góc nhìn.

Phím trong cửa sổ: `P` mở/đóng panel, `N` tạo QR mới, `D` ngắt điện thoại. Các nút trên panel làm việc tương tự; *Đổi IP* hiện khi PC có nhiều card mạng.

## Nội dung QR

```text
https://<IP-LAN>:<port>/phone/<session_id>#<token>
```

- `<IP-LAN>` được dò lúc tạo QR, không hard-code. Thứ tự ưu tiên: IP riêng (10.x, 172.16–31.x, 192.168.x) của card mạng đang giữ default route; card ảo (WSL, Hyper-V, VirtualBox, Docker…) và 169.254.x bị xếp sau hoặc loại.
- `<port>` mặc định `8443` (`--phone-port`); cổng bận thì tự thử 9 cổng kế tiếp.
- `session_id` và `token` sinh ngẫu nhiên cho mỗi QR. Token nằm sau `#` nên không đi vào request trang; trang đọc nó rồi gửi kèm mọi lệnh API.
- Điện thoại đầu tiên dùng token sẽ giữ phiên; máy khác quét cùng QR nhận lỗi *đã được điện thoại khác sử dụng*. Chính điện thoại đó vẫn được kết nối lại sau khi rớt mạng.
- QR chưa dùng hết hạn sau 10 phút (`--phone-session-ttl`). `QR mới` hoặc `Ngắt` thu hồi token cũ ngay.

Endpoint phía PC:

| Endpoint | Việc |
|---|---|
| `GET /phone/<session_id>` | Trang camera cho điện thoại |
| `POST /api/phone/<session_id>/offer` | WebRTC signalling: SDP offer → answer |
| `GET /api/phone/<session_id>/ws` | WebSocket dự phòng, frame JPEG |
| `POST /api/phone/<session_id>/hangup` | Điện thoại chủ động ngắt |

## Vì sao WebRTC, khi nào WebSocket

Runtime cần video liên tục (YOLO vài frame một lần, DINOv2 lấy mẫu theo thời gian cho BiLSTM), nên mặc định dùng **WebRTC** (`aiortc`): video nén VP8/H.264 realtime, không polling. Nếu WebRTC không nối được trong 10 giây (Wi-Fi chặn UDP giữa thiết bị, tunnel không có TURN, trình duyệt cũ), trang tự chuyển sang **WebSocket JPEG** trên cùng cổng HTTPS: mỗi lần chỉ một frame đang gửi, PC xác nhận xong mới gửi frame tiếp, nên không dồn hàng đợi. Ép một chế độ bằng `--phone-transport webrtc|websocket`.

HTTPS là bắt buộc vì trình duyệt điện thoại chỉ cho `getUserMedia` trên trang an toàn. Chứng chỉ tự ký được tạo tại `artifacts/phone_camera/tls/` với SAN gồm các IP hiện tại, và tự tạo lại khi IP đổi. Có chứng chỉ tin cậy (ví dụ `mkcert`) thì dùng `--phone-cert` và `--phone-key` để bỏ cảnh báo.

## Khi điện thoại không kết nối được

| Tình huống | Dashboard hiển thị | Cách xử lý |
|---|---|---|
| PC không có IP LAN (chỉ có loopback/169.254) | Cảnh báo, không hiện QR | Nối PC vào Wi-Fi/LAN hoặc bật hotspot điện thoại rồi cho PC vào hotspot đó, nhấn `QR mới` |
| QR chọn sai card mạng | Nút *Đổi IP* | Nhấn *Đổi IP* đến khi IP cùng dải với điện thoại, hoặc `--phone-host 192.168.x.y` |
| Điện thoại không mở được trang | Trạng thái giữ *Chờ quét QR* | Kiểm tra cùng Wi-Fi; tắt *client/AP isolation* (Wi-Fi khách thường bật); cho Python qua Windows Firewall ở mạng **Private** (hộp thoại xuất hiện lần đầu) |
| Mở trang được nhưng WebRTC lỗi | Trang ghi *Connected · WebSocket* | Tự dự phòng; có thể cần cho phép UDP trong firewall |
| Điện thoại khác mạng hoàn toàn | — | Chạy tunnel HTTPS tới `https://localhost:8443`, ví dụ `cloudflared tunnel --url https://localhost:8443 --no-tls-verify`, rồi thêm `--phone-public-url https://<tên>.trycloudflare.com`. Qua tunnel thường chỉ WebSocket chạy được; muốn WebRTC cần thêm TURN bằng `--phone-ice-server turn:...` |
| Điện thoại khóa màn hình | *Disconnected*, runtime quay về camera PC | Mở lại trang; trang tự kết nối lại cùng phiên |

## Cấu trúc code

```text
src/assembly/phone_network.py      # dò và xếp hạng IP LAN
src/assembly/phone_camera.py       # HTTPS + WebRTC/WebSocket server, session/token, QR
src/assembly/phone_camera_page.html# trang camera trên điện thoại
src/assembly/camera_input.py       # CameraInputRouter: adapter giống cv2.VideoCapture
src/assembly/phone_camera_ui.py    # nút/phím trên dashboard -> server
src/assembly/hybrid_dashboard.py   # vẽ nút, trạng thái Connected/Disconnected và panel QR
```

Kiểm thử: `python -m unittest tests.test_phone_camera -v` (gồm một phiên WebRTC thật qua `aiortc` và một phiên WebSocket; tự bỏ qua nếu chưa cài thư viện).
