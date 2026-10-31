use anyhow::Result;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread::{sleep, spawn};
use std::time::Duration;
use tracing::{error, info, warn};

use crate::bootloader::boot_device;
use crate::device::{scan_devices, DeviceState, PID_MYRIAD2_UNBOOTED};
use crate::graph::{dissect_graph, DissectedGraph};
use crate::health;
use crate::run::{self, InputKind, Preprocess};
use crate::v1::{Graph, GraphInfo, Link, Options};
use anyhow::Context;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub struct ServerState {
    pub firmware_path: String,
    /// Held while anything talks to the stick, so requests never interleave on the USB link.
    pub device: Mutex<()>,
}

fn json_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c if (c as u32) < 0x20 => o.push(' '),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

pub fn start_server(bind_addr: &str, firmware_path: String, auto_boot: bool) -> Result<()> {
    let state = Arc::new(ServerState { firmware_path, device: Mutex::new(()) });

    if auto_boot {
        let wd = Arc::clone(&state);
        spawn(move || {
            info!("auto-boot watchdog active (interval 5s)");
            loop {
                sleep(Duration::from_secs(5));
                let devices = scan_devices().unwrap_or_default();
                if devices.iter().any(|d| d.pid == PID_MYRIAD2_UNBOOTED) {
                    info!("watchdog: unbooted stick (03e7:2150) found, booting");
                    let _g = wd.device.lock().unwrap();
                    match boot_device(&wd.firmware_path) {
                        Ok(res) => info!("watchdog: booted into 0x{:04x} in {:.2?}", res.re_enumerated_pid, res.total_boot_duration),
                        Err(e) => warn!("watchdog: boot failed: {e:#}"),
                    }
                }
            }
        });
    }

    let listener = TcpListener::bind(bind_addr)?;
    info!("Movidius HTTP daemon listening on http://{bind_addr}");
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let st = Arc::clone(&state);
                spawn(move || {
                    if let Err(e) = handle_client(stream, st) {
                        warn!("client error: {e}");
                    }
                });
            }
            Err(e) => error!("accept error: {e}"),
        }
    }
    Ok(())
}

/// Read the v1 state word. Returns (http status, json body).
fn v1_status(state: &ServerState) -> (u16, String) {
    let Ok(_g) = state.device.try_lock() else {
        return (200, "{\"status\":\"busy\",\"detail\":\"another request holds the stick\"}".into());
    };
    match Link::open().and_then(|l| health::check(&l)) {
        Ok(h) => (
            200,
            format!(
                "{{\"status\":\"ok\",\"protocol\":\"ncsdk-v1\",\"myriad_state\":{},\"status_read_us\":{:.0}}}",
                json_str(&h.state.name()),
                h.status_read_us
            ),
        ),
        Err(e) => (503, format!("{{\"status\":\"error\",\"error\":{}}}", json_str(&format!("{e:#}")))),
    }
}

fn handle_client(mut stream: TcpStream, state: Arc<ServerState>) -> Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(3)))?;
    let mut buffer = [0u8; 2048];
    let n = stream.read(&mut buffer)?;
    if n == 0 {
        return Ok(());
    }
    let request = String::from_utf8_lossy(&buffer[..n]);
    let mut parts = request.lines().next().unwrap_or("").split_whitespace();
    let method = parts.next().unwrap_or("GET");
    let target = parts.next().unwrap_or("/");
    let (path, query) = target.split_once('?').unwrap_or((target, ""));

    let (code, body) = match (method, path) {
        ("GET", "/health") => {
            let devices = scan_devices().unwrap_or_default();
            if let Some(d) = devices.iter().find(|d| d.state == DeviceState::Booted) {
                let (c, v1) = v1_status(&state);
                let healthy = c == 200 && v1.contains("\"status\":\"ok\"");
                let status = if healthy { "healthy" } else if v1.contains("busy") { "busy" } else { "degraded" };
                (
                    if status == "degraded" { 503 } else { 200 },
                    format!(
                        "{{\"status\":\"{status}\",\"device\":{},\"usb\":\"booted\",\"pid\":\"0x{:04x}\",\"v1\":{}}}",
                        json_str(d.chip_generation), d.pid, v1
                    ),
                )
            } else if let Some(d) = devices.iter().find(|d| d.state == DeviceState::Unbooted) {
                (200, format!("{{\"status\":\"standby\",\"device\":{},\"usb\":\"unbooted\",\"pid\":\"0x{:04x}\"}}", json_str(d.chip_generation), d.pid))
            } else {
                (503, "{\"status\":\"not_found\",\"error\":\"No Movidius VPU detected\"}".into())
            }
        }
        ("GET", "/status") => {
            let list: Vec<String> = scan_devices()
                .unwrap_or_default()
                .iter()
                .map(|d| format!(
                    "{{\"generation\":{},\"vid\":\"0x{:04x}\",\"pid\":\"0x{:04x}\",\"state\":\"{:?}\",\"bus\":{},\"port\":{}}}",
                    json_str(d.chip_generation), d.vid, d.pid, d.state, d.bus_number, json_str(&d.port_path)))
                .collect();
            (200, format!("{{\"devices\":[{}],\"node\":{}}}", list.join(","), json_str(&node_name())))
        }
        ("GET", "/v1/status") => v1_status(&state),
        ("POST", "/ping") | ("GET", "/ping") => {
            let count = query
                .split('&')
                .find_map(|kv| kv.strip_prefix("count="))
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(20)
                .clamp(1, 1000);
            let _g = state.device.lock().unwrap();
            match Link::open() {
                Ok(link) => {
                    let s = health::ping(&link, count);
                    let st = s.last_state.map(|x| x.name()).unwrap_or_else(|| "none".into());
                    (
                        if s.valid > 0 { 200 } else { 503 },
                        format!(
                            "{{\"protocol\":\"ncsdk-v1 status read\",\"sent\":{},\"valid\":{},\"invalid\":{},\"rtt_us\":{{\"min\":{:.0},\"p50\":{:.0},\"avg\":{:.0},\"max\":{:.0}}},\"myriad_state\":{},\"last_error\":{}}}",
                            s.sent, s.valid, s.invalid, s.min_us, s.p50_us, s.avg_us, s.max_us, json_str(&st), s.last_error.as_deref().map(json_str).unwrap_or_else(|| "null".into())
                        ),
                    )
                }
                Err(e) => (503, format!("{{\"status\":\"error\",\"error\":{}}}", json_str(&format!("{e:#}")))),
            }
        }
        ("POST", "/boot") => {
            let _g = state.device.lock().unwrap();
            match boot_device(&state.firmware_path) {
                Ok(res) => (
                    200,
                    format!(
                        "{{\"status\":\"ok\",\"booted_pid\":\"0x{:04x}\",\"upload_ms\":{:.2},\"total_ms\":{:.2}}}",
                        res.re_enumerated_pid,
                        res.upload_duration.as_secs_f64() * 1000.0,
                        res.total_boot_duration.as_secs_f64() * 1000.0
                    ),
                ),
                Err(e) => (500, format!("{{\"status\":\"error\",\"error\":{}}}", json_str(&format!("{e:#}")))),
            }
        }
        _ => (404, "{\"error\":\"Not Found\"}".into()),
    };

    let reason = match code { 200 => "OK", 404 => "Not Found", 500 => "Internal Server Error", 503 => "Service Unavailable", _ => "OK" };
    let resp = format!(
        "HTTP/1.1 {code} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(resp.as_bytes())?;
    stream.flush()?;
    Ok(())
}

pub struct LemonadeEngineState {
    pub model_path: String,
    pub firmware_path: String,
    pub graph_blob: Vec<u8>,
    pub graph_info: DissectedGraph,
    pub device_idx: String,
    /// The graph is uploaded to the stick once (lazily, on the first /v1/infer) and stays loaded:
    /// the firmware cannot take a second graph without a reset. The lock also serialises requests.
    session: Mutex<Option<Graph<'static>>>,
    pub requests_served: AtomicU64,
}

impl LemonadeEngineState {
    /// Validate the graph the same way the stick-side loader will, and build the state.
    pub fn new(model_path: &str, firmware_path: &str, device_idx: &str, graph_blob: Vec<u8>) -> Result<Self> {
        let graph_info = dissect_graph(&graph_blob)
            .with_context(|| format!("Invalid Movidius NCSDK v1 .graph checkpoint: {model_path}"))?;
        GraphInfo::parse(&graph_blob)
            .with_context(|| format!("Graph {model_path} would be rejected by the host loader"))?;
        Ok(Self {
            model_path: model_path.to_string(),
            firmware_path: firmware_path.to_string(),
            graph_blob,
            graph_info,
            device_idx: device_idx.to_string(),
            session: Mutex::new(None),
            requests_served: AtomicU64::new(0),
        })
    }

    /// Input tensor shape (width, height, channels) taken from the first stage of the graph.
    pub fn input_shape(&self) -> (u32, u32, u32) {
        self.graph_info
            .stages
            .first()
            .map(|s| (s.dim_x, s.dim_y, s.dim_z))
            .unwrap_or((0, 0, 0))
    }

    fn input_len(&self) -> usize {
        let (w, h, c) = self.input_shape();
        (w * h * c) as usize
    }

    /// Run one inference on the stick, uploading the graph first if needed.
    /// On any device error the stick is reset and re-booted so the next request starts clean.
    fn run_on_stick(&self, input: &[u16], want_aux: bool) -> Result<crate::v1::InferenceResult> {
        let mut guard = self.session.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            let link = crate::v1::prepare_link(&self.firmware_path)?;
            *guard = Some(Arc::new(link).allocate_graph_owned(&self.graph_blob, Options::default())?);
        }
        let g = guard.as_mut().expect("session just created");
        let outcome = (|| {
            let t = g.submit(input)?;
            g.collect(t, want_aux)
        })();
        match outcome {
            Ok(r) => {
                if let Some(e) = &r.device_error {
                    let e = e.clone();
                    self.recover(&mut guard);
                    anyhow::bail!("firmware reported an error: {e}");
                }
                Ok(r)
            }
            Err(e) => {
                self.recover(&mut guard);
                Err(e)
            }
        }
    }

    fn recover(&self, guard: &mut Option<Graph<'static>>) {
        if let Some(g) = guard.take() {
            let _ = g.reset_device();
            drop(g);
            std::thread::sleep(Duration::from_millis(1200));
            let _ = boot_device(&self.firmware_path);
        }
    }
}

/// Start a native Lemonade backend server (`movidius-rust serve --model <path> --host <host> --port <port> --device <idx>`).
/// Validates the NCSDK v1 `.graph` checkpoint at startup and serves `/health`, `/v1/models`, `/v1/stats`
/// and `POST /v1/infer`. Chat and text completions answer 501: the NCS1 runs compiled vision graphs only.
pub fn start_lemonade_server(model_path: &str, host: &str, port: u16, device_idx: &str, firmware: &str) -> Result<()> {
    let graph_blob = std::fs::read(model_path)
        .with_context(|| format!("Failed to read Movidius .graph checkpoint: {model_path}"))?;
    let state = Arc::new(LemonadeEngineState::new(model_path, firmware, device_idx, graph_blob)?);

    let bind_addr = format!("{host}:{port}");
    let listener = TcpListener::bind(&bind_addr)
        .with_context(|| format!("Failed to bind Lemonade Movidius backend on {bind_addr}"))?;
    let (w, h, c) = state.input_shape();
    info!(
        "Movidius backend ready on http://{} for graph '{}' ({} stages, SHAVEs {}..={}, {} weight bytes, input {}x{}x{})",
        bind_addr,
        model_path,
        state.graph_info.nstages,
        state.graph_info.first_shave,
        state.graph_info.last_shave,
        state.graph_info.weights_len,
        w,
        h,
        c
    );

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let st = Arc::clone(&state);
                spawn(move || {
                    if let Err(e) = handle_lemonade_client(stream, st) {
                        warn!("Lemonade backend client error: {e}");
                    }
                });
            }
            Err(e) => error!("Lemonade backend accept error: {e}"),
        }
    }
    Ok(())
}

const MAX_BODY: usize = 64 * 1024 * 1024;

struct HttpRequest {
    method: String,
    path: String,
    query: String,
    content_type: String,
    body: Vec<u8>,
}

fn read_http_request(stream: &mut TcpStream) -> Result<HttpRequest> {
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    let mut buf = Vec::with_capacity(4096);
    let mut chunk = [0u8; 8192];
    let mut header_end = None;
    let mut content_len = 0usize;
    let mut content_type = String::new();

    loop {
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
        if header_end.is_none() {
            if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                let hend = pos + 4;
                header_end = Some(hend);
                let hdr_str = String::from_utf8_lossy(&buf[..hend]);
                for line in hdr_str.lines() {
                    if let Some((k, v)) = line.split_once(':') {
                        if k.trim().eq_ignore_ascii_case("content-length") {
                            content_len = v.trim().parse::<usize>().unwrap_or(0);
                        } else if k.trim().eq_ignore_ascii_case("content-type") {
                            content_type = v.trim().to_ascii_lowercase();
                        }
                    }
                }
                if content_len > MAX_BODY {
                    anyhow::bail!("request body of {content_len} bytes exceeds the {MAX_BODY} byte limit");
                }
            }
        }
        if let Some(hend) = header_end {
            if buf.len() >= hend + content_len {
                break;
            }
        }
    }

    let hend = header_end.unwrap_or(buf.len());
    let hdr_str = String::from_utf8_lossy(&buf[..hend]).to_string();
    let first_line = hdr_str.lines().next().unwrap_or("");
    let mut parts = first_line.split_whitespace();
    let method = parts.next().unwrap_or("GET").to_string();
    let target = parts.next().unwrap_or("/").to_string();
    let (path, query) = target.split_once('?').unwrap_or((&target, ""));
    let body = buf[hend.min(buf.len())..].to_vec();
    Ok(HttpRequest { method, path: path.to_string(), query: query.to_string(), content_type, body })
}

fn query_get<'a>(query: &'a str, key: &str) -> Option<&'a str> {
    query.split('&').find_map(|kv| {
        let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
        (k == key).then_some(v)
    })
}

/// A parsed, validated `POST /v1/infer` request.
#[derive(Debug)]
pub struct InferRequest {
    pub input_f16: Vec<u16>,
    pub top: usize,
    pub full_output: bool,
    pub want_aux: bool,
}

/// Parse the query string and body of `POST /v1/infer`. Pure: touches no hardware.
///
/// Query: `format=image|f32|f16` (default: image for image/* content types, else f32 for
/// octet-stream), `top=K` (default 5), `output=1` to include the full output vector, `aux=0` to skip the
/// firmware timing read, and for images `width`, `height`, `mean=a,b,c` (default 104,117,123),
/// `scale` (default 1), `order=bgr|rgb` (default bgr). Width/height default to the graph's first stage.
pub fn parse_infer_request(
    query: &str,
    content_type: &str,
    body: &[u8],
    shape: (u32, u32, u32),
) -> std::result::Result<InferRequest, String> {
    if body.is_empty() {
        return Err("empty request body: POST image bytes, or raw f32/f16 HWC values".into());
    }
    let kind = match query_get(query, "format") {
        Some(f) => InputKind::parse(f).ok_or_else(|| format!("unknown format '{f}' (use image, f32 or f16)"))?,
        None if content_type.starts_with("image/") => InputKind::Image,
        None if content_type.starts_with("application/octet-stream") => InputKind::F32,
        None => InputKind::Image,
    };
    let num = |key: &str, default: f32| -> std::result::Result<f32, String> {
        match query_get(query, key) {
            Some(v) => v.parse::<f32>().map_err(|_| format!("'{key}' must be a number")),
            None => Ok(default),
        }
    };
    let top = match query_get(query, "top") {
        Some(v) => v.parse::<usize>().map_err(|_| "'top' must be a positive integer".to_string())?,
        None => 5,
    };
    if top == 0 || top > 1000 {
        return Err("'top' must be between 1 and 1000".into());
    }
    let mean = match query_get(query, "mean") {
        Some(m) => {
            let v: Vec<f32> = m.split(',').filter_map(|x| x.parse().ok()).collect();
            if v.len() != 3 || m.split(',').count() != 3 {
                return Err("'mean' needs three comma separated numbers".into());
            }
            [v[0], v[1], v[2]]
        }
        None => [104.0, 117.0, 123.0],
    };
    let width = num("width", shape.0 as f32)? as u32;
    let height = num("height", shape.1 as f32)? as u32;
    if kind == InputKind::Image {
        if width == 0 || height == 0 || width > 4096 || height > 4096 {
            return Err("image width/height must be between 1 and 4096".into());
        }
        if shape.2 != 3 {
            return Err(format!("the graph takes {} input channels; image input produces 3, send f32/f16 instead", shape.2));
        }
    }
    let pre = Preprocess {
        width,
        height,
        mean,
        scale: num("scale", 1.0)?,
        bgr: !query_get(query, "order").is_some_and(|o| o.eq_ignore_ascii_case("rgb")),
    };
    let input_f16 = run::decode_input(body, kind, &pre).map_err(|e| format!("{e:#}"))?;
    let expected = (shape.0 * shape.1 * shape.2) as usize;
    if input_f16.len() != expected {
        return Err(format!(
            "input has {} values, the graph expects {} ({}x{}x{} HWC)",
            input_f16.len(),
            expected,
            shape.0,
            shape.1,
            shape.2
        ));
    }
    Ok(InferRequest {
        input_f16,
        top,
        full_output: query_get(query, "output") == Some("1"),
        want_aux: query_get(query, "aux") != Some("0"),
    })
}

fn not_text_generation_error() -> String {
    "{\"error\":{\"message\":\"This engine runs compiled vision graphs on an Intel NCS1 and does not generate text. Use POST /v1/infer with an image or tensor.\",\"type\":\"not_implemented\",\"code\":\"text_generation_unsupported\"}}".to_string()
}

fn error_json(code: &str, msg: &str) -> String {
    format!("{{\"error\":{{\"message\":{},\"type\":{}}}}}", json_str(msg), json_str(code))
}

fn f32_json(v: f32) -> String {
    if v.is_finite() { format!("{v}") } else { "null".into() }
}

/// Route one request. Anything that touches the stick goes through `run_on_stick`.
fn lemonade_route(state: &LemonadeEngineState, req: &HttpRequest) -> (u16, String) {
    let now_ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    let (w, h, c) = state.input_shape();
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/health") => {
            let booted = scan_devices().unwrap_or_default().iter().any(|d| d.state == DeviceState::Booted);
            let status = if booted { "ok" } else { "no_device" };
            (
                if booted { 200 } else { 503 },
                format!(
                    "{{\"status\":\"{status}\",\"engine\":\"movidius-rust\",\"model\":{},\"graph_valid\":true,\"stick_booted\":{booted},\"version\":{},\"stages\":{},\"shaves\":\"{}..={}\",\"weights_bytes\":{}}}",
                    json_str(&state.model_path),
                    state.graph_info.version,
                    state.graph_info.nstages,
                    state.graph_info.first_shave,
                    state.graph_info.last_shave,
                    state.graph_info.weights_len
                ),
            )
        }
        ("GET", "/v1/models") => (
            200,
            format!(
                "{{\"object\":\"list\",\"data\":[{{\"id\":{},\"object\":\"model\",\"created\":{now_ts},\"owned_by\":\"movidius-rust\",\"type\":\"vision_graph\",\"generates_text\":false,\"input\":{{\"width\":{w},\"height\":{h},\"channels\":{c},\"layout\":\"HWC\",\"dtype\":\"fp16\"}},\"output_values\":{},\"endpoint\":\"/v1/infer\"}}]}}",
                json_str(&state.model_path),
                GraphInfo::parse(&state.graph_blob).map(|i| i.noutputs).unwrap_or(0)
            ),
        ),
        ("GET", "/v1/stats") => {
            let devices = scan_devices().unwrap_or_default();
            (
                200,
                format!(
                    "{{\"engine\":\"movidius-rust\",\"model\":{},\"device_index\":{},\"stages\":{},\"shaves\":{},\"weights_bytes\":{},\"usb_vpus_detected\":{},\"infer_requests\":{}}}",
                    json_str(&state.model_path),
                    json_str(&state.device_idx),
                    state.graph_info.nstages,
                    state.graph_info.last_shave - state.graph_info.first_shave + 1,
                    state.graph_info.weights_len,
                    devices.len(),
                    state.requests_served.load(Ordering::Relaxed)
                ),
            )
        }
        ("POST", "/v1/chat/completions") | ("POST", "/v1/completions") => (501, not_text_generation_error()),
        ("POST", "/v1/infer") => {
            let ir = match parse_infer_request(&req.query, &req.content_type, &req.body, state.input_shape()) {
                Ok(r) => r,
                Err(m) => return (400, error_json("invalid_request", &m)),
            };
            debug_assert_eq!(ir.input_f16.len(), state.input_len());
            let t0 = std::time::Instant::now();
            match state.run_on_stick(&ir.input_f16, ir.want_aux) {
                Ok(r) => {
                    state.requests_served.fetch_add(1, Ordering::Relaxed);
                    let out = r.output_f32();
                    let top: Vec<String> = run::top_k(&out, ir.top)
                        .into_iter()
                        .map(|(i, v)| format!("{{\"class\":{i},\"value\":{}}}", f32_json(v)))
                        .collect();
                    let device_ms = if r.layer_ms.is_empty() { "null".to_string() } else { format!("{:.3}", r.device_total_ms()) };
                    let layers = r.layer_ms.iter().map(|x| format!("{x:.3}")).collect::<Vec<_>>().join(",");
                    let full = if ir.full_output {
                        format!(",\"output\":[{}]", out.iter().map(|v| f32_json(*v)).collect::<Vec<_>>().join(","))
                    } else {
                        String::new()
                    };
                    (
                        200,
                        format!(
                            "{{\"model\":{},\"output_values\":{},\"top\":[{}],\"wall_ms\":{:.3},\"request_ms\":{:.3},\"device_ms\":{device_ms},\"layer_ms\":[{layers}],\"throttle_level\":{}{full}}}",
                            json_str(&state.model_path),
                            out.len(),
                            top.join(","),
                            r.wall.as_secs_f64() * 1e3,
                            t0.elapsed().as_secs_f64() * 1e3,
                            r.throttle_level
                        ),
                    )
                }
                Err(e) => (503, error_json("device_error", &format!("{e:#}"))),
            }
        }
        _ => (404, "{\"error\":\"Not Found\"}".to_string()),
    }
}

fn handle_lemonade_client(mut stream: TcpStream, state: Arc<LemonadeEngineState>) -> Result<()> {
    let req = match read_http_request(&mut stream) {
        Ok(r) => r,
        Err(e) => {
            let body = error_json("invalid_request", &format!("{e:#}"));
            let resp = format!(
                "HTTP/1.1 413 Payload Too Large\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(resp.as_bytes());
            return Ok(());
        }
    };
    let (code, resp_body) = lemonade_route(&state, &req);
    let reason = match code {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        503 => "Service Unavailable",
        _ => "OK",
    };
    let resp = format!(
        "HTTP/1.1 {code} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{resp_body}",
        resp_body.len()
    );
    stream.write_all(resp.as_bytes())?;
    stream.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::compile_gemm_f32;

    fn test_state() -> LemonadeEngineState {
        // 4x8 input, 8x4 weights: a valid graph blob without any hardware.
        let w = vec![0.1f32; 8 * 4];
        let blob = compile_gemm_f32(4, 8, 4, &w, None, false, 12).expect("compile");
        LemonadeEngineState::new("test.graph", "/nonexistent.mvcmd", "0", blob).expect("valid graph")
    }

    fn req(method: &str, path: &str, query: &str, body: &[u8]) -> HttpRequest {
        HttpRequest {
            method: method.into(),
            path: path.into(),
            query: query.into(),
            content_type: String::new(),
            body: body.to_vec(),
        }
    }

    #[test]
    fn chat_and_completions_are_501_not_fake_text() {
        let st = test_state();
        for p in ["/v1/chat/completions", "/v1/completions"] {
            let (code, body) = lemonade_route(&st, &req("POST", p, "", b"{}"));
            assert_eq!(code, 501, "{p}");
            assert!(body.contains("text_generation_unsupported"));
            assert!(!body.contains("choices"));
            assert!(!body.contains("usage"));
        }
        assert_eq!(st.requests_served.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn models_lists_a_vision_graph() {
        let st = test_state();
        let (code, body) = lemonade_route(&st, &req("GET", "/v1/models", "", b""));
        assert_eq!(code, 200);
        assert!(body.contains("\"type\":\"vision_graph\""));
        assert!(body.contains("\"generates_text\":false"));
        assert!(body.contains("/v1/infer"));
    }

    #[test]
    fn infer_rejects_bad_requests_before_touching_the_stick() {
        let st = test_state();
        let (code, _) = lemonade_route(&st, &req("POST", "/v1/infer", "", b""));
        assert_eq!(code, 400);
        let (code, b) = lemonade_route(&st, &req("POST", "/v1/infer", "format=f32", &[0u8; 12]));
        assert_eq!(code, 400);
        assert!(b.contains("expects"));
        assert_eq!(lemonade_route(&st, &req("GET", "/nope", "", b"")).0, 404);
    }

    #[test]
    fn parse_f32_and_f16_bodies() {
        let shape = (2, 2, 3);
        let f32s: Vec<u8> = (0..12).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let r = parse_infer_request("format=f32&top=3&output=1&aux=0", "", &f32s, shape).unwrap();
        assert_eq!(r.input_f16.len(), 12);
        assert_eq!(r.top, 3);
        assert!(r.full_output);
        assert!(!r.want_aux);
        let f16s = vec![0u8; 24];
        assert!(parse_infer_request("format=f16", "", &f16s, shape).is_ok());
        assert!(parse_infer_request("format=f16", "", &f16s[..23], shape).is_err());
    }

    #[test]
    fn parse_rejects_bad_parameters() {
        let shape = (2, 2, 3);
        let b = vec![0u8; 48];
        assert!(parse_infer_request("format=xml", "", &b, shape).is_err());
        assert!(parse_infer_request("format=f32&top=0", "", &b, shape).is_err());
        assert!(parse_infer_request("format=f32&top=abc", "", &b, shape).is_err());
        assert!(parse_infer_request("format=image&mean=1,2", "", &b, shape).is_err());
        assert!(parse_infer_request("format=image", "", b"junk", shape).is_err());
        assert!(parse_infer_request("format=image", "", &b, (2, 2, 1)).is_err());
    }

    #[test]
    fn parse_image_uses_graph_shape() {
        let mut png = Vec::new();
        image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(5, 7, image::Rgb([1, 2, 3])))
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        let r = parse_infer_request("", "image/png", &png, (4, 4, 3)).unwrap();
        assert_eq!(r.input_f16.len(), 4 * 4 * 3);
    }

    #[test]
    fn graph_validation_rejects_garbage() {
        assert!(LemonadeEngineState::new("x", "fw", "0", vec![0u8; 100]).is_err());
        let mut bad = vec![0u8; 264 + 227 + 64];
        bad[36] = 3;
        assert!(LemonadeEngineState::new("x", "fw", "0", bad).is_err());
    }
}

/// This host's name for /health, read from the kernel; "unknown" if unavailable.
fn node_name() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|h| h.trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string())
}
