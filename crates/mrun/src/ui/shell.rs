//! The native shell of `mrun ui` (macOS, feature `app`; 018 US5): one window with a WebKit view
//! that loads only the UI host's page, a menu bar item with a real menu (HIG: a menu, not a
//! popover), and the app menu so Copy, Paste and ⌘Q work. The page talks HTTP to its host; menu
//! items reach it as `mrun:menu` events. The only IPC is the page asking for what a title bar
//! and a right-click do natively: drag, zoom, and a context menu (019 contract).

use super::host::Host;
use super::store::{self, Frame};
use muda::{
    AboutMetadata, ContextMenu, Menu, MenuEvent, MenuId, MenuItem, PredefinedMenuItem, Submenu,
};
use serde_json::Value;
use std::time::{Duration, Instant};
use tao::dpi::{LogicalPosition, LogicalSize};
use tao::event::{Event, StartCause, WindowEvent};
use tao::event_loop::{ControlFlow, EventLoopBuilder};
use tao::platform::macos::{
    ActivationPolicy, EventLoopWindowTargetExtMacOS, WindowBuilderExtMacOS, WindowExtMacOS,
};
use tao::window::WindowBuilder;
use tray_icon::{Icon, TrayIconBuilder};
use wry::WebViewBuilder;

/// What reaches the event loop from other threads.
#[derive(Debug)]
enum UserEvent {
    Menu(MenuEvent),
    /// A status snapshot for the menu bar item (server state, model, memory, requests).
    Status(Snapshot),
    Show,
    PageLoaded,
    Quit,
    /// A message from the page (`window.ipc.postMessage`).
    Ipc(String),
}

#[derive(Clone, Debug, Default, PartialEq)]
struct Snapshot {
    running: bool,
    model: Option<String>,
    busy: usize,
    queued: u64,
    reading: bool,
    used: u64,
    budget: u64,
    menu_bar: bool,
    /// A rented GPU's spend so far, in cents (024 FR-004), and its price.
    spend_cents: Option<u64>,
    cost_per_hr: Option<String>,
    /// In a pool (026): the Macs it runs on as host, `Some(0)` as a member.
    pool_macs: Option<usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Glyph {
    Ready,
    Busy,
    Stopped,
    /// Reading a prompt: the bolt fills from the bottom, frame 0..3 (019 FR-014).
    Reading(u8),
}

/// Frames of the reading animation and their pace (≤ 4 fps).
const READING_FRAMES: u8 = 3;
const READING_STEP: Duration = Duration::from_millis(300);
/// The window's frame is saved this long after the last move or resize.
const SAVE_DELAY: Duration = Duration::from_millis(500);

impl Snapshot {
    /// The menu bar glyph; `frame` steps the reading animation, `None` when motion is reduced.
    fn glyph(&self, frame: Option<u8>) -> Glyph {
        match (self.running, self.model.is_some(), self.busy > 0) {
            (false, _, _) | (true, false, _) => Glyph::Stopped,
            (true, true, true) => match frame {
                Some(f) if self.reading => Glyph::Reading(f % READING_FRAMES),
                _ => Glyph::Busy,
            },
            (true, true, false) => Glyph::Ready,
        }
    }

    /// The text beside the glyph: what a rented GPU cost so far, a pool's Macs, nothing while
    /// local.
    fn title(&self) -> Option<String> {
        if let Some(c) = self.spend_cents {
            return Some(format!("${}.{:02}", c / 100, c % 100));
        }
        match self.pool_macs? {
            0 => Some("Pool".into()),
            n => Some(format!("{n} Macs")),
        }
    }

    fn state_line(&self) -> String {
        match (&self.model, self.running) {
            (_, false) => "mrun is not running".into(),
            (None, true) => "Loading…".into(),
            (Some(m), true) if self.reading => format!("{m} · reading a prompt"),
            (Some(m), true) if self.busy > 0 => format!("{m} · generating"),
            (Some(m), true) => format!("{m} · ready"),
        }
    }
}

fn gib(bytes: u64) -> String {
    format!("{:.1}", bytes as f64 / 1024f64.powi(3))
}

/// The page script that delivers a menu item (`App.svelte` listens for `mrun:menu`).
fn menu_script(item: &str, target: Option<&str>) -> String {
    let detail = serde_json::json!({"item": item, "target": target});
    format!("window.dispatchEvent(new CustomEvent('mrun:menu', {{detail: {detail}}}))")
}

/// A template glyph (black, alpha only) at 36 × 36 px (18 pt @2x): mrun's bolt, solid when
/// ready, with a dot when busy, hollow when stopped. Drawn here, so no image ships.
fn glyph_icon(glyph: Glyph) -> Option<Icon> {
    Icon::from_rgba(glyph_rgba(glyph), GLYPH_PX as u32, GLYPH_PX as u32).ok()
}

const GLYPH_PX: usize = 36;

/// The glyph's pixels: black, with coverage in alpha.
fn glyph_rgba(glyph: Glyph) -> Vec<u8> {
    const N: usize = GLYPH_PX;
    let bolt: [(f32, f32); 6] = [
        (20.5, 3.0),
        (8.0, 20.0),
        (16.5, 20.0),
        (15.0, 33.0),
        (28.0, 15.0),
        (19.5, 15.0),
    ];
    let inside = |x: f32, y: f32| {
        let mut c = false;
        let mut j = bolt.len() - 1;
        for i in 0..bolt.len() {
            let (xi, yi) = bolt[i];
            let (xj, yj) = bolt[j];
            if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
                c = !c;
            }
            j = i;
        }
        c
    };
    let mut rgba = vec![0u8; N * N * 4];
    for py in 0..N {
        for px in 0..N {
            // 4× supersampling for smooth edges.
            let mut cover = 0u32;
            for s in 0..16 {
                let x = px as f32 + (s % 4) as f32 / 4.0 + 0.125;
                let y = py as f32 + (s / 4) as f32 / 4.0 + 0.125;
                let solid = inside(x, y);
                // Hollow: the bolt's outline, 2.5 px wide.
                let outline = || {
                    solid
                        && [(-2.5f32, 0.0f32), (2.5, 0.0), (0.0, -2.5), (0.0, 2.5)]
                            .iter()
                            .any(|(dx, dy)| !inside(x + dx, y + dy))
                };
                let on = match glyph {
                    Glyph::Ready | Glyph::Busy => solid,
                    Glyph::Stopped => outline(),
                    // The outline, filled up to the frame's level (bolt spans y 3..33).
                    Glyph::Reading(f) => {
                        let level = 33.0 - 30.0 * f32::from(f + 1) / f32::from(READING_FRAMES);
                        outline() || (solid && y >= level)
                    }
                };
                let dot = glyph == Glyph::Busy && (x - 30.0).powi(2) + (y - 29.0).powi(2) <= 16.0;
                if on || dot {
                    cover += 1;
                }
            }
            let alpha = (cover * 255 / 16) as u8;
            let at = (py * N + px) * 4;
            rgba[at + 3] = alpha;
        }
    }
    rgba
}

/// Polls the host's view of the server once a second for the menu bar item.
async fn poll(host: Host, proxy: tao::event_loop::EventLoopProxy<UserEvent>) {
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(std::time::Duration::from_millis(800))
        .build()
        .unwrap_or_default();
    let url = format!("http://127.0.0.1:{}/mrun/status", host.server_port);
    let mut last = None;
    loop {
        let mut snap = Snapshot {
            menu_bar: host.store.prefs().menu_bar,
            ..Snapshot::default()
        };
        if let Ok(response) = client.get(&url).send().await
            && let Ok(status) = response.json::<Value>().await
        {
            snap.running = true;
            snap.model = status["model"]["id"].as_str().map(str::to_string);
            let requests = status["requests"].as_array().cloned().unwrap_or_default();
            snap.busy = requests.iter().filter(|r| r["phase"] != "queued").count();
            snap.reading = requests.iter().any(|r| r["phase"] == "prefill");
            snap.queued = status["queued"].as_u64().unwrap_or(0);
            let b = &status["budget"];
            let c = &status["cache"];
            snap.budget = b["budget"].as_u64().unwrap_or(0);
            snap.used = b["weights"].as_u64().unwrap_or(0)
                + b["scratch"].as_u64().unwrap_or(0)
                + c["buffer_bytes"].as_u64().unwrap_or(0)
                + c["checkpoint_bytes"].as_u64().unwrap_or(0);
            let pool = &status["pool"];
            snap.pool_macs = match pool["role"].as_str() {
                Some("member") => Some(0),
                Some("host") => {
                    let answering = pool["members"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter(|m| m["state"] != "lost")
                        .count();
                    (answering > 0).then_some(answering + 1)
                }
                _ => None,
            };
            let pod = &status["pod"];
            if pod.is_object() {
                snap.spend_cents =
                    Some((pod["spend_usd"].as_f64().unwrap_or(0.0) * 100.0).round() as u64);
                snap.cost_per_hr = pod["cost_per_hr"].as_f64().map(|p| format!("${p:.2}/h"));
            }
        }
        if last.as_ref() != Some(&snap) {
            last = Some(snap.clone());
            if proxy.send_event(UserEvent::Status(snap)).is_err() {
                return;
            }
        }
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }
}

/// A system setting read once at start (`defaults read`); `None` when unset.
fn system_default(domain: &str, key: &str) -> Option<String> {
    let out = std::process::Command::new("defaults")
        .args(["read", domain, key])
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// What a title-bar double-click does (System Settings › Desktop & Dock).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DoubleClick {
    Zoom,
    Minimize,
    Nothing,
}

fn double_click(setting: Option<&str>) -> DoubleClick {
    match setting {
        Some("Minimize") => DoubleClick::Minimize,
        Some("None") => DoubleClick::Nothing,
        _ => DoubleClick::Zoom,
    }
}

/// One entry of a context menu the page asked for (019 contract: ≤ 32 items, titles ≤ 80).
#[derive(Debug, PartialEq)]
enum Entry {
    Item {
        id: String,
        title: String,
        enabled: bool,
    },
    Separator,
}

/// The page's menu request: its entries and where to open it; `None` for a malformed request.
fn menu_request(request: &Value) -> Option<(Vec<Entry>, LogicalPosition<f64>)> {
    let mut entries = Vec::new();
    for item in request["items"].as_array()?.iter().take(32) {
        entries.push(if item.as_str() == Some("-") {
            Entry::Separator
        } else {
            Entry::Item {
                id: item["id"].as_str()?.to_string(),
                title: item["title"].as_str()?.chars().take(80).collect(),
                enabled: item["enabled"].as_bool().unwrap_or(true),
            }
        });
    }
    let at = LogicalPosition::new(request["x"].as_f64()?, request["y"].as_f64()?);
    Some((entries, at))
}

/// The native menu for a request (main thread only, as muda requires).
fn context_menu(entries: &[Entry]) -> Menu {
    let menu = Menu::new();
    for entry in entries {
        let _ = match entry {
            Entry::Separator => menu.append(&PredefinedMenuItem::separator()),
            Entry::Item { id, title, enabled } => menu.append(&MenuItem::with_id(
                format!("ctx:{id}"),
                title,
                *enabled,
                None,
            )),
        };
    }
    menu
}

/// The page script that answers a context menu (`native.svelte.ts` waits for it).
fn pick_script(id: Option<&str>) -> String {
    let detail = serde_json::json!({ "id": id });
    format!("window.dispatchEvent(new CustomEvent('mrun:menu-pick', {{detail: {detail}}}))")
}

/// The window's frame in logical points (outer position, inner size).
fn frame_of(window: &tao::window::Window) -> Option<Frame> {
    let scale = window.scale_factor();
    let at = window.outer_position().ok()?.to_logical::<f64>(scale);
    let size = window.inner_size().to_logical::<f64>(scale);
    Some(Frame {
        x: at.x,
        y: at.y,
        width: size.width,
        height: size.height,
    })
}

/// Runs the window and the menu bar item until Quit; the host keeps serving on `runtime`.
pub fn run(runtime: tokio::runtime::Runtime, host: Host, url: String, background: bool) -> u8 {
    let event_loop = EventLoopBuilder::<UserEvent>::with_user_event().build();
    let proxy = event_loop.create_proxy();
    {
        let proxy = proxy.clone();
        MenuEvent::set_event_handler(Some(move |event| {
            let _ = proxy.send_event(UserEvent::Menu(event));
        }));
    }
    runtime.spawn(poll(host.clone(), proxy.clone()));
    {
        let proxy = proxy.clone();
        let mut quit = host.quit.subscribe();
        runtime.spawn(async move {
            if quit.wait_for(|q| *q).await.is_ok() {
                let _ = proxy.send_event(UserEvent::Quit);
            }
        });
    }
    {
        let proxy = proxy.clone();
        let mut focus = host.focus.subscribe();
        runtime.spawn(async move {
            while focus.changed().await.is_ok() {
                let _ = proxy.send_event(UserEvent::Show);
            }
        });
    }

    // The app menu: About, Hide, Quit; Edit so the page's text fields copy and paste; Window.
    let app_menu = Menu::new();
    let about = PredefinedMenuItem::about(
        Some("About mrun"),
        Some(AboutMetadata {
            name: Some("mrun".into()),
            version: Some(env!("CARGO_PKG_VERSION").into()),
            ..AboutMetadata::default()
        }),
    );
    let settings = MenuItem::with_id("settings", "Settings…", true, "cmd+,".parse().ok());
    let quit = MenuItem::with_id("quit", "Quit mrun", true, "cmd+q".parse().ok());
    let mrun = Submenu::with_items(
        "mrun",
        true,
        &[
            &about,
            &PredefinedMenuItem::separator(),
            &settings,
            &PredefinedMenuItem::separator(),
            &PredefinedMenuItem::hide(None),
            &PredefinedMenuItem::hide_others(None),
            &PredefinedMenuItem::separator(),
            &quit,
        ],
    );
    let edit = Submenu::with_items(
        "Edit",
        true,
        &[
            &PredefinedMenuItem::undo(None),
            &PredefinedMenuItem::redo(None),
            &PredefinedMenuItem::separator(),
            &PredefinedMenuItem::cut(None),
            &PredefinedMenuItem::copy(None),
            &PredefinedMenuItem::paste(None),
            &PredefinedMenuItem::select_all(None),
        ],
    );
    let close = MenuItem::with_id("close", "Close Window", true, "cmd+w".parse().ok());
    let window_menu = Submenu::with_items(
        "Window",
        true,
        &[&PredefinedMenuItem::minimize(None), &close],
    );
    if let (Ok(mrun), Ok(edit), Ok(window_menu)) = (mrun, edit, window_menu) {
        let _ = app_menu.append_items(&[&mrun, &edit, &window_menu]);
    }

    // The menu bar item's menu (US5 AS2).
    let tray_menu = Menu::new();
    let state_item = MenuItem::new("mrun", false, None);
    let memory_item = MenuItem::new("", false, None);
    let requests_item = MenuItem::new("", false, None);
    let connect = Submenu::new("Connect", true);
    for (id, name) in [
        ("connect:claude-code", "Claude Code"),
        ("connect:codex", "Codex"),
        ("connect:opencode", "OpenCode"),
    ] {
        let _ = connect.append(&MenuItem::with_id(id, name, true, None));
    }
    let open = MenuItem::with_id("open", "Open mrun", true, None);
    let new_chat = MenuItem::with_id("new-chat", "New Chat", true, None);
    let tray_quit = MenuItem::with_id("quit", "Quit mrun", true, None);
    let tray_stop = MenuItem::with_id("stop", "Stop Serving", false, None);
    let _ = tray_menu.append_items(&[
        &state_item,
        &memory_item,
        &requests_item,
        &tray_stop,
        &PredefinedMenuItem::separator(),
        &connect,
        &PredefinedMenuItem::separator(),
        &open,
        &new_chat,
        &PredefinedMenuItem::separator(),
        &tray_quit,
    ]);

    // The unified title bar of Apple's own apps (019 FR-011): the content runs under it and the
    // traffic lights sit centred in the page's 52 pt toolbar row, over the sidebar.
    let monitors: Vec<Frame> = event_loop
        .available_monitors()
        .map(|m| {
            let scale = m.scale_factor();
            let at = m.position().to_logical::<f64>(scale);
            let size = m.size().to_logical::<f64>(scale);
            Frame {
                x: at.x,
                y: at.y,
                width: size.width,
                height: size.height,
            }
        })
        .collect();
    let mut builder = WindowBuilder::new()
        .with_title("mrun")
        .with_inner_size(LogicalSize::new(1100.0, 720.0))
        .with_min_inner_size(LogicalSize::new(800.0, 560.0))
        .with_visible(false)
        .with_titlebar_transparent(true)
        .with_fullsize_content_view(true)
        .with_title_hidden(true)
        .with_traffic_light_inset(LogicalPosition::new(20.0, 19.0));
    if let Some(frame) = store::restore_frame(host.store.prefs().window, &monitors) {
        builder = builder
            .with_position(LogicalPosition::new(frame.x, frame.y))
            .with_inner_size(LogicalSize::new(frame.width, frame.height));
    }
    let window = match builder.build(&event_loop) {
        Ok(window) => window,
        Err(e) => {
            eprintln!("error: cannot open the window: {e}");
            return 1;
        }
    };
    let origin = format!("http://127.0.0.1:{}/", host.ui_port);
    let webview = {
        let origin = origin.clone();
        let loaded = proxy.clone();
        let ipc = proxy.clone();
        let ipc_origin = origin.clone();
        WebViewBuilder::new()
            .with_url(&url)
            // Tells the page it runs in the window (traffic-light space, native menus).
            .with_initialization_script(
                "(function(){var s=function(){document.documentElement.dataset.shell='native'};\
                 if(document.documentElement)s();else document.addEventListener('readystatechange',s,{once:true})})()",
            )
            .with_ipc_handler(move |request| {
                // Only the host's own page may ask; the navigation handler keeps others out.
                if request.uri().to_string().starts_with(&ipc_origin) {
                    let _ = ipc.send_event(UserEvent::Ipc(request.into_body()));
                }
            })
            .with_navigation_handler(move |target| {
                if target.starts_with(&origin) {
                    return true;
                }
                // Links leave in the browser (only https; the page asks the host for others).
                if target.starts_with("https://") {
                    let _ = std::process::Command::new("open").arg(&target).status();
                }
                false
            })
            .with_on_page_load_handler(move |event, _| {
                if matches!(event, wry::PageLoadEvent::Finished) {
                    let _ = loaded.send_event(UserEvent::PageLoaded);
                }
            })
            .build(&window)
    };
    let webview = match webview {
        Ok(webview) => webview,
        Err(e) => {
            eprintln!("error: cannot show the page: {e}");
            return 1;
        }
    };

    let mut tray = None;
    let mut shown_once = false;
    let host_for_quit = host.clone();
    let reduce_motion =
        system_default("com.apple.universalaccess", "reduceMotion").as_deref() == Some("1");
    let on_double_click = double_click(system_default("-g", "AppleActionOnDoubleClick").as_deref());
    let mut snap = Snapshot::default();
    let mut frame: u8 = 0;
    let mut next_frame: Option<Instant> = None;
    let mut save_at: Option<Instant> = None;
    event_loop.run(move |event, target, control_flow| {
        let save = |window: &tao::window::Window| {
            if let Some(f) = frame_of(window) {
                let _ = host.store.set_window(f);
            }
        };
        let show = |target: &tao::event_loop::EventLoopWindowTarget<UserEvent>| {
            target.set_activation_policy_at_runtime(ActivationPolicy::Regular);
            window.set_visible(true);
            window.set_focus();
        };
        // Quitting asks first while a download runs (the page answers through /app/quit, FR-020),
        // and stops the server this window started.
        let quit = |target: &tao::event_loop::EventLoopWindowTarget<UserEvent>,
                    control_flow: &mut ControlFlow| {
            let pulling = host_for_quit.jobs.list().iter().any(|j| {
                j.kind == super::actions::Kind::Pull && j.state == super::actions::State::Running
            });
            if pulling {
                show(target);
                let _ = webview.evaluate_script(&menu_script("quit", None));
            } else {
                host_for_quit.stop_own_server();
                *control_flow = ControlFlow::Exit;
            }
        };
        match event {
            Event::NewEvents(StartCause::Init) => {
                app_menu.init_for_nsapp();
                if background {
                    target.set_activation_policy_at_runtime(ActivationPolicy::Accessory);
                }
                if host.store.prefs().menu_bar {
                    tray = TrayIconBuilder::new()
                        .with_menu(Box::new(tray_menu.clone()))
                        .with_tooltip("mrun")
                        .with_icon_templated(glyph_icon(Glyph::Stopped).expect("glyph"))
                        .build()
                        .ok();
                }
            }
            Event::UserEvent(UserEvent::PageLoaded) if !shown_once => {
                shown_once = true;
                if !background {
                    show(target);
                }
            }
            Event::UserEvent(UserEvent::Show) | Event::Reopen { .. } => show(target),
            Event::NewEvents(StartCause::ResumeTimeReached { .. }) => {
                let now = Instant::now();
                if save_at.is_some_and(|at| at <= now) {
                    save_at = None;
                    save(&window);
                }
                if next_frame.is_some_and(|at| at <= now) {
                    frame = frame.wrapping_add(1);
                    if let Some(t) = &tray {
                        let _ = t.set_icon_templated(glyph_icon(snap.glyph(Some(frame))));
                    }
                    next_frame = Some(now + READING_STEP);
                }
            }
            Event::WindowEvent {
                event: WindowEvent::Moved(_) | WindowEvent::Resized(_),
                ..
            } if shown_once => save_at = Some(Instant::now() + SAVE_DELAY),
            Event::UserEvent(UserEvent::Ipc(message)) => {
                let request: Value = serde_json::from_str(&message).unwrap_or_default();
                match request["op"].as_str() {
                    Some("drag") => {
                        let _ = window.drag_window();
                    }
                    Some("zoom") => match on_double_click {
                        DoubleClick::Zoom => window.set_maximized(!window.is_maximized()),
                        DoubleClick::Minimize => window.set_minimized(true),
                        DoubleClick::Nothing => {}
                    },
                    Some("menu") => match menu_request(&request) {
                        Some((entries, at)) => {
                            let menu = context_menu(&entries);
                            // SAFETY: the view is the live window's content view, on the main
                            // thread; muda only reads it while the menu tracks.
                            #[allow(unsafe_code)]
                            let picked = unsafe {
                                menu.show_context_menu_for_nsview(window.ns_view(), Some(at.into()))
                            };
                            // A pick arrives as a menu event; a dismissal is answered here.
                            if !picked {
                                let _ = webview.evaluate_script(&pick_script(None));
                            }
                        }
                        None => {
                            let _ = webview.evaluate_script(&pick_script(None));
                        }
                    },
                    // Anything else is ignored: the page has no other powers here.
                    _ => {}
                }
            }
            Event::UserEvent(UserEvent::Status(next)) => {
                snap = next;
                // The reading animation runs only while a prompt is read (FR-014).
                next_frame = (snap.reading && snap.busy > 0 && !reduce_motion)
                    .then(|| next_frame.unwrap_or_else(|| Instant::now() + READING_STEP));
                let glyph = snap.glyph((!reduce_motion).then_some(frame));
                state_item.set_text(snap.state_line());
                tray_stop.set_enabled(snap.running);
                memory_item.set_text(if let Some(price) = &snap.cost_per_hr {
                    format!("On RunPod · {price}")
                } else if snap.budget > 0 {
                    format!("{} of {} GB in use", gib(snap.used), gib(snap.budget))
                } else {
                    String::new()
                });
                requests_item.set_text(match (snap.busy, snap.queued) {
                    (0, 0) => "No requests".to_string(),
                    (b, 0) => format!("{b} running"),
                    (b, q) => format!("{b} running · {q} waiting"),
                });
                match (&tray, snap.menu_bar) {
                    (Some(t), visible) => {
                        let _ = t.set_visible(visible);
                        let _ = t.set_icon_templated(glyph_icon(glyph));
                        t.set_title(snap.title());
                    }
                    (None, true) => {
                        tray = TrayIconBuilder::new()
                            .with_menu(Box::new(tray_menu.clone()))
                            .with_tooltip("mrun")
                            .with_icon_templated(glyph_icon(glyph).expect("glyph"))
                            .with_title(snap.title().unwrap_or_default())
                            .build()
                            .ok();
                    }
                    (None, false) => {}
                }
            }
            Event::UserEvent(UserEvent::Menu(event)) => {
                let id: &MenuId = event.id();
                match id.as_ref() {
                    "open" => show(target),
                    "settings" => {
                        show(target);
                        let _ = webview.evaluate_script(&menu_script("settings", None));
                    }
                    "new-chat" => {
                        show(target);
                        let _ = webview.evaluate_script(&menu_script("new-chat", None));
                    }
                    "close" => window.set_visible(false),
                    "stop" => {
                        let _ = webview.evaluate_script(&menu_script("stop", None));
                    }
                    "quit" => quit(target, control_flow),
                    other => {
                        if let Some(id) = other.strip_prefix("ctx:") {
                            let _ = webview.evaluate_script(&pick_script(Some(id)));
                        } else if let Some(client) = other.strip_prefix("connect:") {
                            show(target);
                            let _ = webview.evaluate_script(&menu_script("connect", Some(client)));
                        }
                    }
                }
            }
            Event::UserEvent(UserEvent::Quit) => {
                host_for_quit.stop_own_server();
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                save_at = None;
                save(&window);
                if tray.is_some() || background {
                    // The menu bar item keeps mrun at hand; Quit there (or ⌘Q) ends it.
                    window.set_visible(false);
                    target.set_activation_policy_at_runtime(ActivationPolicy::Accessory);
                } else {
                    // Nothing would be left to bring the window back: closing it quits.
                    quit(target, control_flow);
                }
            }
            _ => {}
        }
        if *control_flow != ControlFlow::Exit {
            *control_flow = match [save_at, next_frame].into_iter().flatten().min() {
                Some(at) => ControlFlow::WaitUntil(at),
                None => ControlFlow::Wait,
            };
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn busy(reading: bool) -> Snapshot {
        Snapshot {
            running: true,
            model: Some("qwen3.6".into()),
            busy: 1,
            reading,
            ..Snapshot::default()
        }
    }

    #[test]
    fn the_menu_bar_shows_a_pods_spend_only() {
        assert_eq!(busy(false).title(), None);
        let remote = Snapshot {
            spend_cents: Some(60),
            ..busy(false)
        };
        assert_eq!(remote.title().as_deref(), Some("$0.60"));
        let long = Snapshot {
            spend_cents: Some(1352),
            ..busy(false)
        };
        assert_eq!(long.title().as_deref(), Some("$13.52"));
    }

    #[test]
    fn the_menu_bar_shows_a_pools_macs() {
        let host = Snapshot {
            pool_macs: Some(3),
            ..busy(false)
        };
        assert_eq!(host.title().as_deref(), Some("3 Macs"));
        let member = Snapshot {
            pool_macs: Some(0),
            ..busy(false)
        };
        assert_eq!(member.title().as_deref(), Some("Pool"));
    }

    #[test]
    fn the_bolt_fills_only_while_a_prompt_is_read() {
        assert_eq!(busy(true).glyph(Some(4)), Glyph::Reading(1));
        assert_eq!(busy(true).glyph(None), Glyph::Busy, "Reduce Motion: still");
        assert_eq!(busy(false).glyph(Some(4)), Glyph::Busy);
        assert_eq!(Snapshot::default().glyph(Some(1)), Glyph::Stopped);
        // Each frame draws more of the bolt than the one before, the last all of it.
        let ink = |g| {
            glyph_rgba(g)
                .iter()
                .skip(3)
                .step_by(4)
                .map(|&a| u64::from(a))
                .sum::<u64>()
        };
        let frames: Vec<u64> = (0..READING_FRAMES)
            .map(|f| ink(Glyph::Reading(f)))
            .collect();
        assert!(frames.windows(2).all(|w| w[0] < w[1]), "{frames:?}");
        assert_eq!(frames[2], ink(Glyph::Ready));
        assert!(frames[0] > ink(Glyph::Stopped));
    }

    #[test]
    fn a_double_click_does_what_the_system_setting_says() {
        assert_eq!(double_click(None), DoubleClick::Zoom);
        assert_eq!(double_click(Some("Maximize")), DoubleClick::Zoom);
        assert_eq!(double_click(Some("Minimize")), DoubleClick::Minimize);
        assert_eq!(double_click(Some("None")), DoubleClick::Nothing);
    }

    #[test]
    fn a_context_menu_request_is_bounded_and_malformed_ones_are_refused() {
        let ok = serde_json::json!({
            "op": "menu", "x": 10.0, "y": 20.0,
            "items": [{"id": "use", "title": "Use"}, "-", {"id": "remove", "title": "Remove…", "enabled": false}],
        });
        let (entries, at) = menu_request(&ok).expect("menu");
        assert_eq!((at.x, at.y), (10.0, 20.0));
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[1], Entry::Separator);
        assert_eq!(
            entries[2],
            Entry::Item {
                id: "remove".into(),
                title: "Remove…".into(),
                enabled: false
            }
        );
        let long: Vec<Value> = (0..40)
            .map(|i| serde_json::json!({"id": i.to_string(), "title": "x".repeat(200)}))
            .collect();
        let (entries, _) =
            menu_request(&serde_json::json!({"items": long, "x": 0, "y": 0})).expect("menu");
        assert_eq!(entries.len(), 32);
        assert!(matches!(&entries[0], Entry::Item { title, .. } if title.len() == 80));
        let no_id = serde_json::json!({"items": [{"title": "no id"}], "x": 0, "y": 0});
        assert!(menu_request(&no_id).is_none());
        assert!(
            menu_request(&serde_json::json!({"items": []})).is_none(),
            "no position"
        );
        assert_eq!(
            pick_script(None),
            "window.dispatchEvent(new CustomEvent('mrun:menu-pick', {detail: {\"id\":null}}))"
        );
    }
}
