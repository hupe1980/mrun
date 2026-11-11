//! `mrun ui`: the window (spec 018, D067). A loopback host serves the embedded page, runs the
//! CLI's own commands for every action and proxies status and chat to the server; the native
//! shell (feature `app`, macOS) shows it in a window with a menu bar item, `--browser` in the
//! default browser.

pub mod actions;
pub mod host;
#[cfg(all(feature = "app", target_os = "macos"))]
mod shell;
pub mod store;
mod tools;

use crate::cli::server;
use std::io::Read;

#[derive(Debug, clap::Args)]
pub struct UiArgs {
    /// Open the window in the default browser instead of mrun's own window.
    #[arg(long)]
    pub browser: bool,
    /// Only the menu bar item, no window (used by "Open mrun at login").
    #[arg(long, conflicts_with = "browser")]
    pub background: bool,
    /// Port of the window's own loopback host (the next free one when taken).
    #[arg(long, env = "MRUN_UI_PORT", default_value_t = 8485)]
    pub port: u16,
    /// Port of the mrun server the window shows and starts.
    #[arg(long, env = "MRUN_PORT", default_value_t = server::default_port())]
    pub server_port: u16,
}

/// 32 random bytes as hex, from the OS (no RNG crate).
fn token() -> std::io::Result<String> {
    let mut bytes = [0u8; 32];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}

/// `http://127.0.0.1:<port>/#t=<token>`: the fragment is never sent to any server.
pub fn url(port: u16, token: &str) -> String {
    format!("http://127.0.0.1:{port}/#t={token}")
}

/// Opens `url` in the default browser.
fn open_in_browser(url: &str) {
    let opener = if cfg!(target_os = "macos") {
        "open"
    } else {
        "xdg-open"
    };
    if std::process::Command::new(opener)
        .arg(url)
        .status()
        .is_err()
    {
        eprintln!("open {url} in a browser");
    }
}

pub fn run(args: UiArgs) -> u8 {
    let home = crate::cli::mrun_home();
    // A second `mrun ui` brings the running one forward (Edge Cases).
    if let Some(instance) = store::running_instance(&home) {
        if args.browser || !cfg!(all(feature = "app", target_os = "macos")) {
            open_in_browser(&url(instance.port, &instance.token));
        } else {
            focus_running(instance.port, &instance.token);
        }
        return 0;
    }
    let token = match token() {
        Ok(token) => token,
        Err(e) => {
            eprintln!("error: cannot create the window's key: {e}");
            return 1;
        }
    };
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(e) => {
            eprintln!("error: cannot start the async runtime: {e}");
            return 1;
        }
    };
    let listener = match runtime.block_on(bind(args.port)) {
        Ok(listener) => listener,
        Err(e) => {
            eprintln!("error: cannot listen on 127.0.0.1:{}: {e}", args.port);
            return 1;
        }
    };
    let port = listener.local_addr().map_or(args.port, |a| a.port());
    let host = host::Host::new(
        token.clone(),
        port,
        args.server_port,
        store::Store::open(&home),
    );
    let _ = store::write_instance(
        &home,
        &store::Instance {
            pid: std::process::id(),
            port,
            token: token.clone(),
        },
    );
    let app = host::router(host.clone());
    let serve = async move {
        let _ = axum::serve(listener, app).await;
    };
    let link = url(port, &token);
    let code = if args.browser || !cfg!(all(feature = "app", target_os = "macos")) {
        crate::cli::info(&format!(
            "mrun's window: http://127.0.0.1:{port}/ (opening in your browser; Ctrl-C quits)"
        ));
        open_in_browser(&link);
        runtime.block_on(async {
            tokio::select! {
                () = serve => {}
                () = crate::cli::serve::terminate() => {}
            }
        });
        0
    } else {
        run_shell(runtime, serve, host, link, args.background)
    };
    let _ = std::fs::remove_file(store::instance_file(&home));
    code
}

#[cfg(all(feature = "app", target_os = "macos"))]
fn run_shell(
    runtime: tokio::runtime::Runtime,
    serve: impl std::future::Future<Output = ()> + Send + 'static,
    host: host::Host,
    link: String,
    background: bool,
) -> u8 {
    runtime.spawn(serve);
    shell::run(runtime, host, link, background)
}

#[cfg(not(all(feature = "app", target_os = "macos")))]
fn run_shell(
    _runtime: tokio::runtime::Runtime,
    _serve: impl std::future::Future<Output = ()> + Send + 'static,
    _host: host::Host,
    _link: String,
    _background: bool,
) -> u8 {
    unreachable!("window mode needs the `app` feature on macOS")
}

/// Asks the running window to come forward (`POST /app/focus`).
fn focus_running(port: u16, token: &str) {
    use std::io::Write;
    if let Ok(mut stream) = std::net::TcpStream::connect(("127.0.0.1", port)) {
        let _ = write!(
            stream,
            "POST /app/focus HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nAuthorization: Bearer {token}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        );
        let _ = stream.read(&mut [0u8; 256]);
    }
}

/// The window's port, or the next free one of the 20 after it (another program may hold it).
async fn bind(port: u16) -> std::io::Result<tokio::net::TcpListener> {
    let mut last = None;
    for candidate in port..port.saturating_add(20) {
        match tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, candidate)).await {
            Ok(listener) => return Ok(listener),
            Err(e) => last = Some(e),
        }
    }
    Err(last.unwrap_or_else(|| std::io::Error::other("no free port")))
}
