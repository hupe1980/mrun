//! `mrun pool` (026 contracts/cli.md): host a pool, find pools nearby, join one with the code
//! both Macs show, admit, leave and remove. Every command talks to the local server.

use crate::cli::server;
use crate::errors::{CliError, Code};
use serde_json::{Value, json};
use std::time::Duration;

#[derive(Debug, clap::Subcommand)]
pub enum PoolCommand {
    /// Host a pool on this Mac: announce it on the local network and accept joins.
    Start(StartArgs),
    /// This Mac's pool with its members, and the pools nearby.
    Ls(CommonArgs),
    /// Ask to join a pool (its name, host or `address:port`); prints the code to compare.
    Join(JoinArgs),
    /// Confirm the join this Mac asked for with the code the host shows.
    Confirm(ConfirmArgs),
    /// Admit a Mac waiting to join this Mac's pool (asks unless --yes).
    Admit(AdmitArgs),
    /// Decline a Mac waiting to join.
    Decline(MacArgs),
    /// Leave the pool (a member), or end it for everyone (the host).
    Leave(LeaveArgs),
    /// Remove a member from this Mac's pool; both Macs forget the membership.
    Remove(MacArgs),
    /// The plan for a model on the pool's Macs: layers, memory, context and speed per Mac.
    Plan(ModelArgs),
    /// Serve a model across the pool (each Mac loads its layers; agents use this Mac's server).
    Serve(ModelArgs),
    /// Rejoin the pool this Mac was in (after it was lost, or its host restarted), without a code.
    Rejoin(CommonArgs),
    /// Host: serve the pool's model again on the Macs answering now.
    Replan(CommonArgs),
}

#[derive(Debug, clap::Args)]
pub struct ModelArgs {
    /// The model: an alias, `org/name` or a path.
    pub model: String,
    #[command(flatten)]
    pub common: CommonArgs,
}

#[derive(Debug, clap::Args)]
pub struct CommonArgs {
    /// Print JSON.
    #[arg(long)]
    pub json: bool,
    /// Port of the server.
    #[arg(long, env = "MRUN_PORT", default_value_t = server::default_port())]
    pub port: u16,
}

#[derive(Debug, clap::Args)]
pub struct StartArgs {
    /// The pool's name (default: "<your name>'s pool").
    #[arg(long)]
    pub name: Option<String>,
    /// Do not encrypt the model's traffic between the Macs (still authenticated; prompts and
    /// replies stay encrypted). Members are told before they join.
    #[arg(long)]
    pub unencrypted: bool,
    #[command(flatten)]
    pub common: CommonArgs,
}

#[derive(Debug, clap::Args)]
pub struct JoinArgs {
    /// The pool: its name, its host's name, or `address:port`.
    pub pool: String,
    #[command(flatten)]
    pub common: CommonArgs,
}

#[derive(Debug, clap::Args)]
pub struct ConfirmArgs {
    /// The six digits the host shows.
    pub code: String,
    #[command(flatten)]
    pub common: CommonArgs,
}

#[derive(Debug, clap::Args)]
pub struct MacArgs {
    /// The Mac: its name or id.
    pub mac: String,
    #[command(flatten)]
    pub common: CommonArgs,
}

#[derive(Debug, clap::Args)]
pub struct AdmitArgs {
    /// The Mac waiting to join: its name or id (the only one when omitted).
    pub mac: Option<String>,
    /// Admit without asking.
    #[arg(long, short = 'y')]
    pub yes: bool,
    #[command(flatten)]
    pub common: CommonArgs,
}

#[derive(Debug, clap::Args)]
pub struct LeaveArgs {
    /// Do not ask (the host ends the pool for everyone).
    #[arg(long, short = 'y')]
    pub yes: bool,
    #[command(flatten)]
    pub common: CommonArgs,
}

pub fn run(command: PoolCommand) -> u8 {
    crate::cli::exit(match command {
        PoolCommand::Start(args) => start(&args),
        PoolCommand::Ls(args) => ls(&args),
        PoolCommand::Join(args) => join(&args),
        PoolCommand::Confirm(args) => confirm(&args),
        PoolCommand::Admit(args) => admit(&args),
        PoolCommand::Decline(args) => decide(&args.common, &args.mac, false),
        PoolCommand::Leave(args) => leave(&args),
        PoolCommand::Remove(args) => remove(&args),
        PoolCommand::Plan(args) => plan(&args),
        PoolCommand::Serve(args) => serve(&args),
        PoolCommand::Rejoin(args) => rejoin(&args),
        PoolCommand::Replan(args) => replan(&args),
    })
}

/// The server's answer, or its pool error as a CLI error.
fn call(
    port: u16,
    method: &str,
    path: &str,
    body: Option<Value>,
    wait: Duration,
) -> Result<Value, CliError> {
    let (status, body) = server::request_json(port, method, path, body.as_ref(), wait)?;
    if (200..300).contains(&status) {
        return Ok(body);
    }
    let error = &body["error"];
    let code = match error["code"].as_str() {
        Some("E060") => Code::PoolMemberLost,
        Some("E061") => Code::PoolRefused,
        Some("E062") => Code::PoolDoesNotFit,
        Some("E063") => Code::PoolVersion,
        _ if status == 404 => Code::ServerUnreachable,
        _ => Code::Internal,
    };
    let message = error["message"].as_str().unwrap_or("the server refused");
    Err(CliError::new(code, message.to_string()))
}

const SHORT: Duration = Duration::from_secs(10);
/// A join waits for the host's answer (2 minutes) and a little more.
const JOIN: Duration = Duration::from_secs(130);

fn print(common: &CommonArgs, value: &Value, text: impl FnOnce(&Value) -> String) {
    if common.json {
        println!(
            "{}",
            serde_json::to_string_pretty(value).unwrap_or_default()
        );
    } else {
        print!("{}", text(value));
    }
}

fn start(args: &StartArgs) -> Result<(), CliError> {
    let body = call(
        args.common.port,
        "POST",
        "/mrun/pool/start",
        Some(json!({"name": args.name, "unencrypted": args.unencrypted})),
        SHORT,
    )?;
    print(&args.common, &body, |b| {
        format!(
            "{} is open at {}. Other Macs join with: mrun pool join \"{}\"\n",
            b["pool"]["name"].as_str().unwrap_or("The pool"),
            b["pool"]["address"].as_str().unwrap_or("?"),
            b["pool"]["name"].as_str().unwrap_or("?"),
        )
    });
    Ok(())
}

fn gib(bytes: &Value) -> String {
    format!("{} GB", bytes.as_u64().unwrap_or(0) >> 30)
}

/// `mrun pool ls` as text (contracts/cli.md).
pub fn render(body: &Value) -> String {
    let mut out = String::new();
    let pool = &body["pool"];
    match pool["role"].as_str() {
        Some("host") => {
            out.push_str(&format!(
                "{} (this Mac hosts)\n",
                pool["name"].as_str().unwrap_or("?")
            ));
            out.push_str(&format!(
                "  1  {:<22} {:<16} host\n",
                body["this"]["name"].as_str().unwrap_or("This Mac"),
                format!(
                    "{} {}",
                    chip(&body["this"]["chip"]),
                    gib(&body["this"]["memory_bytes"])
                ),
            ));
            for (i, m) in pool["members"].as_array().into_iter().flatten().enumerate() {
                out.push_str(&format!(
                    "  {}  {:<22} {:<16} {}\n",
                    i + 2,
                    m["name"].as_str().unwrap_or("?"),
                    format!("{} {}", chip(&m["chip"]), gib(&m["memory_bytes"])),
                    m["power"].as_str().unwrap_or(""),
                ));
            }
            for w in pool["waiting"].as_array().into_iter().flatten() {
                out.push_str(&format!(
                    "  waiting to join: {} · code {} · mrun pool admit \"{}\"\n",
                    w["name"].as_str().unwrap_or("?"),
                    crate::pool::join::spaced(w["code"].as_str().unwrap_or("")),
                    w["name"].as_str().unwrap_or("?"),
                ));
            }
        }
        Some("member") => out.push_str(&format!(
            "{} (hosted by {})\n",
            pool["name"].as_str().unwrap_or("?"),
            pool["host"]["name"].as_str().unwrap_or("?"),
        )),
        _ => out.push_str("This Mac is in no pool.\n"),
    }
    let nearby: Vec<&Value> = body["nearby"].as_array().into_iter().flatten().collect();
    if !nearby.is_empty() {
        out.push_str("Nearby\n");
        for p in nearby {
            out.push_str(&format!(
                "  {:<22} {} {}  {}\n",
                p["name"].as_str().unwrap_or("?"),
                p["members"].as_u64().unwrap_or(1),
                if p["members"] == 1 { "Mac " } else { "Macs" },
                p["model"].as_str().unwrap_or(""),
            ));
        }
    }
    out
}

/// "M4 Pro" from "Apple M4 Pro".
fn chip(chip: &Value) -> String {
    chip.as_str()
        .unwrap_or("")
        .trim_start_matches("Apple ")
        .to_string()
}

fn ls(args: &CommonArgs) -> Result<(), CliError> {
    let body = call(args.port, "GET", "/mrun/pool?browse=true", None, SHORT)?;
    print(args, &body, render);
    Ok(())
}

/// The `address:port` of `pool`: given as one, or found by pool or host name nearby.
fn resolve(port: u16, pool: &str) -> Result<String, CliError> {
    if pool.parse::<std::net::SocketAddr>().is_ok() {
        return Ok(pool.to_string());
    }
    let body = call(port, "GET", "/mrun/pool?browse=true", None, SHORT)?;
    let wanted = pool.to_lowercase();
    body["nearby"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|p| {
            [&p["name"], &p["host"], &p["host_id"]]
                .iter()
                .any(|v| v.as_str().is_some_and(|s| s.to_lowercase() == wanted))
        })
        .and_then(|p| p["addresses"][0].as_str().map(str::to_string))
        .ok_or_else(|| {
            CliError::new(
                Code::PoolRefused,
                format!("no pool named \"{pool}\" nearby"),
            )
            .hint("'mrun pool ls' lists the pools on this network")
        })
}

fn join(args: &JoinArgs) -> Result<(), CliError> {
    let port = args.common.port;
    let address = resolve(port, &args.pool)?;
    let asked = call(
        port,
        "POST",
        "/mrun/pool/join",
        Some(json!({"address": address})),
        SHORT,
    )?;
    let code = asked["code"].as_str().unwrap_or("").to_string();
    let host = asked["host"].as_str().unwrap_or("the host");
    // Without a terminal (the window), the join waits for `mrun pool confirm <code>`.
    if args.common.json || !std::io::IsTerminal::is_terminal(&std::io::stdin()) {
        print(&args.common, &asked, |_| {
            format!(
                "Joining {}: {host} shows {}. Confirm with: mrun pool confirm {code}\n",
                asked["pool"].as_str().unwrap_or("the pool"),
                crate::pool::join::spaced(&code)
            )
        });
        return Ok(());
    }
    eprintln!(
        "Joining {} lends this Mac's memory to everyone in it; your prompts are read on {host}.{}",
        asked["pool"].as_str().unwrap_or("the pool"),
        if asked["unencrypted"] == true {
            " Its model traffic between the Macs is not encrypted (prompts and replies are)."
        } else {
            ""
        },
    );
    let yes = crate::cli::confirm(
        &format!("Does {host} show {}?", crate::pool::join::spaced(&code)),
        false,
    )?;
    eprintln!("Waiting for {host} to admit this Mac…");
    let joined = call(
        port,
        "POST",
        "/mrun/pool/join/confirm",
        Some(json!({"yes": yes, "code": yes.then_some(&code)})),
        JOIN,
    )?;
    if joined["joined"] == true {
        println!("Joined {}.", joined["pool"].as_str().unwrap_or("the pool"));
    } else {
        println!("Not joined.");
    }
    Ok(())
}

fn confirm(args: &ConfirmArgs) -> Result<(), CliError> {
    let joined = call(
        args.common.port,
        "POST",
        "/mrun/pool/join/confirm",
        Some(json!({"yes": true, "code": args.code})),
        JOIN,
    )?;
    print(&args.common, &joined, |j| {
        format!("Joined {}.\n", j["pool"].as_str().unwrap_or("the pool"))
    });
    Ok(())
}

/// The waiting join matching `mac` (its name or id), or the only one.
fn waiting(port: u16, mac: Option<&str>) -> Result<Value, CliError> {
    let body = call(port, "GET", "/mrun/pool", None, SHORT)?;
    let all: Vec<Value> = body["pool"]["waiting"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let found: Vec<&Value> = match mac {
        Some(mac) => all
            .iter()
            .filter(|w| {
                w["mac_id"] == mac
                    || w["name"]
                        .as_str()
                        .is_some_and(|n| n.eq_ignore_ascii_case(mac))
            })
            .collect(),
        None => all.iter().collect(),
    };
    match found.as_slice() {
        [one] => Ok((*one).clone()),
        [] => Err(CliError::new(
            Code::PoolRefused,
            "no Mac is waiting to join",
        )),
        _ => Err(CliError::new(
            Code::InvalidArguments,
            "several Macs are waiting; name one",
        )),
    }
}

fn admit(args: &AdmitArgs) -> Result<(), CliError> {
    let w = waiting(args.common.port, args.mac.as_deref())?;
    let yes = crate::cli::confirm(
        &format!(
            "Admit {} ({}, {})? It shows {}.",
            w["name"].as_str().unwrap_or("?"),
            chip(&w["chip"]),
            gib(&w["memory_bytes"]),
            crate::pool::join::spaced(w["code"].as_str().unwrap_or("")),
        ),
        args.yes,
    )?;
    decide(&args.common, w["mac_id"].as_str().unwrap_or(""), yes)
}

fn decide(common: &CommonArgs, mac: &str, yes: bool) -> Result<(), CliError> {
    let w = waiting(common.port, Some(mac))?;
    let body = call(
        common.port,
        "POST",
        "/mrun/pool/admit",
        Some(json!({"mac_id": w["mac_id"], "yes": yes})),
        SHORT,
    )?;
    print(common, &body, |_| {
        format!(
            "{} {}.\n",
            if yes { "Admitted" } else { "Declined" },
            w["name"].as_str().unwrap_or("?")
        )
    });
    Ok(())
}

fn leave(args: &LeaveArgs) -> Result<(), CliError> {
    let port = args.common.port;
    let status = call(port, "GET", "/mrun/pool", None, SHORT)?;
    if status["pool"]["role"] == "host" {
        let members = status["pool"]["members"].as_array().map_or(0, Vec::len);
        if !crate::cli::confirm(
            &format!("End the pool for everyone ({members} other Macs)?"),
            args.yes,
        )? {
            return Ok(());
        }
    }
    let body = call(port, "POST", "/mrun/pool/leave", None, SHORT)?;
    print(&args.common, &body, |b| match (&b["left"], &b["ended"]) {
        (Value::String(pool), _) => format!("Left {pool}.\n"),
        (_, Value::String(pool)) => format!("Ended {pool}.\n"),
        _ => String::new(),
    });
    Ok(())
}

fn remove(args: &MacArgs) -> Result<(), CliError> {
    let port = args.common.port;
    let status = call(port, "GET", "/mrun/pool", None, SHORT)?;
    let wanted = args.mac.to_lowercase();
    let member = status["pool"]["members"]
        .as_array()
        .into_iter()
        .flatten()
        .chain(status["memberships"].as_array().into_iter().flatten())
        .find(|m| {
            m["mac_id"] == args.mac.as_str()
                || m["name"]
                    .as_str()
                    .is_some_and(|n| n.to_lowercase() == wanted)
        })
        .cloned()
        .ok_or_else(|| {
            CliError::new(
                Code::PoolRefused,
                format!("no member named \"{}\"", args.mac),
            )
        })?;
    let body = call(
        port,
        "POST",
        "/mrun/pool/remove",
        Some(json!({"mac_id": member["mac_id"]})),
        SHORT,
    )?;
    print(&args.common, &body, |_| {
        format!("Removed {}.\n", member["name"].as_str().unwrap_or("?"))
    });
    Ok(())
}

/// A plan as text: one line per Mac, then context, speed and the pace-setter.
pub fn render_plan(plan: &Value) -> String {
    let gb = |v: &Value| v.as_u64().unwrap_or(0) as f64 / 1e9;
    let mut out = format!("{}\n", plan["model"].as_str().unwrap_or("?"));
    for s in plan["stages"].as_array().into_iter().flatten() {
        out.push_str(&format!(
            "  {:<22} layers {:>3}–{:<3} {:>5.1} of {:>5.1} GB{}\n",
            s["name"].as_str().unwrap_or("?"),
            s["first"].as_u64().unwrap_or(0) + 1,
            s["end"].as_u64().unwrap_or(0),
            gb(&s["weights_bytes"]) + gb(&s["state_bytes"]) + gb(&s["scratch_bytes"]),
            gb(&s["budget"]),
            if s["fits"] == false {
                "  does not fit"
            } else {
                ""
            },
        ));
    }
    if plan["fits"] == true {
        out.push_str(&format!(
            "  context {} tokens · about {:.0}–{:.0} tok/s",
            plan["context_tokens"]
                .as_u64()
                .unwrap_or(0)
                .to_string()
                .as_str(),
            plan["speed"]["low"].as_f64().unwrap_or(0.0),
            plan["speed"]["high"].as_f64().unwrap_or(0.0),
        ));
        if let Some(name) = plan["pace"]["name"].as_str() {
            let what = if plan["pace"]["kind"] == "link" {
                "'s link"
            } else {
                ""
            };
            out.push_str(&format!(" · {name}{what} sets the pace"));
        }
        out.push('\n');
        if plan["numerics_match"] == false {
            out.push_str("  the Macs have different GPUs: replies may differ in the last digits from one Mac's\n");
        }
    } else if let Some(why) = plan["refusal"].as_str() {
        out.push_str(&format!("  {why}\n"));
    }
    out
}

fn plan(args: &ModelArgs) -> Result<(), CliError> {
    let path = format!("/mrun/pool/plan?model={}", urlencode(&args.model));
    let body = call(
        args.common.port,
        "GET",
        &path,
        None,
        Duration::from_secs(60),
    )?;
    print(&args.common, &body, render_plan);
    if body["fits"] == false {
        return Err(CliError::new(
            Code::PoolDoesNotFit,
            body["refusal"]
                .as_str()
                .unwrap_or("the model does not fit the pool")
                .to_string(),
        ));
    }
    Ok(())
}

fn serve(args: &ModelArgs) -> Result<(), CliError> {
    let body = call(
        args.common.port,
        "POST",
        "/mrun/pool/serve",
        Some(json!({"model": args.model})),
        Duration::from_secs(60),
    )?;
    print(&args.common, &body, |b| {
        format!(
            "{}Loading {} on every Mac; 'mrun ps' shows when it serves.\n",
            render_plan(&b["plan"]),
            args.model
        )
    });
    Ok(())
}

fn rejoin(args: &CommonArgs) -> Result<(), CliError> {
    let body = call(args.port, "POST", "/mrun/pool/rejoin", None, SHORT)?;
    print(args, &body, |b| {
        format!("Back in {}.\n", b["pool"].as_str().unwrap_or("the pool"))
    });
    Ok(())
}

fn replan(args: &CommonArgs) -> Result<(), CliError> {
    let body = call(
        args.port,
        "POST",
        "/mrun/pool/replan",
        None,
        Duration::from_secs(60),
    )?;
    print(args, &body, |b| {
        format!(
            "{}Loading {} on the Macs answering now; 'mrun ps' shows when it serves.\n",
            render_plan(&b["plan"]),
            b["serving"].as_str().unwrap_or("the model")
        )
    });
    Ok(())
}

fn urlencode(text: &str) -> String {
    text.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (b as char).to_string()
            }
            _ => format!("%{b:02X}"),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plans_print_a_line_per_mac() {
        let plan = json!({"model": "qwen3-235b", "fits": true, "context_tokens": 65536,
            "speed": {"low": 11.2, "high": 14.8}, "numerics_match": true,
            "pace": {"kind": "link", "name": "Dana's MacBook Air"},
            "stages": [
                {"name": "Ben's MacBook Pro", "first": 0, "end": 24, "weights_bytes": 31_000_000_000u64,
                 "state_bytes": 2_000_000_000u64, "scratch_bytes": 1_000_000_000u64, "budget": 40_000_000_000u64, "fits": true},
                {"name": "Dana's MacBook Air", "first": 24, "end": 94, "weights_bytes": 1u64,
                 "state_bytes": 0, "scratch_bytes": 0, "budget": 20_000_000_000u64, "fits": true}]});
        assert_eq!(
            render_plan(&plan),
            "qwen3-235b\n\
             \x20 Ben's MacBook Pro      layers   1–24   34.0 of  40.0 GB\n\
             \x20 Dana's MacBook Air     layers  25–94    0.0 of  20.0 GB\n\
             \x20 context 65536 tokens · about 11–15 tok/s · Dana's MacBook Air's link sets the pace\n"
        );
        assert_eq!(urlencode("org/name@x y"), "org%2Fname%40x%20y");
    }

    #[test]
    fn ls_prints_the_pool_and_the_ones_nearby() {
        let body = json!({
            "this": {"name": "Ben's MacBook Pro", "chip": "Apple M4 Max", "memory_bytes": 48u64 << 30},
            "pool": {"role": "host", "name": "Ben's pool",
                "members": [{"name": "Anna's MacBook Pro", "chip": "Apple M4 Pro",
                    "memory_bytes": 48u64 << 30, "power": "battery"}],
                "waiting": [{"name": "Dana's MacBook Air", "code": "482913"}]},
            "nearby": [{"name": "Chris's pool", "members": 2, "model": "qwen3.6"}],
        });
        assert_eq!(
            render(&body),
            "Ben's pool (this Mac hosts)\n\
             \x20 1  Ben's MacBook Pro      M4 Max 48 GB     host\n\
             \x20 2  Anna's MacBook Pro     M4 Pro 48 GB     battery\n\
             \x20 waiting to join: Dana's MacBook Air · code 482 913 · mrun pool admit \"Dana's MacBook Air\"\n\
             Nearby\n\
             \x20 Chris's pool           2 Macs  qwen3.6\n"
        );
        assert_eq!(render(&json!({"pool": null})), "This Mac is in no pool.\n");
    }
}
