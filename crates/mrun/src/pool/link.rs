//! The link to another Mac of the pool (026 R6): its kind (Thunderbolt, Ethernet, Wi-Fi,
//! loopback) from the interface the route to it uses, and its delay and throughput measured over
//! the pinned pool connection.

use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Thunderbolt,
    Ethernet,
    #[serde(rename = "wifi")]
    WiFi,
    Loopback,
}

impl Kind {
    pub fn word(self) -> &'static str {
        match self {
            Self::Thunderbolt => "Thunderbolt",
            Self::Ethernet => "Ethernet",
            Self::WiFi => "Wi-Fi",
            Self::Loopback => "this Mac",
        }
    }
}

/// A measured link.
#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct Link {
    pub kind: Kind,
    pub rtt_ms_p50: f64,
    pub rtt_ms_p90: f64,
    pub mbps: f64,
}

/// The interface `route -n get <ip>` names.
fn route_interface(route: &str) -> Option<&str> {
    route
        .lines()
        .find_map(|l| l.trim().strip_prefix("interface:"))
        .map(str::trim)
}

/// The kind of `interface`, given `networksetup -listallhardwareports`.
fn kind_of(interface: &str, ports: &str) -> Kind {
    if interface.starts_with("lo") {
        return Kind::Loopback;
    }
    if interface.starts_with("bridge") {
        return Kind::Thunderbolt;
    }
    let mut port = "";
    for line in ports.lines() {
        if let Some(name) = line.strip_prefix("Hardware Port: ") {
            port = name.trim();
        } else if line.strip_prefix("Device: ").map(str::trim) == Some(interface) {
            if port == "Wi-Fi" || port == "AirPort" {
                return Kind::WiFi;
            }
            if port.starts_with("Thunderbolt") {
                return Kind::Thunderbolt;
            }
            return Kind::Ethernet;
        }
    }
    Kind::Ethernet
}

fn run(program: &str, args: &[&str]) -> String {
    std::process::Command::new(program)
        .args(args)
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .unwrap_or_default()
}

/// The kind of link to `ip`.
pub fn kind(ip: std::net::IpAddr) -> Kind {
    if ip.is_loopback() {
        return Kind::Loopback;
    }
    let route = run("/sbin/route", &["-n", "get", &ip.to_string()]);
    let Some(interface) = route_interface(&route) else {
        return Kind::Ethernet;
    };
    kind_of(
        interface,
        &run("/usr/sbin/networksetup", &["-listallhardwareports"]),
    )
}

/// The value at quantile `q` of `values` (sorted here).
fn quantile(values: &mut [f64], q: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    values
        .get(((values.len() as f64 - 1.0) * q).round() as usize)
        .copied()
        .unwrap_or(0.0)
}

/// Measures the link to the pool server at `address` over `client`: 20 heartbeats and a 4 MB
/// transfer.
pub async fn measure(client: &reqwest::Client, address: &str, kind: Kind) -> Result<Link, String> {
    let url = format!("https://{address}/pool/v1/heartbeat");
    let mut rtts = Vec::with_capacity(20);
    for _ in 0..21 {
        let start = Instant::now();
        client
            .get(&url)
            .timeout(Duration::from_secs(5))
            .send()
            .await
            .map_err(|e| e.to_string())?;
        rtts.push(start.elapsed().as_secs_f64() * 1e3);
    }
    rtts.remove(0); // the first opens the connection
    let payload = vec![0u8; 4 << 20];
    let start = Instant::now();
    client
        .post(format!("https://{address}/pool/v1/echo"))
        .body(payload)
        .timeout(Duration::from_secs(60))
        .send()
        .await
        .map_err(|e| e.to_string())?;
    let secs = start.elapsed().as_secs_f64().max(1e-6);
    Ok(Link {
        kind,
        rtt_ms_p50: quantile(&mut rtts.clone(), 0.5),
        rtt_ms_p90: quantile(&mut rtts, 0.9),
        mbps: (4 << 20) as f64 * 8.0 / 1e6 / secs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const PORTS: &str = "\nHardware Port: Ethernet Adapter (en4)\nDevice: en4\nEthernet Address: 56:62\n\n\
        Hardware Port: Thunderbolt Bridge\nDevice: bridge0\n\nHardware Port: Wi-Fi\nDevice: en0\n\n\
        Hardware Port: Thunderbolt 1\nDevice: en1\n";

    #[test]
    fn link_kinds_come_from_the_routes_interface() {
        let route =
            "   route to: 10.0.0.5\ndestination: 10.0.0.5\n  interface: en0\n      flags: <UP>";
        assert_eq!(route_interface(route), Some("en0"));
        assert_eq!(kind_of("en0", PORTS), Kind::WiFi);
        assert_eq!(kind_of("bridge0", PORTS), Kind::Thunderbolt);
        assert_eq!(kind_of("en1", PORTS), Kind::Thunderbolt);
        assert_eq!(kind_of("en4", PORTS), Kind::Ethernet);
        assert_eq!(kind_of("lo0", PORTS), Kind::Loopback);
        assert_eq!(kind_of("en9", PORTS), Kind::Ethernet);
        assert_eq!(kind("127.0.0.1".parse().unwrap()), Kind::Loopback);
    }

    #[test]
    fn quantiles() {
        let mut v = vec![5.0, 1.0, 3.0, 2.0, 4.0];
        assert_eq!(quantile(&mut v, 0.5), 3.0);
        assert_eq!(quantile(&mut v, 0.9), 5.0);
        assert_eq!(quantile(&mut [], 0.5), 0.0);
    }
}
