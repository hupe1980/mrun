//! What the rest of the system holds (D062): the memory macOS can hand out without swapping and
//! the largest other processes by resident memory. Read through `sysctl` (`kern.memorystatus_level`,
//! the kernel's own measure), `vm_stat` as a fallback, and `ps` (no unsafe
//! code, no arguments of other processes: names only); the parsers are pure and tested. Outside
//! macOS nothing is read and callers fall back to D048's budget.

/// Bytes macOS can hand out without swapping: free + inactive + purgeable + speculative pages,
/// from `vm_stat` text. `None` when the page size or the free count is missing.
pub fn parse_vm_stat(text: &str) -> Option<u64> {
    let page: u64 = text
        .lines()
        .next()?
        .split("page size of ")
        .nth(1)?
        .split_whitespace()
        .next()?
        .parse()
        .ok()?;
    let count = |key: &str| -> Option<u64> {
        text.lines()
            .find_map(|line| line.strip_prefix(key))?
            .trim()
            .trim_end_matches('.')
            .parse()
            .ok()
    };
    let free = count("Pages free:")?;
    let rest: u64 = ["Pages inactive:", "Pages purgeable:", "Pages speculative:"]
        .iter()
        .filter_map(|key| count(key))
        .sum();
    (free + rest).checked_mul(page)
}

/// Bytes available from the kernel's own measure: `kern.memorystatus_level` (the percentage of
/// memory available, the figure Jetsam acts on; it counts reclaimable file cache and compression)
/// × `hw.memsize`. `None` when either value is missing or out of range.
pub fn parse_memorystatus(level: &str, memsize: &str) -> Option<u64> {
    let level: u64 = level.trim().parse().ok()?;
    let memsize: u64 = memsize.trim().parse().ok()?;
    (level <= 100).then(|| memsize / 100 * level)
}

/// Available memory now: the kernel's measure ([`parse_memorystatus`]), else `vm_stat`'s free +
/// inactive + purgeable + speculative pages ([`parse_vm_stat`], which misses reclaimable file
/// cache and so reads low: 19 GiB where the kernel reported 36.5 GiB on a 48 GB Mac). `None`
/// outside macOS or when neither can be read.
pub fn available() -> Option<u64> {
    if !cfg!(target_os = "macos") {
        return None;
    }
    let sysctl = |name: &str| -> Option<String> {
        let out = std::process::Command::new("/usr/sbin/sysctl")
            .args(["-n", name])
            .output()
            .ok()?;
        out.status
            .success()
            .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
    };
    if let (Some(level), Some(memsize)) = (sysctl("kern.memorystatus_level"), sysctl("hw.memsize"))
        && let Some(bytes) = parse_memorystatus(&level, &memsize)
    {
        return Some(bytes);
    }
    let out = std::process::Command::new("/usr/bin/vm_stat")
        .output()
        .ok()?;
    out.status.success().then_some(())?;
    parse_vm_stat(&String::from_utf8_lossy(&out.stdout))
}

/// A process holding memory: its executable's name (no path, no arguments) and resident bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Process {
    pub pid: u32,
    pub name: String,
    pub resident: u64,
}

impl Process {
    /// A virtual machine host (Apple's Virtualization framework, QEMU, Docker, Parallels,
    /// VMware, VirtualBox): its guest's memory is not visible to the guest's own tools.
    pub fn is_vm(&self) -> bool {
        const VMS: [&str; 7] = [
            "com.apple.Virtualization.VirtualMachine",
            "qemu-system",
            "com.docker.virtualization",
            "com.docker.krun",
            "prl_vm_app",
            "vmware-vmx",
            "VBoxHeadless",
        ];
        VMS.iter().any(|vm| self.name.starts_with(vm))
    }
}

/// `ps -axo pid=,rss=,comm=` text: one process per line, resident size in KiB, the executable
/// path last (it may contain spaces). Only the basename is kept.
pub fn parse_ps(text: &str) -> Vec<Process> {
    text.lines()
        .filter_map(|line| {
            let mut fields = line.trim_start().splitn(2, char::is_whitespace);
            let pid = fields.next()?.parse().ok()?;
            let rest = fields.next()?.trim_start();
            let mut fields = rest.splitn(2, char::is_whitespace);
            let kib: u64 = fields.next()?.parse().ok()?;
            let command = fields.next()?.trim();
            let name = command.rsplit('/').next().unwrap_or(command).to_string();
            (!name.is_empty()).then(|| Process {
                pid,
                name,
                resident: kib * 1024,
            })
        })
        .collect()
}

/// Every process but this one, largest resident memory first; empty when `ps` fails.
pub fn processes() -> Vec<Process> {
    let Ok(out) = std::process::Command::new("ps")
        .args(["-axo", "pid=,rss=,comm="])
        .output()
    else {
        return Vec::new();
    };
    let own = std::process::id();
    let mut all: Vec<Process> = parse_ps(&String::from_utf8_lossy(&out.stdout))
        .into_iter()
        .filter(|p| p.pid != own)
        .collect();
    all.sort_by(|a, b| b.resident.cmp(&a.resident).then(a.name.cmp(&b.name)));
    all
}

/// `com.apple.Virtualization.VirtualMachine 32.3 GiB, rust-analyzer 4.3 GiB`.
pub fn describe(processes: &[Process]) -> String {
    processes
        .iter()
        .map(|p| {
            format!(
                "{} {:.1} GiB",
                p.name,
                p.resident as f64 / (1u64 << 30) as f64
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    const VM_STAT: &str = "Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                   173578.
Pages active:                                1261979.
Pages inactive:                              1148361.
Pages speculative:                            112595.
Pages throttled:                                   0.
Pages wired down:                             175483.
Pages purgeable:                                7909.
\"Translation faults\":                      112492617.
";

    #[test]
    fn available_is_the_kernels_percentage_of_physical_memory() {
        assert_eq!(
            parse_memorystatus("76\n", "51539607552\n"),
            Some(39_170_101_700)
        );
        assert_eq!(parse_memorystatus("0", "51539607552"), Some(0));
        assert_eq!(parse_memorystatus("101", "51539607552"), None);
        assert_eq!(parse_memorystatus("x", "51539607552"), None);
    }

    #[test]
    fn available_is_free_inactive_purgeable_and_speculative() {
        assert_eq!(
            parse_vm_stat(VM_STAT),
            Some((173_578 + 1_148_361 + 112_595 + 7909) * 16_384)
        );
        assert_eq!(parse_vm_stat("Pages free: 1."), None);
        let four_k = VM_STAT.replace("16384", "4096");
        assert_eq!(
            parse_vm_stat(&four_k),
            Some((173_578 + 1_148_361 + 112_595 + 7909) * 4096)
        );
    }

    #[test]
    fn processes_are_named_by_their_executable_only() {
        let text = "  812 33868800 /System/Library/Frameworks/Virtualization.framework/Versions/A/XPCServices/com.apple.Virtualization.VirtualMachine.xpc/Contents/MacOS/com.apple.Virtualization.VirtualMachine
 9001  4508876 /Users/me/.vscode/extensions/rust-analyzer/server/rust-analyzer
   77  2306867 /Applications/Visual Studio Code.app/Contents/Frameworks/Code Helper (Renderer).app/Contents/MacOS/Code Helper (Renderer)
   12      100 launchd
 garbage
";
        let all = parse_ps(text);
        assert_eq!(all.len(), 4);
        assert_eq!(all[0].name, "com.apple.Virtualization.VirtualMachine");
        assert_eq!(all[0].resident, 33_868_800 * 1024);
        assert!(all[0].is_vm());
        assert_eq!(all[2].name, "Code Helper (Renderer)");
        assert!(!all[1].is_vm());
        assert_eq!(
            describe(&all[..2]),
            "com.apple.Virtualization.VirtualMachine 32.3 GiB, rust-analyzer 4.3 GiB"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn this_mac_reports_available_memory_and_processes() {
        assert!(available().is_some_and(|bytes| bytes > 0));
        assert!(!processes().is_empty());
    }
}
