//! Wire types of REST v1 and the GraphQL GPU query (fields mrun reads; unknown fields ignored).

use serde::{Deserialize, Deserializer, Serialize};
use std::collections::BTreeMap;

/// A pod as `GET /v1/pods/{id}` returns it.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", default)]
pub struct Pod {
    pub id: String,
    pub name: String,
    /// `RUNNING`, `EXITED` or `TERMINATED`.
    pub desired_status: String,
    /// $/h as billed (RunPod sends a number or a string).
    #[serde(deserialize_with = "number_or_string")]
    pub cost_per_hr: f64,
    pub image_name: String,
    pub gpu: Option<PodGpu>,
    pub cpu_flavor_id: Option<String>,
    /// The pod's environment as created; holds `MRUN_POD_TOKEN` for mrun pods.
    pub env: BTreeMap<String, String>,
    pub created_at: Option<String>,
    pub last_started_at: Option<String>,
    pub last_status_change: Option<String>,
    pub ports: Vec<String>,
    pub machine: Option<Machine>,
}

impl Pod {
    /// `https://<id>-<port>.proxy.runpod.net`: the HTTPS proxy of an exposed HTTP port.
    pub fn proxy_url(&self, port: u16) -> String {
        format!("https://{}-{port}.proxy.runpod.net", self.id)
    }

    pub fn is_running(&self) -> bool {
        self.desired_status == "RUNNING"
    }
}

/// The GPU of a pod.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", default)]
pub struct PodGpu {
    pub id: String,
    pub count: u32,
    pub display_name: Option<String>,
}

/// Where a pod runs.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", default)]
pub struct Machine {
    pub data_center_id: Option<String>,
    pub gpu_type_id: Option<String>,
    pub location: Option<String>,
}

/// Secure or Community Cloud.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CloudType {
    #[default]
    Secure,
    Community,
}

/// What a pod computes on.
#[derive(Clone, Debug, PartialEq)]
pub enum Compute {
    Gpu {
        type_id: String,
        cloud: CloudType,
    },
    /// A CPU pod (tests): flavor (`cpu3c`) and vCPUs.
    Cpu {
        flavor: String,
        vcpus: u32,
    },
}

/// `POST /v1/pods`.
#[derive(Clone, Debug, PartialEq)]
pub struct PodCreate {
    pub name: String,
    pub image_name: String,
    pub compute: Compute,
    pub container_disk_gb: u32,
    pub volume_gb: u32,
    pub network_volume_id: Option<String>,
    pub data_center_ids: Vec<String>,
    /// `8000/http`.
    pub ports: Vec<String>,
    pub env: BTreeMap<String, String>,
    pub docker_entrypoint: Vec<String>,
    pub docker_start_cmd: Vec<String>,
    pub allowed_cuda_versions: Vec<String>,
    /// Least host RAM per GPU in GB (`minRAMPerGPU`; `None`: RunPod's default). Models that keep
    /// weights in host memory need it.
    pub min_ram_per_gpu_gb: Option<u32>,
}

impl PodCreate {
    /// The request body.
    pub fn body(&self) -> serde_json::Value {
        let mut body = serde_json::json!({
            "name": self.name,
            "imageName": self.image_name,
            "containerDiskInGb": self.container_disk_gb,
            "volumeInGb": self.volume_gb,
            "ports": self.ports,
            "env": self.env,
            "dockerEntrypoint": self.docker_entrypoint,
            "dockerStartCmd": self.docker_start_cmd,
        });
        match &self.compute {
            Compute::Gpu { type_id, cloud } => {
                body["computeType"] = "GPU".into();
                body["gpuTypeIds"] = serde_json::json!([type_id]);
                body["gpuCount"] = 1.into();
                body["cloudType"] = serde_json::to_value(cloud).unwrap_or_default();
                if !self.allowed_cuda_versions.is_empty() {
                    body["allowedCudaVersions"] = serde_json::json!(self.allowed_cuda_versions);
                }
                if let Some(gb) = self.min_ram_per_gpu_gb {
                    body["minRAMPerGPU"] = gb.into();
                }
            }
            Compute::Cpu { flavor, vcpus } => {
                body["computeType"] = "CPU".into();
                body["cpuFlavorIds"] = serde_json::json!([flavor]);
                body["vcpuCount"] = (*vcpus).into();
            }
        }
        if let Some(id) = &self.network_volume_id {
            body["networkVolumeId"] = id.clone().into();
        }
        if !self.data_center_ids.is_empty() {
            body["dataCenterIds"] = serde_json::json!(self.data_center_ids);
            body["dataCenterPriority"] = "custom".into();
        }
        body
    }
}

/// A GPU type with its prices (GraphQL `gpuTypes`).
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", default)]
pub struct GpuType {
    /// `NVIDIA L40S`: what `gpuTypeIds` takes.
    pub id: String,
    /// `L40S`.
    pub display_name: String,
    #[serde(rename = "memoryInGb")]
    pub memory_gb: u32,
    pub secure_cloud: bool,
    pub community_cloud: bool,
    #[serde(deserialize_with = "number_or_string")]
    pub secure_price: f64,
    #[serde(deserialize_with = "number_or_string")]
    pub community_price: f64,
    /// Stock in Secure Cloud right now (`High`, `Medium`, `Low`; `None` when sold out).
    #[serde(rename = "secureStock")]
    pub secure_stock: Option<Stock>,
    #[serde(rename = "communityStock")]
    pub community_stock: Option<Stock>,
}

/// A GPU type's stock (GraphQL `lowestPrice { stockStatus }`).
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", default)]
pub struct Stock {
    pub stock_status: Option<String>,
}

impl GpuType {
    /// $/h in `cloud`, `None` when the type is not offered there.
    pub fn price(&self, cloud: CloudType) -> Option<f64> {
        match cloud {
            CloudType::Secure => {
                (self.secure_cloud && self.secure_price > 0.0).then_some(self.secure_price)
            }
            CloudType::Community => {
                (self.community_cloud && self.community_price > 0.0).then_some(self.community_price)
            }
        }
    }

    /// An NVIDIA GPU (CUDA images such as vLLM's run on these only).
    pub fn is_nvidia(&self) -> bool {
        self.id.starts_with("NVIDIA ") || self.id.starts_with("Tesla ")
    }

    /// Stock in `cloud` right now; `None` when none is available (or unknown).
    pub fn stock(&self, cloud: CloudType) -> Option<&str> {
        match cloud {
            CloudType::Secure => self.secure_stock.as_ref(),
            CloudType::Community => self.community_stock.as_ref(),
        }
        .and_then(|s| s.stock_status.as_deref())
    }

    /// The cheaper of the two offered prices.
    pub fn lowest_price(&self) -> Option<f64> {
        match (
            self.price(CloudType::Secure),
            self.price(CloudType::Community),
        ) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }

    /// `name` is the id or the display name, case-insensitive, with or without `NVIDIA `.
    pub fn matches(&self, name: &str) -> bool {
        let name = name.trim();
        let strip = |s: &str| {
            let lower = s.to_ascii_lowercase();
            lower
                .strip_prefix("nvidia ")
                .map(str::to_string)
                .unwrap_or(lower)
        };
        let wanted = strip(name);
        strip(&self.id) == wanted
            || strip(&self.display_name) == wanted
            || strip(&self.id).strip_prefix("geforce ") == Some(wanted.as_str())
    }
}

/// A network volume.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", default)]
pub struct NetworkVolume {
    pub id: String,
    pub name: String,
    #[serde(rename = "size")]
    pub size_gb: u32,
    pub data_center_id: String,
}

fn number_or_string<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
    let value = serde_json::Value::deserialize(deserializer)?;
    Ok(match value {
        serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.0),
        serde_json::Value::String(s) => s.trim().parse().unwrap_or(0.0),
        _ => 0.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pod_parses_with_string_or_number_prices_and_unknown_fields() {
        let pod: Pod = serde_json::from_str(
            r#"{"id":"abc","name":"mrun-x","desiredStatus":"RUNNING","costPerHr":"0.74",
                "imageName":"img","env":{"A":"1"},"gpu":{"id":"NVIDIA L40S","count":1},
                "somethingNew":true}"#,
        )
        .unwrap();
        assert_eq!(pod.cost_per_hr, 0.74);
        assert!(pod.is_running());
        assert_eq!(pod.proxy_url(8000), "https://abc-8000.proxy.runpod.net");
        let pod: Pod = serde_json::from_str(r#"{"id":"b","costPerHr":0.06}"#).unwrap();
        assert_eq!(pod.cost_per_hr, 0.06);
    }

    #[test]
    fn gpu_names_match_ids_and_display_names() {
        let gpu = GpuType {
            id: "NVIDIA GeForce RTX 4090".into(),
            display_name: "RTX 4090".into(),
            memory_gb: 24,
            secure_cloud: true,
            community_cloud: true,
            secure_price: 0.89,
            community_price: 0.34,
            ..GpuType::default()
        };
        for name in [
            "RTX 4090",
            "rtx 4090",
            "NVIDIA GeForce RTX 4090",
            "geforce rtx 4090",
        ] {
            assert!(gpu.matches(name), "{name}");
        }
        assert!(!gpu.matches("RTX 3090"));
        assert_eq!(gpu.price(CloudType::Community), Some(0.34));
        assert_eq!(gpu.lowest_price(), Some(0.34));
        let only_secure = GpuType {
            community_cloud: false,
            ..gpu
        };
        assert_eq!(only_secure.price(CloudType::Community), None);
    }

    #[test]
    fn create_bodies_for_gpu_and_cpu() {
        let mut create = PodCreate {
            name: "mrun-x".into(),
            image_name: "img@sha256:00".into(),
            compute: Compute::Gpu {
                type_id: "NVIDIA L40S".into(),
                cloud: CloudType::Community,
            },
            container_disk_gb: 60,
            volume_gb: 0,
            network_volume_id: None,
            data_center_ids: vec![],
            ports: vec!["8000/http".into()],
            env: BTreeMap::from([("MRUN_MODEL".into(), "m".into())]),
            docker_entrypoint: vec!["python3".into()],
            docker_start_cmd: vec![],
            allowed_cuda_versions: vec!["12.9".into()],
            min_ram_per_gpu_gb: Some(120),
        };
        let body = create.body();
        assert_eq!(body["computeType"], "GPU");
        assert_eq!(body["gpuTypeIds"][0], "NVIDIA L40S");
        assert_eq!(body["cloudType"], "COMMUNITY");
        assert_eq!(body["allowedCudaVersions"][0], "12.9");
        assert_eq!(body["minRAMPerGPU"], 120);
        assert!(body.get("networkVolumeId").is_none());
        create.compute = Compute::Cpu {
            flavor: "cpu3c".into(),
            vcpus: 2,
        };
        create.network_volume_id = Some("v1".into());
        create.data_center_ids = vec!["EU-RO-1".into()];
        let body = create.body();
        assert_eq!(body["computeType"], "CPU");
        assert_eq!(body["cpuFlavorIds"][0], "cpu3c");
        assert!(body.get("gpuTypeIds").is_none());
        assert!(body.get("minRAMPerGPU").is_none());
        assert_eq!(body["networkVolumeId"], "v1");
        assert_eq!(body["dataCenterPriority"], "custom");
    }
}
