//! A small client for RunPod, for mrun's opt-in remote GPU mode (013, D064): pods and network
//! volumes over REST v1, GPU types with prices over GraphQL (REST v1 has no such route), and the
//! v2 pod action a pod uses to stop itself with its pod-scoped key.
//!
//! Every request carries `User-Agent: mrun/<version>` (Cloudflare in front of RunPod rejects
//! library defaults, error 1010) and `Authorization: Bearer <key>`. The key is never logged,
//! never part of an error and never put in a URL.

mod error;
mod types;

pub use error::Error;
pub use types::{
    CloudType, Compute, GpuType, Machine, NetworkVolume, Pod, PodCreate, PodGpu, Stock,
};

use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use std::path::Path;
use std::time::Duration;

/// `mrun/<version>`: the user agent of every request (013 FR-011).
pub const USER_AGENT: &str = concat!("mrun/", env!("CARGO_PKG_VERSION"));

/// The variable holding the key: the account key on the Mac, the pod-scoped key inside a pod.
pub const KEY_VARIABLE: &str = "RUNPOD_API_KEY";

/// A RunPod API key. `Debug` and `Display` never show it.
#[derive(Clone, PartialEq, Eq)]
pub struct ApiKey(String);

impl ApiKey {
    pub fn new(key: impl Into<String>) -> Self {
        Self(key.into())
    }

    /// `RUNPOD_API_KEY` from the environment, else a `RUNPOD_API_KEY=` line in `<dir>/.env`
    /// (013 FR-001). Empty values count as missing.
    pub fn from_env_or_dotenv(dir: &Path) -> Option<Self> {
        if let Ok(key) = std::env::var(KEY_VARIABLE)
            && !key.trim().is_empty()
        {
            return Some(Self(key.trim().to_string()));
        }
        let text = std::fs::read_to_string(dir.join(".env")).ok()?;
        dotenv_value(&text, KEY_VARIABLE).map(Self)
    }

    /// The key itself, for the `Authorization` header only.
    fn expose(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ApiKey(***)")
    }
}

impl std::fmt::Display for ApiKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("***")
    }
}

/// The value of `name` in a `.env` text: `NAME=value`, `export NAME=value`, quotes stripped,
/// `#` comments and blank lines skipped.
pub fn dotenv_value(text: &str, name: &str) -> Option<String> {
    text.lines().find_map(|line| {
        let line = line.trim();
        let line = line.strip_prefix("export ").unwrap_or(line).trim_start();
        let (key, value) = line.split_once('=')?;
        if key.trim() != name {
            return None;
        }
        let value = value.trim();
        let value = value
            .strip_prefix('"')
            .and_then(|v| v.strip_suffix('"'))
            .or_else(|| value.strip_prefix('\'').and_then(|v| v.strip_suffix('\'')))
            .unwrap_or(value);
        (!value.is_empty()).then(|| value.to_string())
    })
}

/// Base URLs (tests point them at a local mock server).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Endpoints {
    /// `https://rest.runpod.io/v1`
    pub rest: String,
    /// `https://api.runpod.io/graphql`
    pub graphql: String,
    /// `https://api.runpod.io/v2`
    pub v2: String,
}

impl Default for Endpoints {
    fn default() -> Self {
        Self {
            rest: "https://rest.runpod.io/v1".into(),
            graphql: "https://api.runpod.io/graphql".into(),
            v2: "https://api.runpod.io/v2".into(),
        }
    }
}

/// The RunPod client.
#[derive(Clone, Debug)]
pub struct Client {
    http: reqwest::Client,
    key: ApiKey,
    endpoints: Endpoints,
}

impl Client {
    pub fn new(key: ApiKey) -> Result<Self, Error> {
        Self::with_endpoints(key, Endpoints::default())
    }

    pub fn with_endpoints(key: ApiKey, endpoints: Endpoints) -> Result<Self, Error> {
        let http = reqwest::Client::builder()
            .user_agent(USER_AGENT)
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(60))
            .build()
            .map_err(|e| Error::transport(&e))?;
        Ok(Self {
            http,
            key,
            endpoints,
        })
    }

    pub fn endpoints(&self) -> &Endpoints {
        &self.endpoints
    }

    async fn send(&self, request: reqwest::RequestBuilder) -> Result<(u16, String), Error> {
        let response = request
            .bearer_auth(self.key.expose())
            .send()
            .await
            .map_err(|e| Error::transport(&e))?;
        let status = response.status().as_u16();
        let body = response.text().await.map_err(|e| Error::transport(&e))?;
        if (200..300).contains(&status) {
            Ok((status, body))
        } else {
            Err(Error::from_status(status, &body))
        }
    }

    async fn json<T: DeserializeOwned>(
        &self,
        request: reqwest::RequestBuilder,
    ) -> Result<T, Error> {
        let (_, body) = self.send(request).await?;
        serde_json::from_str(&body).map_err(|e| Error::Decode(e.to_string()))
    }

    fn rest(&self, path: &str) -> String {
        format!("{}{path}", self.endpoints.rest.trim_end_matches('/'))
    }

    /// Every GPU type with VRAM and Secure/Community prices (GraphQL).
    pub async fn gpu_types(&self) -> Result<Vec<GpuType>, Error> {
        let query = "query { gpuTypes { id displayName memoryInGb secureCloud communityCloud \
                     securePrice communityPrice \
                     secureStock: lowestPrice(input: {gpuCount: 1, secureCloud: true}) { stockStatus } \
                     communityStock: lowestPrice(input: {gpuCount: 1, secureCloud: false}) { stockStatus } } }";
        let value: Value = self
            .json(
                self.http
                    .post(&self.endpoints.graphql)
                    .json(&json!({ "query": query })),
            )
            .await?;
        if let Some(message) = value["errors"][0]["message"].as_str() {
            return Err(Error::Api {
                status: 200,
                message: message.to_string(),
            });
        }
        let types: Vec<GpuType> = serde_json::from_value(value["data"]["gpuTypes"].clone())
            .map_err(|e| Error::Decode(e.to_string()))?;
        Ok(types.into_iter().filter(|g| g.id != "unknown").collect())
    }

    /// `GET /pods`.
    pub async fn list_pods(&self) -> Result<Vec<Pod>, Error> {
        self.json(self.http.get(self.rest("/pods"))).await
    }

    /// `GET /pods/{id}`.
    pub async fn get_pod(&self, id: &str) -> Result<Pod, Error> {
        self.json(self.http.get(self.rest(&format!("/pods/{id}"))))
            .await
    }

    /// `POST /pods`.
    pub async fn create_pod(&self, create: &PodCreate) -> Result<Pod, Error> {
        self.json(self.http.post(self.rest("/pods")).json(&create.body()))
            .await
    }

    /// `POST /pods/{id}/stop` (account key; the pod keeps its disk and is not billed for GPU).
    pub async fn stop_pod(&self, id: &str) -> Result<(), Error> {
        self.send(self.http.post(self.rest(&format!("/pods/{id}/stop"))))
            .await
            .map(drop)
    }

    /// `DELETE /pods/{id}`. A pod that is already gone counts as terminated.
    pub async fn terminate_pod(&self, id: &str) -> Result<(), Error> {
        match self
            .send(self.http.delete(self.rest(&format!("/pods/{id}"))))
            .await
        {
            Ok(_) | Err(Error::NotFound) => Ok(()),
            Err(error) => Err(error),
        }
    }

    /// `GET /networkvolumes`.
    pub async fn list_volumes(&self) -> Result<Vec<NetworkVolume>, Error> {
        self.json(self.http.get(self.rest("/networkvolumes"))).await
    }

    /// `POST /networkvolumes`.
    pub async fn create_volume(
        &self,
        name: &str,
        size_gb: u32,
        data_center_id: &str,
    ) -> Result<NetworkVolume, Error> {
        self.json(
            self.http
                .post(self.rest("/networkvolumes"))
                .json(&json!({"name": name, "size": size_gb, "dataCenterId": data_center_id})),
        )
        .await
    }

    /// `DELETE /networkvolumes/{id}`; an absent volume counts as deleted.
    pub async fn delete_volume(&self, id: &str) -> Result<(), Error> {
        match self
            .send(
                self.http
                    .delete(self.rest(&format!("/networkvolumes/{id}"))),
            )
            .await
        {
            Ok(_) | Err(Error::NotFound) => Ok(()),
            Err(error) => Err(error),
        }
    }

    /// Stops pod `pod_id` through `POST <v2>/pods/{id}/action {"action":"stop"}`: the call a
    /// pod's own pod-scoped key may make (REST v1 `stop` answers 403 for it; measured
    /// 2026-10-19, D064). An answer carrying an `error` is a failure.
    pub async fn self_stop(&self, pod_id: &str) -> Result<(), Error> {
        let url = format!(
            "{}/pods/{pod_id}/action",
            self.endpoints.v2.trim_end_matches('/')
        );
        let (status, body) = self
            .send(self.http.post(url).json(&json!({"action": "stop"})))
            .await?;
        if let Ok(value) = serde_json::from_str::<Value>(&body)
            && let Some(error) = value.get("error").filter(|e| !e.is_null())
        {
            return Err(Error::Api {
                status,
                message: error
                    .as_str()
                    .map_or_else(|| error.to_string(), str::to_string),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::Router;
    use axum::extract::{Request, State};
    use axum::http::StatusCode;
    use axum::response::{IntoResponse, Response};
    use std::sync::{Arc, Mutex};

    const KEY: &str = "rpa_TESTKEY0123456789";

    /// One recorded request: method, path, user agent, authorization, body.
    #[derive(Clone, Debug)]
    struct Seen {
        method: String,
        path: String,
        user_agent: String,
        auth: String,
        body: Value,
    }

    type Log = Arc<Mutex<Vec<Seen>>>;
    type Reply = Arc<dyn Fn(&str, &str) -> (u16, String) + Send + Sync>;

    async fn record(State((log, reply)): State<(Log, Reply)>, request: Request) -> Response {
        let (parts, body) = request.into_parts();
        let bytes = axum::body::to_bytes(body, 1 << 20).await.unwrap();
        let header = |name: &str| {
            parts
                .headers
                .get(name)
                .and_then(|v| v.to_str().ok())
                .unwrap_or_default()
                .to_string()
        };
        let seen = Seen {
            method: parts.method.to_string(),
            path: parts.uri.path().to_string(),
            user_agent: header("user-agent"),
            auth: header("authorization"),
            body: serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        };
        let (status, text) = reply(&seen.method, &seen.path);
        log.lock().unwrap().push(seen);
        (StatusCode::from_u16(status).unwrap(), text).into_response()
    }

    /// A mock RunPod on 127.0.0.1 answering with `reply(method, path)`.
    async fn mock(reply: Reply) -> (Client, Log) {
        let log: Log = Arc::default();
        let app = Router::new()
            .fallback(record)
            .with_state((log.clone(), reply));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await });
        let base = format!("http://{addr}");
        let client = Client::with_endpoints(
            ApiKey::new(KEY),
            Endpoints {
                rest: format!("{base}/v1"),
                graphql: format!("{base}/graphql"),
                v2: format!("{base}/v2"),
            },
        )
        .unwrap();
        (client, log)
    }

    fn pod_json(id: &str) -> String {
        json!({"id": id, "name": "mrun-test", "desiredStatus": "RUNNING", "costPerHr": 0.06,
               "imageName": "python:3.12-slim", "env": {"MRUN_POD_TOKEN": "t"}})
        .to_string()
    }

    #[tokio::test]
    async fn every_call_sends_the_user_agent_and_the_bearer_key() {
        let reply: Reply = Arc::new(|method, path| match (method, path) {
            ("POST", "/graphql") => (
                200,
                json!({"data": {"gpuTypes": [
                    {"id": "NVIDIA L40S", "displayName": "L40S", "memoryInGb": 48,
                     "secureCloud": true, "communityCloud": true,
                     "securePrice": 1.09, "communityPrice": 0.79,
                     "secureStock": {"stockStatus": "Medium"}, "communityStock": null},
                    {"id": "unknown", "displayName": "unknown", "memoryInGb": 0}
                ]}})
                .to_string(),
            ),
            ("GET", "/v1/pods") => (200, format!("[{}]", pod_json("p1"))),
            ("GET", "/v1/pods/p1") => (200, pod_json("p1")),
            ("POST", "/v1/pods") => (201, pod_json("p2")),
            ("POST", "/v1/pods/p1/stop") => (200, "{}".into()),
            ("DELETE", "/v1/pods/p1") => (204, String::new()),
            ("GET", "/v1/networkvolumes") => (
                200,
                r#"[{"id":"v1","name":"mrun-m","size":50,"dataCenterId":"EU-RO-1"}]"#.into(),
            ),
            ("POST", "/v1/networkvolumes") => (
                200,
                r#"{"id":"v2","name":"mrun-m","size":60,"dataCenterId":"EU-RO-1"}"#.into(),
            ),
            ("DELETE", "/v1/networkvolumes/v1") => (204, String::new()),
            ("POST", "/v2/pods/p1/action") => (200, r#"{"status":"ok"}"#.into()),
            _ => (418, String::new()),
        });
        let (client, log) = mock(reply).await;
        let gpus = client.gpu_types().await.unwrap();
        assert_eq!(gpus.len(), 1, "the `unknown` type is dropped");
        assert_eq!(gpus[0].community_price, 0.79);
        assert_eq!(gpus[0].stock(CloudType::Secure), Some("Medium"));
        assert_eq!(gpus[0].stock(CloudType::Community), None);
        assert_eq!(client.list_pods().await.unwrap()[0].id, "p1");
        assert_eq!(
            client.get_pod("p1").await.unwrap().env["MRUN_POD_TOKEN"],
            "t"
        );
        let create = PodCreate {
            name: "mrun-test-x".into(),
            image_name: "python:3.12-slim".into(),
            compute: Compute::Cpu {
                flavor: "cpu3c".into(),
                vcpus: 2,
            },
            container_disk_gb: 5,
            volume_gb: 0,
            network_volume_id: None,
            data_center_ids: vec![],
            ports: vec!["8000/http".into()],
            env: Default::default(),
            docker_entrypoint: vec![],
            docker_start_cmd: vec![],
            allowed_cuda_versions: vec![],
            min_ram_per_gpu_gb: None,
        };
        assert_eq!(client.create_pod(&create).await.unwrap().id, "p2");
        client.stop_pod("p1").await.unwrap();
        client.terminate_pod("p1").await.unwrap();
        assert_eq!(client.list_volumes().await.unwrap()[0].size_gb, 50);
        assert_eq!(
            client
                .create_volume("mrun-m", 60, "EU-RO-1")
                .await
                .unwrap()
                .id,
            "v2"
        );
        client.delete_volume("v1").await.unwrap();
        client.self_stop("p1").await.unwrap();

        let log = log.lock().unwrap();
        assert_eq!(log.len(), 10);
        for seen in log.iter() {
            assert_eq!(seen.user_agent, USER_AGENT, "{seen:?}");
            assert!(USER_AGENT.starts_with("mrun/"));
            assert_eq!(
                seen.auth,
                format!("Bearer {KEY}"),
                "{} {}",
                seen.method,
                seen.path
            );
        }
        let created = log
            .iter()
            .find(|s| s.path == "/v1/pods" && s.method == "POST");
        assert_eq!(created.unwrap().body["computeType"], "CPU");
        let action = log.iter().find(|s| s.path == "/v2/pods/p1/action").unwrap();
        assert_eq!(action.body, json!({"action": "stop"}));
        let volume = log
            .iter()
            .find(|s| s.path == "/v1/networkvolumes" && s.method == "POST")
            .unwrap();
        assert_eq!(volume.body["dataCenterId"], "EU-RO-1");
    }

    #[tokio::test]
    async fn errors_are_typed_and_never_contain_the_key() {
        let reply: Reply = Arc::new(|method, path| {
            match (method, path) {
            ("GET", "/v1/pods") => (401, r#"{"error":"invalid api key"}"#.into()),
            ("GET", "/v1/pods/gone") => (404, "{}".into()),
            ("DELETE", "/v1/pods/gone") => (404, "{}".into()),
            ("POST", "/v1/pods") => (
                400,
                r#"{"error":"There are no longer any instances available with the requested specifications."}"#
                    .into(),
            ),
            ("POST", "/v1/pods/p1/stop") => (403, "forbidden".into()),
            ("POST", "/graphql") => (200, r#"{"errors":[{"message":"bad query"}]}"#.into()),
            ("POST", "/v2/pods/p1/action") => (200, r#"{"error":"pod not found"}"#.into()),
            ("GET", "/v1/networkvolumes") => (403, "<html>error code: 1010</html>".into()),
            _ => (500, "boom".into()),
        }
        });
        let (client, _) = mock(reply).await;
        let errors = [
            client.list_pods().await.unwrap_err(),
            client.get_pod("gone").await.unwrap_err(),
            client.stop_pod("p1").await.unwrap_err(),
            client.gpu_types().await.unwrap_err(),
            client.self_stop("p1").await.unwrap_err(),
            client.list_volumes().await.unwrap_err(),
        ];
        assert!(matches!(errors[0], Error::Unauthorized { status: 401 }));
        assert!(matches!(errors[1], Error::NotFound));
        assert!(matches!(errors[2], Error::Unauthorized { status: 403 }));
        assert!(matches!(errors[3], Error::Api { .. }));
        assert!(matches!(errors[4], Error::Api { .. }));
        assert!(matches!(errors[5], Error::Blocked));
        client.terminate_pod("gone").await.unwrap();
        let create = PodCreate {
            name: "mrun-test".into(),
            image_name: "i".into(),
            compute: Compute::Gpu {
                type_id: "NVIDIA L40S".into(),
                cloud: CloudType::Secure,
            },
            container_disk_gb: 5,
            volume_gb: 0,
            network_volume_id: None,
            data_center_ids: vec![],
            ports: vec![],
            env: Default::default(),
            docker_entrypoint: vec![],
            docker_start_cmd: vec![],
            allowed_cuda_versions: vec![],
            min_ram_per_gpu_gb: None,
        };
        let error = client.create_pod(&create).await.unwrap_err();
        assert!(
            error.to_string().contains("no longer any instances"),
            "{error}"
        );
        for error in errors.iter().chain([&error]) {
            assert!(!error.to_string().contains(KEY));
            assert!(!format!("{error:?}").contains(KEY));
        }
        assert!(!format!("{client:?}").contains(KEY));
    }

    #[tokio::test]
    async fn an_unreachable_server_is_a_transport_error() {
        let client = Client::with_endpoints(
            ApiKey::new(KEY),
            Endpoints {
                rest: "http://127.0.0.1:9/v1".into(),
                graphql: "http://127.0.0.1:9/graphql".into(),
                v2: "http://127.0.0.1:9/v2".into(),
            },
        )
        .unwrap();
        let error = client.list_pods().await.unwrap_err();
        assert!(matches!(error, Error::Transport(_)), "{error:?}");
        assert!(!error.to_string().contains(KEY));
    }

    #[test]
    fn keys_are_redacted_and_read_from_dotenv() {
        let key = ApiKey::new(KEY);
        assert_eq!(format!("{key:?}"), "ApiKey(***)");
        assert_eq!(key.to_string(), "***");
        let text = "# comment\nOTHER=1\nexport RUNPOD_API_KEY=\"rpa_abc\"\n";
        assert_eq!(dotenv_value(text, KEY_VARIABLE).as_deref(), Some("rpa_abc"));
        assert_eq!(
            dotenv_value("RUNPOD_API_KEY='x y'", KEY_VARIABLE).as_deref(),
            Some("x y")
        );
        assert_eq!(dotenv_value("RUNPOD_API_KEY=", KEY_VARIABLE), None);
        assert_eq!(dotenv_value("RUNPOD_API_KEY_2=a", KEY_VARIABLE), None);
        let dir = std::env::temp_dir().join(format!("mrun-runpod-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(".env"), "RUNPOD_API_KEY=rpa_file\n").unwrap();
        if std::env::var(KEY_VARIABLE).is_err() {
            assert_eq!(
                ApiKey::from_env_or_dotenv(&dir),
                Some(ApiKey::new("rpa_file"))
            );
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
