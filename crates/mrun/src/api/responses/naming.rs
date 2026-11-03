//! How calls go back to a Responses client.

use crate::api::model::ChatRequest;
use serde_json::{Value, json};
use std::collections::HashMap;

/// How calls go back to the client: namespaced tools split into `namespace` and `name`
/// (FR-027), and calls of Codex's client-executed `tool_search` as `tool_search_call` items.
#[derive(Clone, Debug, Default)]
pub struct Naming {
    namespaced: HashMap<String, (String, String)>,
    tool_search: bool,
}

/// Name of the synthetic tool standing for Codex's `tool_search` (client-executed).
pub const TOOL_SEARCH: &str = "tool_search";

impl Naming {
    pub fn of(request: &ChatRequest) -> Self {
        let mut naming = Self::default();
        for tool in &request.tools {
            if let Some((namespace, name)) = tool.split_name() {
                naming
                    .namespaced
                    .insert(tool.name.clone(), (namespace.to_string(), name.to_string()));
            }
            naming.tool_search |= tool.name == TOOL_SEARCH && tool.namespace.is_none();
        }
        naming
    }

    pub fn item(
        &self,
        id: &str,
        call_id: &str,
        name: &str,
        arguments: &str,
        status: &str,
    ) -> Value {
        if self.tool_search && name == TOOL_SEARCH {
            let arguments: Value = serde_json::from_str(arguments).unwrap_or_else(|_| json!({}));
            return json!({"type": "tool_search_call", "id": id, "call_id": call_id,
                "execution": "client", "arguments": arguments, "status": status});
        }
        let mut item = json!({"type": "function_call", "id": id, "call_id": call_id,
            "name": name, "arguments": arguments, "status": status});
        if let Some((namespace, tool)) = self.namespaced.get(name) {
            item["name"] = tool.as_str().into();
            item["namespace"] = namespace.as_str().into();
        }
        item
    }
}
