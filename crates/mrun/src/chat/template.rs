//! Chat templates from the model's own files, rendered with MiniJinja (Python-compatible methods).

use minijinja::{Environment, Error, ErrorKind, Value};
use serde::Serialize;
use std::fs;
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum TemplateError {
    #[error(
        "no chat template found in {0} (looked for chat_template.jinja and tokenizer_config.json)"
    )]
    Missing(String),
    #[error("cannot read {path}: {message}")]
    Read { path: String, message: String },
    #[error("chat template error: {0}")]
    Render(String),
}

/// Values available to the template besides the messages.
#[derive(Clone, Debug, Default, Serialize)]
pub struct RenderOptions {
    pub add_generation_prompt: bool,
    pub enable_thinking: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<serde_json::Value>,
}

#[derive(Debug)]
pub struct ChatTemplate {
    env: Environment<'static>,
    bos_token: String,
    eos_token: String,
}

impl ChatTemplate {
    pub fn new(source: String) -> Result<Self, TemplateError> {
        Self::with_tokens(source, String::new(), String::new())
    }

    fn with_tokens(
        source: String,
        bos_token: String,
        eos_token: String,
    ) -> Result<Self, TemplateError> {
        let mut env = Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function(
            "raise_exception",
            |message: String| -> Result<Value, Error> {
                Err(Error::new(ErrorKind::InvalidOperation, message))
            },
        );
        env.add_template_owned("chat", source)
            .map_err(|error| TemplateError::Render(error.to_string()))?;
        Ok(Self {
            env,
            bos_token,
            eos_token,
        })
    }

    /// Loads `chat_template.jinja`, or `chat_template` from `tokenizer_config.json`.
    pub fn from_model_dir(dir: &Path) -> Result<Self, TemplateError> {
        let read = |name: &str| -> Result<Option<String>, TemplateError> {
            let path = dir.join(name);
            match fs::read_to_string(&path) {
                Ok(text) => Ok(Some(text)),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
                Err(error) => Err(TemplateError::Read {
                    path: path.display().to_string(),
                    message: error.to_string(),
                }),
            }
        };
        let config: serde_json::Value = match read("tokenizer_config.json")? {
            Some(text) => serde_json::from_str(&text).map_err(|error| TemplateError::Read {
                path: dir.join("tokenizer_config.json").display().to_string(),
                message: error.to_string(),
            })?,
            None => serde_json::Value::Null,
        };
        let token = |key: &str| match &config[key] {
            serde_json::Value::String(text) => text.clone(),
            serde_json::Value::Object(object) => object
                .get("content")
                .and_then(|value| value.as_str())
                .unwrap_or_default()
                .to_string(),
            _ => String::new(),
        };
        let source = match read("chat_template.jinja")? {
            Some(text) => text,
            None => match &config["chat_template"] {
                serde_json::Value::String(text) => text.clone(),
                serde_json::Value::Array(templates) => templates
                    .iter()
                    .find(|entry| entry["name"] == "default")
                    .and_then(|entry| entry["template"].as_str())
                    .map(str::to_string)
                    .ok_or_else(|| TemplateError::Missing(dir.display().to_string()))?,
                _ => return Err(TemplateError::Missing(dir.display().to_string())),
            },
        };
        Self::with_tokens(source, token("bos_token"), token("eos_token"))
    }

    /// Renders `messages` (OpenAI-style JSON objects) to the prompt text.
    pub fn render(
        &self,
        messages: &serde_json::Value,
        options: &RenderOptions,
    ) -> Result<String, TemplateError> {
        let template = self
            .env
            .get_template("chat")
            .map_err(|error| TemplateError::Render(error.to_string()))?;
        template
            .render(minijinja::context! {
                messages => Value::from(minijinja::value::Serde(messages)),
                tools => Value::from(minijinja::value::Serde(&options.tools)),
                add_generation_prompt => options.add_generation_prompt,
                enable_thinking => options.enable_thinking,
                bos_token => &self.bos_token,
                eos_token => &self.eos_token,
            })
            .map_err(|error| TemplateError::Render(format!("{error:#}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHATML: &str = "{%- for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content.strip() }}<|im_end|>\n{%- endfor %}{%- if add_generation_prompt %}<|im_start|>assistant\n{%- if not enable_thinking %}<think>\n\n</think>\n\n{%- endif %}{%- endif %}";

    #[test]
    fn renders_chatml_with_python_string_methods_and_thinking_switch() {
        let template = ChatTemplate::new(CHATML.to_string()).unwrap();
        let messages = serde_json::json!([{"role": "user", "content": "  hi  "}]);
        let off = template
            .render(
                &messages,
                &RenderOptions {
                    add_generation_prompt: true,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(
            off,
            "<|im_start|>user\nhi<|im_end|><|im_start|>assistant<think>\n\n</think>"
        );
        let on = template
            .render(
                &messages,
                &RenderOptions {
                    add_generation_prompt: true,
                    enable_thinking: true,
                    tools: None,
                },
            )
            .unwrap();
        assert!(on.ends_with("<|im_start|>assistant"));
    }

    #[test]
    fn raise_exception_surfaces_as_an_error() {
        let template =
            ChatTemplate::new("{{ raise_exception('no system role') }}".to_string()).unwrap();
        let error = template
            .render(&serde_json::json!([]), &RenderOptions::default())
            .unwrap_err();
        assert!(error.to_string().contains("no system role"));
    }

    #[test]
    fn loads_template_and_tokens_from_tokenizer_config() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("tokenizer_config.json"),
            serde_json::json!({
                "chat_template": "{{ bos_token }}{{ messages[0].content }}{{ eos_token }}",
                "bos_token": null,
                "eos_token": {"content": "<|im_end|>"}
            })
            .to_string(),
        )
        .unwrap();
        let template = ChatTemplate::from_model_dir(dir.path()).unwrap();
        let text = template
            .render(
                &serde_json::json!([{"role": "user", "content": "x"}]),
                &RenderOptions::default(),
            )
            .unwrap();
        assert_eq!(text, "x<|im_end|>");
    }

    #[test]
    fn missing_template_names_the_directory() {
        let dir = tempfile::tempdir().unwrap();
        let error = ChatTemplate::from_model_dir(dir.path()).unwrap_err();
        assert!(matches!(error, TemplateError::Missing(_)));
    }
}
