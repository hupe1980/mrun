//! Chat templates from the model's own files, rendered with MiniJinja (Python-compatible methods).

use minijinja::value::Kwargs;
use minijinja::{Environment, Error, ErrorKind, Value};
use serde::Serialize;
use std::fs;
use std::path::Path;
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

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
    /// Harmony (gpt-oss) `low`, `medium` or `high`; left undefined when `None`, so the
    /// template's default applies.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    /// Qwen3.6: render reasoning of every replayed assistant turn, not only the current loop.
    pub preserve_thinking: bool,
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
        // Match Hugging Face's Jinja environment: block tags eat their trailing newline and
        // leading indentation, and `tojson` is Python's `json.dumps` (no HTML escaping).
        let mut env = Environment::new();
        env.set_syntax(
            minijinja::syntax::SyntaxConfig::builder()
                .trim_blocks(true)
                .lstrip_blocks(true)
                .build()
                .map_err(|error| TemplateError::Render(error.to_string()))?,
        );
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_filter("tojson", tojson);
        env.add_function(
            "raise_exception",
            |message: String| -> Result<Value, Error> {
                Err(Error::new(ErrorKind::InvalidOperation, message))
            },
        );
        env.add_template_owned("chat", source)
            .map_err(|error| TemplateError::Render(error.to_string()))?;
        let mut template = Self {
            env,
            bos_token,
            eos_token,
        };
        template.set_date(Date::server_start());
        Ok(template)
    }

    /// Fixes the date `strftime_now` reports (the template probe uses a constant one).
    pub fn set_date(&mut self, date: Date) {
        self.env
            .add_function("strftime_now", move |format: String| date.format(&format));
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
                preserve_thinking => options.preserve_thinking,
                reasoning_effort => options
                    .reasoning_effort
                    .as_deref()
                    .map_or(Value::UNDEFINED, Value::from),
                bos_token => &self.bos_token,
                eos_token => &self.eos_token,
            })
            .map_err(|error| TemplateError::Render(format!("{error:#}")))
    }
}

/// A UTC calendar date, as `strftime_now` sees it (time of day is always midnight).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Date {
    /// Days since 1970-01-01.
    days: i64,
}

impl Date {
    const MONTHS: [&str; 12] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    const WEEKDAYS: [&str; 7] = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ];

    /// `month` and `day` are 1-based.
    pub fn from_ymd(year: i32, month: u32, day: u32) -> Self {
        // Days from civil (Howard Hinnant), with March as the first month of the year.
        let year = i64::from(year) - i64::from(month <= 2);
        let era = year.div_euclid(400);
        let year_of_era = year - era * 400;
        let shifted_month = i64::from((month + 9) % 12);
        let day_of_year = (153 * shifted_month + 2) / 5 + i64::from(day) - 1;
        let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
        Self {
            days: era * 146_097 + day_of_era - 719_468,
        }
    }

    /// The UTC date of a Unix timestamp.
    pub fn from_unix_secs(secs: u64) -> Self {
        Self {
            days: (secs / 86_400) as i64,
        }
    }

    /// The UTC date when this process first asked, fixed for its lifetime.
    pub fn server_start() -> Self {
        static START: OnceLock<Date> = OnceLock::new();
        *START.get_or_init(|| {
            let secs = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |elapsed| elapsed.as_secs());
            Self {
                days: (secs / 86_400) as i64,
            }
        })
    }

    /// `(year, month, day)` with 1-based month and day.
    pub fn ymd(self) -> (i32, u32, u32) {
        let z = self.days + 719_468;
        let era = z.div_euclid(146_097);
        let day_of_era = z - era * 146_097;
        let year_of_era =
            (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
        let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
        let shifted_month = (5 * day_of_year + 2) / 153;
        let day = (day_of_year - (153 * shifted_month + 2) / 5 + 1) as u32;
        let month = if shifted_month < 10 {
            shifted_month + 3
        } else {
            shifted_month - 9
        } as u32;
        let year = year_of_era + era * 400 + i64::from(month <= 2);
        (year as i32, month, day)
    }

    /// Python `strftime` for the date directives; time directives format midnight.
    pub fn format(self, format: &str) -> String {
        let (year, month, day) = self.ymd();
        let month_name = Self::MONTHS[month as usize - 1];
        // 1970-01-01 was a Thursday.
        let weekday = Self::WEEKDAYS[(self.days + 3).rem_euclid(7) as usize];
        let day_of_year = self.days - Self::from_ymd(year, 1, 1).days + 1;
        let mut out = String::new();
        let mut chars = format.chars();
        while let Some(ch) = chars.next() {
            if ch != '%' {
                out.push(ch);
                continue;
            }
            match chars.next() {
                Some('Y') => out.push_str(&year.to_string()),
                Some('y') => out.push_str(&format!("{:02}", year.rem_euclid(100))),
                Some('m') => out.push_str(&format!("{month:02}")),
                Some('d') => out.push_str(&format!("{day:02}")),
                Some('e') => out.push_str(&format!("{day:2}")),
                Some('j') => out.push_str(&format!("{day_of_year:03}")),
                Some('B') => out.push_str(month_name),
                Some('b' | 'h') => out.push_str(&month_name[..3]),
                Some('A') => out.push_str(weekday),
                Some('a') => out.push_str(&weekday[..3]),
                Some('H' | 'M' | 'S') => out.push_str("00"),
                Some('I') => out.push_str("12"),
                Some('p') => out.push_str("AM"),
                Some('%') => out.push('%'),
                Some(other) => {
                    out.push('%');
                    out.push(other);
                }
                None => out.push('%'),
            }
        }
        out
    }
}

/// `json.dumps(value, ensure_ascii=False, indent=indent)` as Hugging Face templates expect.
fn tojson(value: Value, kwargs: Kwargs) -> Result<Value, Error> {
    let indent: Option<usize> = kwargs.get("indent")?;
    kwargs.assert_all_used()?;
    let json = serde_json::to_value(&value)
        .map_err(|error| Error::new(ErrorKind::InvalidOperation, error.to_string()))?;
    let mut out = String::new();
    python_dumps(&json, indent, 0, &mut out);
    Ok(Value::from_safe_string(out))
}

fn python_dumps(value: &serde_json::Value, indent: Option<usize>, depth: usize, out: &mut String) {
    use serde_json::Value as Json;
    let newline = |out: &mut String, depth: usize| {
        if let Some(width) = indent {
            out.push('\n');
            out.extend(std::iter::repeat_n(' ', width * depth));
        }
    };
    // Python separators: ", " without indent, "," with indent (the newline follows).
    let item_separator = if indent.is_some() { "," } else { ", " };
    match value {
        Json::Array(items) if !items.is_empty() => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(item_separator);
                }
                newline(out, depth + 1);
                python_dumps(item, indent, depth + 1, out);
            }
            newline(out, depth);
            out.push(']');
        }
        Json::Object(map) if !map.is_empty() => {
            out.push('{');
            for (index, (key, item)) in map.iter().enumerate() {
                if index > 0 {
                    out.push_str(item_separator);
                }
                newline(out, depth + 1);
                out.push_str(&serde_json::to_string(key).unwrap_or_default());
                out.push_str(": ");
                python_dumps(item, indent, depth + 1, out);
            }
            newline(out, depth);
            out.push('}');
        }
        other => out.push_str(&serde_json::to_string(other).unwrap_or_default()),
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
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(on.ends_with("<|im_start|>assistant"));
    }

    #[test]
    fn reasoning_effort_is_passed_only_when_set() {
        let template = ChatTemplate::new(
            "{%- if reasoning_effort is not defined %}{%- set reasoning_effort = 'medium' %}\
             {%- endif %}Reasoning: {{ reasoning_effort }}"
                .into(),
        )
        .unwrap();
        let render = |effort: Option<&str>| {
            let options = RenderOptions {
                reasoning_effort: effort.map(str::to_string),
                ..RenderOptions::default()
            };
            template.render(&serde_json::json!([]), &options).unwrap()
        };
        assert_eq!(render(None), "Reasoning: medium");
        assert_eq!(render(Some("high")), "Reasoning: high");
    }

    #[test]
    fn tojson_matches_python_json_dumps() {
        let template =
            ChatTemplate::new("{{ tools | tojson }}|{{ tools[0] | tojson(indent=2) }}".to_string())
                .unwrap();
        let tools = serde_json::json!([{"name": "grep", "description": "a < b & c's", "n": [1, 2.5], "e": {}}]);
        let text = template
            .render(
                &serde_json::json!([]),
                &RenderOptions {
                    tools: Some(tools),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(
            text,
            "[{\"name\": \"grep\", \"description\": \"a < b & c's\", \"n\": [1, 2.5], \"e\": {}}]|{\n  \"name\": \"grep\",\n  \"description\": \"a < b & c's\",\n  \"n\": [\n    1,\n    2.5\n  ],\n  \"e\": {}\n}"
        );
    }

    #[test]
    fn block_tags_trim_like_hugging_face() {
        let template = ChatTemplate::new(
            "{% for m in messages %}\n  {% if m %}{{ m.content }}{% endif %}\n{% endfor %}"
                .to_string(),
        )
        .unwrap();
        let text = template
            .render(
                &serde_json::json!([{"content": "a"}, {"content": "b"}]),
                &RenderOptions::default(),
            )
            .unwrap();
        assert_eq!(text, "ab", "Jinja trim_blocks + lstrip_blocks semantics");
    }

    #[test]
    fn strftime_now_formats_the_fixed_date_like_python() {
        let source = "{{ strftime_now('%d %b %Y') }}|{{ strftime_now('%Y-%m-%d %A %j %y %e %B %a %H:%M %%') }}";
        let mut template = ChatTemplate::new(source.to_string()).unwrap();
        template.set_date(Date::from_ymd(2026, 3, 5));
        let text = template
            .render(&serde_json::json!([]), &RenderOptions::default())
            .unwrap();
        assert_eq!(
            text,
            "05 Mar 2026|2026-03-05 Thursday 064 26  5 March Thu 00:00 %"
        );
    }

    #[test]
    fn dates_round_trip_through_days() {
        for (ymd, days) in [
            ((1970, 1, 1), 0),
            ((2000, 2, 29), 11_016),
            ((1969, 12, 31), -1),
            ((2026, 10, 15), 20_741),
        ] {
            let date = Date::from_ymd(ymd.0, ymd.1, ymd.2);
            assert_eq!(date.days, days, "{ymd:?}");
            assert_eq!(date.ymd(), ymd);
        }
        assert_eq!(Date::from_ymd(2024, 12, 31).format("%j %a"), "366 Tue");
    }

    #[test]
    fn strftime_now_defaults_to_the_server_start_date() {
        let template = ChatTemplate::new("{{ strftime_now('%Y-%m-%d') }}".to_string()).unwrap();
        let text = template
            .render(&serde_json::json!([]), &RenderOptions::default())
            .unwrap();
        assert_eq!(text, Date::server_start().format("%Y-%m-%d"));
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
