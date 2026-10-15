//! Constrained decoding with llguidance (002 FR-004, FR-036): grammars for strict tool calls,
//! `tool_choice: required`/named and JSON response formats, in each family's own syntax.
//!
//! Nothing is constrained by default. A grammar arms either at the tool-call opener the model
//! writes (strict tools under `auto`), or from the start of the answer (required or named
//! calls, JSON formats); with thinking on, "the start" is the end of the reasoning block.
//! `tool_choice: none` needs no grammar: the opener token is banned with a fixed mask.

use crate::api::model::{ChatRequest, ResponseFormat, ToolChoice, ToolDef};
use crate::chat::tokenizer::Tokenizer;
use llguidance::api::TopLevelGrammar;
use llguidance::{Matcher, ParserFactory};
use mrun_engine::{SyntaxId, TokenMask};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokenizers::DecoderWrapper;
use toktrie::{TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv};

/// A model's grammar environment: the llguidance parser factory over its vocabulary.
pub struct GrammarEnv {
    factory: ParserFactory,
    vocab: u32,
    /// Added tokens by text (`<tool_call>`, `</think>`, ...).
    specials: HashMap<String, u32>,
}

impl std::fmt::Debug for GrammarEnv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GrammarEnv")
            .field("vocab", &self.vocab)
            .finish()
    }
}

impl GrammarEnv {
    /// Builds the environment for a byte-level BPE tokenizer (Qwen, gpt-oss).
    pub fn new(tokenizer: &Tokenizer, eos: &[u32]) -> Result<Self, String> {
        let hf = tokenizer.inner();
        if !matches!(hf.get_decoder(), Some(DecoderWrapper::ByteLevel(_))) {
            return Err("constrained decoding needs a byte-level tokenizer".into());
        }
        let vocab = hf.get_vocab_size(true) as u32;
        let mut eos: Vec<u32> = eos.iter().copied().filter(|&id| id < vocab).collect();
        for end in ["<|im_end|>", "<|return|>", "<|endoftext|>"] {
            if eos.is_empty()
                && let Some(id) = hf.token_to_id(end)
            {
                eos.push(id);
            }
        }
        let mut info = TokRxInfo::new(vocab, eos.first().copied().unwrap_or(0));
        let char_map = byte_level_chars();
        let mut specials = HashMap::new();
        let added: HashSet<u32> = hf.get_added_tokens_decoder().keys().copied().collect();
        let mut token_bytes = vec![Vec::new(); vocab as usize];
        for id in 0..vocab {
            let Some(name) = hf.id_to_token(id) else {
                continue;
            };
            token_bytes[id as usize] = if added.contains(&id) {
                specials.insert(name.clone(), id);
                let mut bytes = name.into_bytes();
                bytes.insert(0, TokTrie::SPECIAL_TOKEN_MARKER);
                bytes
            } else {
                match name.chars().map(|c| char_map.get(&c).copied()).collect() {
                    Some(bytes) => bytes,
                    None => continue,
                }
            };
        }
        if let Some(&end) = specials.get("<|im_end|>") {
            info.tok_end_of_turn = Some(end);
        }
        let mut trie = TokTrie::from(&info, &token_bytes);
        if eos.len() > 1 {
            trie = trie.with_eos_tokens(&eos);
        }
        let env: TokEnv = Arc::new(TrieEnv {
            trie,
            tokenizer: tokenizer.clone(),
        });
        let factory = ParserFactory::new_simple(&env).map_err(|e| e.to_string())?;
        Ok(Self {
            factory,
            vocab,
            specials,
        })
    }

    pub fn special(&self, text: &str) -> Option<u32> {
        self.specials.get(text).copied()
    }

    pub fn vocab(&self) -> u32 {
        self.vocab
    }

    /// Compiles a Lark grammar into a matcher; `Err` is logged `strict-unsupported`.
    pub fn matcher(&self, lark: &str) -> Result<Matcher, String> {
        let parser = self
            .factory
            .create_parser(TopLevelGrammar::from_lark(lark.to_string()))
            .map_err(|e| e.to_string())?;
        let matcher = Matcher::new(Ok(parser));
        match matcher.get_error() {
            Some(error) => Err(error),
            None => Ok(matcher),
        }
    }
}

struct TrieEnv {
    trie: TokTrie,
    tokenizer: Tokenizer,
}

impl TokenizerEnv for TrieEnv {
    fn tok_trie(&self) -> &TokTrie {
        &self.trie
    }

    fn tokenize_bytes(&self, bytes: &[u8]) -> Vec<TokenId> {
        self.trie
            .tokenize_with_greedy_fallback(bytes, |s| self.tokenizer.encode(s).unwrap_or_default())
    }
}

/// The GPT-2 byte-level alphabet: printable bytes map to themselves, the rest to U+0100+.
fn byte_level_chars() -> HashMap<char, u8> {
    let mut map = HashMap::new();
    let mut next = 0x100u32;
    for byte in 0..=255u8 {
        let c = byte as char;
        if matches!(c, '!'..='~' | '\u{A1}'..='\u{AC}' | '\u{AE}'..='\u{FF}') {
            map.insert(c, byte);
        } else {
            map.insert(char::from_u32(next).expect("valid char"), byte);
            next += 1;
        }
    }
    map
}

/// What a request needs from the decode hook.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Plan {
    /// Grammar of one call after its opener token (strict tools under `auto`).
    pub on_opener: Option<String>,
    /// Grammar of the whole answer (required/named calls, JSON formats).
    pub answer: Option<String>,
    /// Ban the tool-call opener (`tool_choice: none`).
    pub ban_opener: bool,
}

/// Opener token of a tool call per syntax (Harmony has none: calls follow a header recipient).
pub fn opener(syntax: SyntaxId) -> Option<&'static str> {
    match syntax {
        SyntaxId::Hermes | SyntaxId::QwenXml => Some("<tool_call>"),
        SyntaxId::Harmony => None,
    }
}

/// Lark string literal.
fn lit(text: &str) -> String {
    serde_json::to_string(text).expect("strings serialize")
}

/// Plans constraints for `request`; grammar text refers to special tokens by id.
pub fn plan(request: &ChatRequest, syntax: SyntaxId, env: &GrammarEnv) -> Plan {
    let mut plan = Plan::default();
    let close = env.special("</tool_call>");
    let open = opener(syntax).and_then(|o| env.special(o));
    match (&request.tool_choice, open, close) {
        (ToolChoice::None, _, _) => plan.ban_opener = open.is_some(),
        (ToolChoice::Required | ToolChoice::Named(_), Some(open), Some(close)) => {
            let tools: Vec<&ToolDef> = match &request.tool_choice {
                ToolChoice::Named(name) => {
                    request.tools.iter().filter(|t| &t.name == name).collect()
                }
                _ => request.tools.iter().collect(),
            };
            if let Some(body) = call_body(syntax, &tools, close) {
                let calls = if request.parallel_tool_calls {
                    "call (\"\\n\" call)*".to_string()
                } else {
                    "call".to_string()
                };
                plan.answer = Some(format!("start: {calls}\ncall: <[{open}]> body\n{body}"));
            }
        }
        (ToolChoice::Auto, Some(_), Some(close)) if request.tools.iter().any(|t| t.strict) => {
            let tools: Vec<&ToolDef> = request.tools.iter().collect();
            if let Some(body) = call_body(syntax, &tools, close) {
                plan.on_opener = Some(format!("start: body\n{body}"));
            }
        }
        _ => {}
    }
    if plan.answer.is_none() {
        let schema = match &request.response_format {
            ResponseFormat::Text => None,
            ResponseFormat::JsonObject => Some(json!({"type": "object"})),
            ResponseFormat::JsonSchema { schema, strict, .. } => strict.then(|| schema.clone()),
        };
        if let Some(schema) = schema {
            plan.answer = Some(format!("start: %json{schema}"));
        }
    }
    plan
}

/// Rules `body` (one call after its opener, through the closer) and helpers, or `None` when no
/// grammar exists for the syntax.
fn call_body(syntax: SyntaxId, tools: &[&ToolDef], close: u32) -> Option<String> {
    if tools.is_empty() {
        return None;
    }
    let mut rules = String::new();
    let mut alts = Vec::new();
    for (i, tool) in tools.iter().enumerate() {
        let schema = if tool.strict {
            tool.schema.clone()
        } else {
            json!({"type": "object"})
        };
        match syntax {
            SyntaxId::Hermes => {
                let head = format!("{{\"name\": {}, \"arguments\": ", lit(&tool.name));
                rules.push_str(&format!("c{i}: {} %json{schema} \"}}\"\n", lit(&head)));
            }
            SyntaxId::QwenXml => {
                let params = xml_params(i, tool, &schema, &mut rules);
                rules.push_str(&format!(
                    "c{i}: {} {params}{}\n",
                    lit(&format!("<function={}>\n", tool.name)),
                    lit("</function>")
                ));
            }
            SyntaxId::Harmony => return None,
        }
        alts.push(format!("c{i}"));
    }
    Some(format!(
        "body: \"\\n\" ({}) \"\\n\" <[{close}]>\n{rules}",
        alts.join(" | ")
    ))
}

/// Qwen XML parameters in declared order: required ones always, optional ones may be skipped.
/// String values run until `\n</parameter>\n`; other types are JSON of their schema.
fn xml_params(i: usize, tool: &ToolDef, schema: &Value, rules: &mut String) -> String {
    let empty = serde_json::Map::new();
    let properties = schema["properties"].as_object().unwrap_or(&empty);
    let required: HashSet<&str> = schema["required"]
        .as_array()
        .map(|r| r.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let mut parts = String::new();
    for (k, (name, prop)) in properties.iter().enumerate() {
        let rule = format!("p{i}_{k}");
        let open = lit(&format!("<parameter={name}>\n"));
        let is_string = !tool.strict || matches!(prop["type"].as_str(), None | Some("string"));
        if is_string {
            rules.push_str(&format!(
                "{rule}: {open} v{i}_{k}\nv{i}_{k}[suffix=\"\\n</parameter>\\n\"]: /(.|\\n)*/\n"
            ));
        } else {
            rules.push_str(&format!(
                "{rule}: {open} %json{prop} \"\\n</parameter>\\n\"\n"
            ));
        }
        if required.contains(name.as_str()) {
            parts.push_str(&format!("{rule} "));
        } else {
            parts.push_str(&format!("{rule}? "));
        }
    }
    parts
}

/// The fixed mask for `tool_choice: none`: everything but the opener.
pub fn ban(vocab: u32, token: u32) -> TokenMask {
    let words: Vec<u32> = (0..vocab.div_ceil(32))
        .map(|w| {
            if w == token / 32 {
                !(1 << (token % 32))
            } else {
                u32::MAX
            }
        })
        .collect();
    TokenMask::from_words(&words, vocab)
}

/// A matcher's mask as the engine's `TokenMask`.
pub fn mask_of(matcher: &mut Matcher, vocab: u32) -> Option<TokenMask> {
    let vob = matcher.compute_mask().ok()?;
    Some(TokenMask::from_words(vob.as_slice(), vocab))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::model::ApiKind;
    use crate::chat::tokenizer::test_support::byte_tokenizer;

    fn env() -> (GrammarEnv, Tokenizer) {
        let tokenizer = byte_tokenizer();
        (GrammarEnv::new(&tokenizer, &[]).unwrap(), tokenizer)
    }

    fn read_tool(strict: bool) -> ToolDef {
        ToolDef {
            name: "read".into(),
            description: String::new(),
            schema: json!({"type": "object", "properties": {"path": {"type": "string"},
                "limit": {"type": "integer"}}, "required": ["path"], "additionalProperties": false}),
            strict,
        }
    }

    fn request(choice: ToolChoice, strict: bool) -> ChatRequest {
        let mut request = ChatRequest::new(ApiKind::Chat);
        request.tools = vec![read_tool(strict)];
        request.tool_choice = choice;
        request
    }

    /// Feeds `text` token by token, appending forced tokens like the decode loop; returns the
    /// text of forced tokens, or the first rejected position.
    fn drive(matcher: &mut Matcher, tokenizer: &Tokenizer, text: &str) -> Result<String, usize> {
        let ids = tokenizer.encode(text).unwrap();
        let mut forced_text = String::new();
        let mut i = 0;
        while i < ids.len() {
            let forced = matcher.compute_ff_tokens();
            if !forced.is_empty() {
                for id in forced {
                    assert_eq!(ids.get(i), Some(&id), "forced token differs at {i}");
                    matcher.consume_token(id).unwrap();
                    forced_text.push_str(&tokenizer.inner().decode(&[id], false).unwrap());
                    i += 1;
                }
                continue;
            }
            let mask = matcher.compute_mask().unwrap();
            if !mask.is_allowed(ids[i]) {
                return Err(i);
            }
            matcher.consume_token(ids[i]).unwrap();
            i += 1;
        }
        Ok(forced_text)
    }

    #[test]
    fn hermes_strict_call_body_accepts_valid_and_rejects_invalid_args() {
        let (env, tokenizer) = env();
        let plan = plan(&request(ToolChoice::Auto, true), SyntaxId::Hermes, &env);
        assert!(plan.answer.is_none() && !plan.ban_opener);
        let lark = plan.on_opener.unwrap();
        let mut matcher = env.matcher(&lark).unwrap();
        drive(
            &mut matcher,
            &tokenizer,
            "\n{\"name\": \"read\", \"arguments\": {\"path\":\"a.rs\"}}\n</tool_call>",
        )
        .unwrap();
        assert!(matcher.is_accepting().unwrap());
        let mut matcher = env.matcher(&lark).unwrap();
        let bad = "\n{\"name\": \"read\", \"arguments\": {\"path\":1}}\n</tool_call>";
        assert!(drive(&mut matcher, &tokenizer, bad).is_err());
    }

    #[test]
    fn xapi_forced_tokens_not_sampled() {
        // `required` with one tool: the opener, the envelope and the only key are forced.
        let (env, tokenizer) = env();
        let plan = plan(&request(ToolChoice::Required, true), SyntaxId::Hermes, &env);
        let mut matcher = env.matcher(&plan.answer.unwrap()).unwrap();
        let forced = drive(
            &mut matcher,
            &tokenizer,
            "<tool_call>\n{\"name\": \"read\", \"arguments\": {\"path\":\"a.rs\"}}\n</tool_call>",
        )
        .unwrap();
        // Forced runs are not contiguous: the envelope up to the arguments, the only required
        // key, and the closing; only the value is sampled.
        assert!(
            forced.starts_with("<tool_call>\n{\"name\": \"read\", \"arguments\": {"),
            "{forced:?}"
        );
        assert!(forced.contains("path"), "{forced:?}");
        assert!(forced.ends_with("}\n</tool_call>"), "{forced:?}");
    }

    #[test]
    fn qwen_xml_strict_params_typed_and_strings_free() {
        let (env, tokenizer) = env();
        let plan = plan(&request(ToolChoice::Auto, true), SyntaxId::QwenXml, &env);
        let lark = plan.on_opener.unwrap();
        let mut matcher = env.matcher(&lark).unwrap();
        let call = "\n<function=read>\n<parameter=path>\nsrc/a </parameter> b.rs\n</parameter>\n<parameter=limit>\n10\n</parameter>\n</function>\n</tool_call>";
        drive(&mut matcher, &tokenizer, call).unwrap();
        let mut matcher = env.matcher(&lark).unwrap();
        let bad = "\n<function=read>\n<parameter=path>\na\n</parameter>\n<parameter=limit>\nten\n</parameter>\n</function>\n</tool_call>";
        assert!(drive(&mut matcher, &tokenizer, bad).is_err());
    }

    #[test]
    fn none_bans_only_the_opener_and_formats_constrain_the_answer() {
        let (env, _) = env();
        let plan_none = plan(&request(ToolChoice::None, false), SyntaxId::Hermes, &env);
        assert!(plan_none.ban_opener && plan_none.on_opener.is_none());
        let opener = env.special("<tool_call>").unwrap();
        let mask = ban(env.vocab(), opener);
        assert!(!mask.is_allowed(opener) && mask.is_allowed(opener - 1) && mask.is_allowed(0));

        let mut request = ChatRequest::new(ApiKind::Chat);
        request.response_format = ResponseFormat::JsonObject;
        let plan = plan(&request, SyntaxId::Hermes, &env);
        assert!(env.matcher(&plan.answer.unwrap()).is_ok());
    }

    #[test]
    fn xapi_strict_schema_uncompilable_unconstrained_logged() {
        let (env, _) = env();
        let mut request = request(ToolChoice::Auto, true);
        request.tools[0].schema =
            json!({"type": "object", "properties": {"x": {"$ref": "#/nope"}}});
        let plan = plan(&request, SyntaxId::Hermes, &env);
        assert!(env.matcher(&plan.on_opener.unwrap()).is_err());
    }
}
