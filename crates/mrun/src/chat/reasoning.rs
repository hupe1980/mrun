//! Reasoning round-trip codecs (002 FR-016): the model's reasoning goes back to the client and
//! returns with the next request, so the model sees the same history.
//!
//! - Anthropic thinking blocks carry the text itself; their `signature` is a short digest that
//!   mrun never checks (any signature is accepted).
//! - Responses reasoning items carry the text in `encrypted_content`. It is not encrypted: the
//!   content stays on the user's machine and only needs to survive the client's round trip.

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use sha2::{Digest, Sha256};

const PREFIX: &str = "mrun1.";

/// Signature of an Anthropic thinking block: `mrun1.` + the first 16 base64 characters of
/// SHA-256 of the text.
pub fn signature(thinking: &str) -> String {
    let digest = STANDARD.encode(Sha256::digest(thinking.as_bytes()));
    format!("{PREFIX}{}", &digest[..16])
}

/// `encrypted_content` of a Responses reasoning item.
pub fn encode_content(reasoning: &str) -> String {
    format!("{PREFIX}{}", STANDARD.encode(reasoning))
}

/// Reasoning text from `encrypted_content`; `None` for content mrun did not produce (another
/// provider's), which is then dropped from the history.
pub fn decode_content(content: &str) -> Option<String> {
    let bytes = STANDARD.decode(content.strip_prefix(PREFIX)?).ok()?;
    String::from_utf8(bytes).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resp_encrypted_content_roundtrip_restores_think() {
        let text = "Check the <tool_call> tag … then edit.";
        assert_eq!(decode_content(&encode_content(text)).as_deref(), Some(text));
        assert_eq!(decode_content("gAAAAB-foreign"), None);
        assert_eq!(decode_content("mrun1.@@@"), None);
    }

    #[test]
    fn signatures_are_short_and_stable() {
        let a = signature("think");
        assert_eq!(a.len(), PREFIX.len() + 16);
        assert_eq!(a, signature("think"));
        assert_ne!(a, signature("thinks"));
    }
}
