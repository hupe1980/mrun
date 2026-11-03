//! Input caps, checked before any tokenization or engine work.

/// Largest prompt accepted, in bytes.
pub const MAX_PROMPT_BYTES: usize = 8 * 1024 * 1024;
/// Most stop sequences per request.
pub const MAX_STOPS: usize = 4;
/// Longest stop sequence, in bytes.
pub const MAX_STOP_BYTES: usize = 256;
