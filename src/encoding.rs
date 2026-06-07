//! Decode raw source bytes into a `String`.
//!
//! PowerShell tooling on Windows often saves scripts as UTF-16, or as UTF-8
//! with a byte-order mark, so a parser that only accepts UTF-8 needs the bytes
//! decoded and the BOM removed first. These helpers do that using only the
//! standard library.

/// Remove a leading byte-order-mark character (`U+FEFF`) if present.
///
/// UTF-16 decoders leave the BOM as the first character of the string, and the
/// lexer would otherwise emit it as an unknown token at the start of the
/// script, so it is stripped before parsing. Only a leading BOM is removed; an
/// interior `U+FEFF` is a zero-width no-break space and belongs to the content.
pub fn strip_bom(s: &str) -> &str {
    s.strip_prefix('\u{feff}').unwrap_or(s)
}

#[derive(Clone, Copy)]
enum Endian {
    Little,
    Big,
}

fn decode_utf16(bytes: &[u8], endian: Endian) -> String {
    // A trailing odd byte cannot form a code unit; `chunks_exact` drops it so
    // that truncated input still decodes instead of failing outright.
    let units: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|c| match endian {
            Endian::Little => u16::from_le_bytes([c[0], c[1]]),
            Endian::Big => u16::from_be_bytes([c[0], c[1]]),
        })
        .collect();
    // Lossy so an unpaired surrogate becomes U+FFFD rather than rejecting the
    // whole file, which matters for adversarial or corrupted scripts.
    String::from_utf16_lossy(&units)
}

fn decode_inner(data: &[u8]) -> String {
    // A BOM is the only fully reliable encoding signal, so it is checked first.
    if let Some(rest) = data.strip_prefix(&[0xEF, 0xBB, 0xBF]) {
        return String::from_utf8_lossy(rest).into_owned();
    }
    // `FF FE` also begins the UTF-32LE BOM, which this decoder does not handle,
    // so that case is excluded rather than mis-read as UTF-16LE.
    if data.starts_with(&[0xFF, 0xFE]) && !data.starts_with(&[0xFF, 0xFE, 0x00, 0x00]) {
        return decode_utf16(&data[2..], Endian::Little);
    }
    if let Some(rest) = data.strip_prefix(&[0xFE, 0xFF]) {
        return decode_utf16(rest, Endian::Big);
    }
    // No BOM: prefer UTF-8, but a high proportion of NUL bytes means the data is
    // almost certainly UTF-16 that was saved without one. The count drives both
    // the UTF-8 acceptance test and the UTF-16 fallback, so do it once.
    let nul = data.iter().filter(|&&b| b == 0).count();
    if let Ok(s) = std::str::from_utf8(data) {
        if nul == 0 || nul <= data.len() / 8 {
            return s.to_owned();
        }
    }
    if data.len() >= 2 && nul > data.len() / 8 {
        // In ASCII-heavy UTF-16 the NULs sit on the high byte, so whether they
        // fall on even or odd offsets reveals the endianness.
        let even = data.iter().step_by(2).filter(|&&b| b == 0).count();
        let odd = data.iter().skip(1).step_by(2).filter(|&&b| b == 0).count();
        let endian = if even > odd {
            Endian::Big
        } else {
            Endian::Little
        };
        return decode_utf16(data, endian);
    }
    // Last resort: keep the valid parts as UTF-8 instead of returning nothing.
    String::from_utf8_lossy(data).into_owned()
}

/// Decode source bytes into a `String`, detecting UTF-8 and UTF-16 (LE/BE) by
/// BOM, with a NUL-byte heuristic for BOM-less UTF-16.
///
/// Invalid sequences are replaced with `U+FFFD` rather than rejected, and any
/// leading BOM is removed, so the result is ready to pass straight to the
/// lexer/parser (v1 or v2). Encodings other than UTF-8 and UTF-16 (for example
/// UTF-32) fall back to a lossy UTF-8 reading.
pub fn decode_bytes(data: &[u8]) -> String {
    let s = decode_inner(data);
    // `decode_inner` already drops the BOM in every branch that could carry one
    // (UTF-8 and both UTF-16 paths strip it). This is a backstop, and it avoids
    // re-allocating on the common no-BOM path by returning `s` by move.
    match s.strip_prefix('\u{feff}') {
        Some(rest) => rest.to_owned(),
        None => s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn utf16le(s: &str) -> Vec<u8> {
        s.encode_utf16().flat_map(u16::to_le_bytes).collect()
    }
    fn utf16be(s: &str) -> Vec<u8> {
        s.encode_utf16().flat_map(u16::to_be_bytes).collect()
    }

    #[test]
    fn strip_bom_removes_only_a_leading_bom() {
        assert_eq!(strip_bom("\u{feff}hi"), "hi");
        assert_eq!(strip_bom("hi"), "hi");
        assert_eq!(strip_bom("hi\u{feff}"), "hi\u{feff}");
    }

    #[test]
    fn plain_utf8_passes_through() {
        assert_eq!(decode_bytes(b"$x = 1"), "$x = 1");
    }

    #[test]
    fn utf8_bom_is_stripped() {
        let mut data = vec![0xEF, 0xBB, 0xBF];
        data.extend_from_slice(b"$x = 1");
        assert_eq!(decode_bytes(&data), "$x = 1");
    }

    #[test]
    fn utf16le_with_bom_decodes_without_bom() {
        let mut data = vec![0xFF, 0xFE];
        data.extend(utf16le("Write-Output 'hi \u{20ac}'"));
        let out = decode_bytes(&data);
        assert_eq!(out, "Write-Output 'hi \u{20ac}'");
        assert!(!out.starts_with('\u{feff}'));
    }

    #[test]
    fn utf16be_with_bom_decodes_without_bom() {
        let mut data = vec![0xFE, 0xFF];
        data.extend(utf16be("Get-ChildItem"));
        assert_eq!(decode_bytes(&data), "Get-ChildItem");
    }

    #[test]
    fn bomless_utf16le_detected_by_nul_pattern() {
        let data = utf16le("$path = 'C:\\temp'");
        assert_eq!(decode_bytes(&data), "$path = 'C:\\temp'");
    }
}
